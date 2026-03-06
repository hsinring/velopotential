from __future__ import annotations

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from scipy import sparse
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hid: int = 128,
        n_layers: int = 2,
        drop: float = 0.0,
    ):
        super().__init__()

        layers = [nn.Linear(n_in, n_hid), nn.GELU(), nn.Dropout(drop)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hid, n_hid), nn.GELU(), nn.Dropout(drop)]
        layers += [nn.Linear(n_hid, 1, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out


class EarlyStop:
    def __init__(self, patience: int = 50, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")

    def __call__(self, loss: float) -> bool:
        if loss < self.best - self.min_delta:
            self.best = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        wd: float = 1e-3,
        patience: int = 20,
        min_delta: float = 1e-4,
        lambda_j: float = 1e-6,
    ):
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.early_stop = EarlyStop(patience=patience, min_delta=min_delta)
        self.lambda_j = lambda_j

    def train_epoch(self, X_batch: torch.Tensor, V_batch: torch.Tensor) -> dict:

        self.model.train()
        self.opt.zero_grad()

        X = X_batch.detach().clone().requires_grad_(True)
        V = V_batch

        J = self.model(X)

        grad_x = torch.autograd.grad(
            outputs=J.sum(),
            inputs=X,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        v_pred = -grad_x

        v_norm_sample = V.norm(dim=1)            
        sample_mask = v_norm_sample > 1e-6       

        if sample_mask.any():
            v_pred_s = v_pred[sample_mask]       
            V_s = V[sample_mask]                

           
            v_norm_feature = V_s.norm(dim=0)     
            feature_mask = v_norm_feature > 1e-6 


            cos_sim_dim1 = F.cosine_similarity(v_pred_s, V_s, dim=1)  # (M,)
            mean_cos_dim1 = cos_sim_dim1.mean()


            if feature_mask.any():
                v_pred_sf = v_pred_s[:, feature_mask]  # (M, D_valid)
                V_sf = V_s[:, feature_mask]           # (M, D_valid)

                cos_sim_dim0 = F.cosine_similarity(v_pred_sf, V_sf, dim=0)  # (D_valid,)
                mean_cos_dim0 = cos_sim_dim0.mean()
            else:

                mean_cos_dim0 = 0.0


            cos_sim = mean_cos_dim1 + mean_cos_dim0
            vel_loss = (2.0 - cos_sim)/2

            J_centered = J - J.mean()
            j_reg = (J_centered ** 2).mean()
            loss = vel_loss + self.lambda_j * j_reg

            loss.backward()
            self.opt.step()
            loss_value = float(vel_loss.detach().cpu().item())

        else:
            loss_value = 0.0

        return {"loss": loss_value}


class Model:
    def __init__(
        self,
        adata: AnnData,
        input_layer: str = "Ms",
        velocity_layer: str = "velocity",
        seed: int = 42,
        model_path: str | None = None,
        sign = True,
        lambda_j: float = 1e-6,
        **kw,
    ):
        set_seed(seed=seed)

        self.n_genes = adata.n_vars
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_j = lambda_j

        if model_path is None:
            self.model = Encoder(self.n_genes, **kw).to(self.device)
        else:
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)

        if input_layer is None:
            X = adata.X.copy()
        else:
            X = adata.layers[input_layer].copy()
        if sparse.issparse(X):
            X = X.toarray()

        V = adata.layers[velocity_layer].copy()
        if sparse.issparse(V):
            V = V.toarray()
        V[np.isnan(V)] = 0

        if sign:
            V = np.sign(V)

        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.V = torch.tensor(V, dtype=torch.float32, device=self.device)

    def train(
        self,
        max_epochs: int = 200,
        batch_size: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        patience: int = 40,
        min_delta: float = 1e-4,
    ):
        trainer = Trainer(
            self.model,
            lr=lr,
            wd=weight_decay,
            patience=patience,
            min_delta=min_delta,
            lambda_j=self.lambda_j,
        )

        dataset = TensorDataset(self.X, self.V)
        batch_size = batch_size or min(256, len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        for epoch in range(max_epochs):
            epoch_loss = []
            for x, v in loader:
                log = trainer.train_epoch(x, v)
                epoch_loss.append(log["loss"])
            avg_loss = float(sum(epoch_loss) / len(epoch_loss)) if epoch_loss else 0.0

            if epoch % 50 == 0 or epoch == max_epochs - 1:
                print(f"epoch {epoch+1:3d}: loss={avg_loss:.6f}")

            if trainer.early_stop(avg_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break

    def get_J(self, batch_size: int | None = None) -> np.ndarray:
        
        self.model.eval()
        X_all = self.X
        N, _ = X_all.shape
        batch_size = batch_size or min(512, N)

        split_J = []
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                X = X_all[start:end]
                J_ = self.model(X)
                split_J.append(J_.cpu())

        J_all = torch.cat(split_J, dim=0).numpy()
        return J_all

    def get_v_pred(self, batch_size: int | None = None) -> np.ndarray:
 
        self.model.eval()
        X_all = self.X
        N = X_all.shape[0]
        batch_size = batch_size or min(256, N)
        v_list = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X = X_all[start:end].detach().clone().requires_grad_(True)
            J = self.model(X)
            grad_x = torch.autograd.grad(
                outputs=J.sum(),
                inputs=X,
                create_graph=False,
                retain_graph=False,
            )[0]
            v_pred = -grad_x  # [B, n_genes]
            v_list.append(v_pred.detach().cpu())
        v_all = torch.cat(v_list, dim=0).numpy()

        mask = (self.V == 0).cpu().numpy()
        v_all[mask] = 0

        return v_all