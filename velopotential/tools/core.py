import numpy as np
from anndata import AnnData
from sklearn.metrics.pairwise import cosine_similarity
from velopotential.tools.model import Model

def add_velocity_noise(
    adata: AnnData,
    layer_key: str = "velocity",
    noise_level: float = 0.2,
    new_layer_key: str = None,
    seed: int = None,
) -> np.ndarray:
    """
    Apply sign-flipping noise to a specific velocity layer in an AnnData object.
    Parameters
    ----------
    adata
        AnnData object containing single-cell data.
    layer_key
        The key of the layer to corrupt (e.g., "velocity").
    noise_level
        Noise intensity ranging from 0.0 to 1.0.
        - 0.0: No noise applied.
        - 1.0: Maximum randomization (50% probability of sign flip, resulting in zero information).
    new_layer_key
        Key for the output layer. If None, defaults to f"{layer_key}_noisy".
    seed
        Random seed for reproducibility.
    Returns
    -------
    V_noisy
        The corrupted velocity matrix as a numpy array.
    """
    if new_layer_key is None:
        new_layer_key = f"{layer_key}_noisy"
    if layer_key not in adata.layers:
        raise ValueError(f"Layer '{layer_key}' not found in adata.")
        
    V_true = adata.layers[layer_key]
    
    V_true = V_true.A if hasattr(V_true, "A") else np.asarray(V_true, dtype=float)
    V_noisy = V_true.copy()
    if noise_level >= 0:
        flip_prob = noise_level / 2.0
        
        if seed is not None:
            np.random.seed(seed)
            
        flip_mask = np.random.rand(*V_noisy.shape) < flip_prob
        V_noisy[flip_mask] *= -1.0

    adata.layers[new_layer_key] = V_noisy
    print(f"Added noise (level={noise_level}, flip_prob={flip_prob:.2%}) to '{new_layer_key}'.")
    
    return V_noisy

def construct_potential(adata, model_path=None,input_layer="Ms",velocity_layer="velocity",sign=True,lambda_j=1e-6):
    if model_path == None:
        model = Model(adata,input_layer=input_layer,velocity_layer=velocity_layer,sign=sign,lambda_j=lambda_j)
        model.train(max_epochs=500)
    else:
        model = Model(adata,input_layer=input_layer)
    
    adata.obs['potential'] = model.get_J()
    adata.obs['potential'] = (adata.obs['potential'] - adata.obs['potential'].min()) / (adata.obs['potential'].max() - adata.obs['potential'].min())

    adata.layers['velocity_pred'] = model.get_v_pred()


def cal_cosine_similarity(layer_v1, layer_v2):
    cos_sims = np.array([cosine_similarity(np.nan_to_num(v1).reshape(1, -1), np.nan_to_num(v2).reshape(1, -1))[0][0] 
                        for v1, v2 in zip(layer_v1, layer_v2)])
    return cos_sims
