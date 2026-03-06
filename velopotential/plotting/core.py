import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from anndata import AnnData
from typing import Optional, Tuple
import scvelo as scv
import pandas as pd

def velocity_graph(*a, **kw):
    """
    Wrapper function for scvelo's velocity_graph computation.
    """
    import warnings

    warnings.filterwarnings("ignore")

    import builtins

    _real_print = builtins.print
    builtins.print = lambda *_, **__: None
    try:
        return scv.tl.velocity_graph(*a, **kw)
    finally:
        builtins.print = _real_print
        print("computing velocity graph\nfinished.")

def velocity_embedding_stream(*a, **kw):
    """
    Wrapper function for scvelo's velocity_embedding_stream visualization.
    """
    import warnings

    warnings.filterwarnings("ignore")

    import builtins

    _real_print = builtins.print
    builtins.print = lambda *_, **__: None
    try:
        return scv.pl.velocity_embedding_stream(*a, **kw)
    finally:
        builtins.print = _real_print
        print("computing velocity embedding\nfinished.")

def plot_velocity_projection(
    adata: AnnData,
    vkey: str = "velocity",
    basis: str = "umap",
    xkey: Optional[str] = None,
    color: Optional[str] = None,
    legend_loc: str = "on data",
    title: str = "",
    show: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    size: Optional[int] = None,
    cmap: Optional[str] = None,
    alpha: float = 0.3,
    colorbar: bool = True,
    palette: Optional[str] = None,
    n_jobs: int = 10,
    graph_T: bool = False,
) -> None:
    """
    Computes a velocity graph from the calculated velocity vectors and projects it as a stream plot on the embedding.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with velocities data and embedding coordinates.
    vkey : str, optional
        Layer containing velocities data. Default is 'velocity'.
    basis : str, optional
        Embedding basis to use (e.g., 'umap', 'tsne'). Default is 'umap'.
    xkey : str, optional
        Layer to use as expression data.
        If None, uses adata.X converted to dense array. Default is None.
    color : str, optional
        Column name in obs for coloring points. If None, no coloring is applied.
        Default is None.
    legend_loc : str, optional
        Location of legend. Default is 'on data'.
    title : str, optional
        Plot title. Default is empty string.
    show : bool, optional
        Whether to display the plot. Default is True.
    figsize : tuple of int, optional
        Figure size (width, height) in inches. Default is (8, 6).
    size : int, optional
        Size of points in scatter plot. If None, uses default. Default is None.
    cmap : str, optional
        Colormap for coloring. If None, uses default. Default is None.
    alpha : float, optional
        Transparency of points. Default is 0.3.
    colorbar : bool, optional
        Whether to show colorbar. Default is True.
    palette : str, optional
        Color palette for categorical coloring. If None, uses default.
        Default is None.
    n_jobs : int, optional
        Number of jobs for parallel computation. Default is 10.
    graph_T : bool, optional
        If True, transpose the adjacency matrix. Default is False.

    Returns
    -------
    None
        Shows the velocity projection plot.

    Raises
    ------
    ValueError
        If required layers or embeddings are not found.
    """

    if vkey not in adata.layers:
        raise ValueError(f"Layer '{vkey}' not found in adata.layers")

    embedding_key = f"X_{basis}" if not basis.startswith("X_") else basis
    if embedding_key not in adata.obsm:
        raise ValueError(
            f"Embedding coordinates '{embedding_key}' not found in adata.obsm"
        )

    if xkey is None:
        xkey = "X"
        if hasattr(adata.X, "toarray"):
            adata.layers["X"] = adata.X.toarray()
    elif xkey not in adata.layers:
        raise ValueError(f"Layer '{xkey}' not found in adata.layers")

    if color is not None and color not in adata.obs.columns:
        raise ValueError(f"Column '{color}' not found in adata.obs")

    velocity_graph(adata, vkey=vkey, xkey=xkey, n_jobs=n_jobs)

    graph_key = f"{vkey}_graph"
    if graph_key not in adata.uns:
        raise KeyError(graph_key)
    if graph_T is True:
        adata.uns[graph_key] = adata.uns[graph_key].T

    velocity_embedding_stream(
        adata,
        basis=basis,
        vkey=vkey,
        color=color,
        legend_loc=legend_loc,
        title=title,
        show=show,
        figsize=figsize,
        size=size,
        cmap=cmap,
        alpha=alpha,
        colorbar=colorbar,
        palette=palette,
    )

def scatter_from_adata(
    adata,
    x_key: str,
    y_key: str,
    color_by: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    s: float = 8.0,
    alpha: float = 0.4,
    figsize: Tuple[float, float] = (4.5,4),
    cmap: str = "magma",
    palette: str = "tab10",
    show: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fontsize: int = 15,
    ticksize: int = 14
) -> float:
    """
    Generate a scatter plot from an AnnData object.
    This function extracts coordinates and metadata from `adata.obs` to create a 
    scatter plot.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing `.obs` dataframe.
    x_key : str
        Key in `adata.obs` for the x-axis coordinates.
    y_key : str
        Key in `adata.obs` for the y-axis coordinates.
    color_by : str, optional
        Key in `adata.obs` to color points by. Supports both categorical 
        (discrete) and numerical (continuous) data.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to plot on. If None, a new figure is initialized.
    s : float, default 8.0
        Point size.
    alpha : float, default 0.8
        Point transparency.
    figsize : tuple, default (4,4)
        Figure dimensions in inches.
    cmap : str, default "magma"
        Colormap for continuous variables.
    palette : str, default "tab10"
        Color palette for categorical variables.
    show : bool, default True
        Whether to call `plt.show()` immediately.
    xlabel : str, optional
        Custom label for the x-axis. Defaults to `x_key`.
    ylabel : str, optional
        Custom label for the y-axis. Defaults to `y_key`.
    fontsize : int, default 10
        Font size for axis labels and statistics text.
    ticksize : int, default 9
        Font size for axis tick labels.
    
    Returns
    -------
    corr : float
        Pearson correlation coefficient.
    """

    if x_key not in adata.obs.columns or y_key not in adata.obs.columns:
        raise KeyError(f"Requested keys '{x_key}' or '{y_key}' missing in adata.obs")
    
    df = adata.obs[[x_key, y_key]].copy()

    if color_by:
        if color_by not in adata.obs.columns:
            raise KeyError(f"Color key '{color_by}' missing in adata.obs")
        df[color_by] = adata.obs[color_by]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if color_by is None:
        ax.scatter(df[x_key], df[y_key], s=s, alpha=alpha, 
                   c="#3D5A80", edgecolors="none")
    else:
        c_data = df[color_by]
    
        if pd.api.types.is_categorical_dtype(c_data) or c_data.dtype == "object":
            categories = pd.Categorical(c_data)
            unique_cats = categories.categories
            colors = plt.get_cmap(palette)(np.linspace(0, 1, len(unique_cats)))
        
            for i, cat in enumerate(unique_cats):
                mask = categories == cat
                ax.scatter(
                    df.loc[mask, x_key], df.loc[mask, y_key],
                    c=[colors[i]], label=cat,
                    s=s, alpha=alpha, edgecolors="none"
                )
        
            ax.legend(
                title=color_by,
                bbox_to_anchor=(1.05, 1), 
                loc='lower left',
                markerscale=1.5,
                title_fontproperties={'weight': 'bold', 'size': ticksize}
            )
        else:
            sc = ax.scatter(
                df[x_key], df[y_key], c=c_data, 
                s=s, alpha=alpha, cmap=cmap, edgecolors="none"
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
            cbar.outline.set_linewidth(0.5)
            cbar.set_label(color_by, size=fontsize, weight='bold')
            cbar.ax.tick_params(labelsize=ticksize)

    from scipy.stats import spearmanr
    corr, p_value = spearmanr(df[x_key], df[y_key])
    if p_value == 0.0 or p_value < 1e-16:
        p_txt = r"$< 10^{-16}$"
    elif p_value < 1e-3:
        p_txt = rf"$= 10^{{{int(np.floor(np.log10(p_value)))}}}$"
    elif p_value >= 0.999:
        p_txt = r"$= 1.00$"
    elif p_value < 0.1:
        p_txt = rf"$= {p_value:.3f}$"
    else:
        p_txt = rf"$= {p_value:.2f}$"
    ax.text(0.05, 0.95, f"Spearman ρ = {corr:.2f}" + "\n" + rf"$p$-value {p_txt}", 
            transform=ax.transAxes, fontsize=fontsize, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

    final_xlabel = xlabel if xlabel else x_key
    final_ylabel = ylabel if ylabel else y_key

    ax.set_xlabel(final_xlabel, fontsize=fontsize)
    ax.set_ylabel(final_ylabel, fontsize=fontsize)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('data',-0.05))
    ax.spines['bottom'].set_position(('data',-0.05))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.plot(xlim[1], -0.05, ">k", clip_on=False)
    ax.plot(-0.05, ylim[1], "^k", clip_on=False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.tick_params(axis='both', which='major', labelsize=ticksize)

    if show:
        plt.tight_layout()
        plt.show()
    
    return corr
