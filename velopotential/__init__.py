from .tools.core import construct_potential, add_velocity_noise, cal_cosine_similarity
from .plotting.core import scatter_from_adata, plot_velocity_projection
class _ToolsNamespace:
    def __init__(self):
        self. construct_potential = construct_potential
        self. add_velocity_noise = add_velocity_noise
        self. cal_cosine_similarity = cal_cosine_similarity
class _PlotNamespace:
    def __init__(self):
        self. scatter_from_adata = scatter_from_adata
        self. plot_velocity_projection = plot_velocity_projection
tl = _ToolsNamespace()
pl = _PlotNamespace()
__all__ = ["tl", "pl"]