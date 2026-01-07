"""Networks 모듈 - MAPPO용 Actor/Critic 및 Light-pH용 GAT 네트워크."""

from .actor import Actor
from .critic import Critic, CriticWithAttention

__all__ = [
    # MAPPO
    'Actor',
    'Critic',
    'CriticWithAttention',
]

# Light-pH GAT (torch_geometric 필요)
try:
    from .gat_backbone import EdgeAwareGATConv, GATBackbone
    from .scalar_heads import bounded_activation, NodeHead, EdgeHead, ObstacleEdgeHead

    __all__.extend([
        'EdgeAwareGATConv',
        'GATBackbone',
        'bounded_activation',
        'NodeHead',
        'EdgeHead',
        'ObstacleEdgeHead',
    ])
except ImportError:
    pass  # torch_geometric not installed
