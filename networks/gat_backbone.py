"""
GAT 네트워크 구현.

Edge feature를 attention 계산에 반영하는 커스텀 GAT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

try:
    from ..config import GATConfig
except ImportError:
    from config import GATConfig


class EdgeAwareGATConv(MessagePassing):
    """
    Edge feature를 attention 계산에 반영하는 커스텀 GAT 레이어.

    torch_geometric.nn.MessagePassing 상속.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        heads: int = 4,
        negative_slope: float = 0.2,
        dropout: float = 0.0
    ):
        """
        Args:
            in_dim: 입력 노드 feature 차원
            out_dim: 출력 노드 feature 차원
            edge_dim: 엣지 feature 차원
            heads: Attention head 수
        """
        super().__init__(aggr='add', node_dim=0)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        # 노드 변환
        self.lin_src = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, heads * out_dim, bias=False)

        # 엣지 변환
        self.lin_edge = nn.Linear(edge_dim, heads * out_dim, bias=False)

        # Attention 파라미터
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_dim))

        # 바이어스
        self.bias = nn.Parameter(torch.Tensor(heads * out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """파라미터 초기화."""
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: 노드 features (N, in_dim)
            edge_index: 엣지 연결 (2, E)
            edge_attr: 엣지 features (E, edge_dim)

        Returns:
            업데이트된 노드 features (N, out_dim * heads)
        """
        H, C = self.heads, self.out_dim

        # 노드 변환
        x_src = self.lin_src(x).view(-1, H, C)  # (N, H, C)
        x_dst = self.lin_dst(x).view(-1, H, C)  # (N, H, C)

        # 엣지 변환
        edge_attr_transformed = self.lin_edge(edge_attr).view(-1, H, C)  # (E, H, C)

        # 소스/타겟 노드 attention 스코어 계산
        alpha_src = (x_src * self.att_src).sum(dim=-1)  # (N, H)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)  # (N, H)

        # 메시지 패싱
        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            alpha=(alpha_src, alpha_dst),
            edge_attr=edge_attr_transformed
        )

        out = out.view(-1, H * C)
        out = out + self.bias

        return out

    def message(
        self,
        x_j: Tensor,
        alpha_i: Tensor,
        alpha_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int]
    ) -> Tensor:
        """
        메시지 패싱 함수.

        Args:
            x_j: 소스 노드 features (E, H, C)
            alpha_i: 타겟 노드 attention 스코어 (E, H)
            alpha_j: 소스 노드 attention 스코어 (E, H)
            edge_attr: 변환된 엣지 features (E, H, C)
            index: 타겟 노드 인덱스
            ptr: CSR 포인터 (optional)
            size_i: 타겟 노드 수 (optional)

        Returns:
            메시지 (E, H, C)
        """
        # 엣지 attention 스코어
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)  # (E, H)

        # 총 attention 스코어
        alpha = alpha_i + alpha_j + alpha_edge  # (E, H)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 메시지 = attention * (소스 노드 + 엣지)
        message = alpha.unsqueeze(-1) * (x_j + edge_attr)  # (E, H, C)

        return message


class GATBackbone(nn.Module):
    """
    3-layer GAT 백본 네트워크.
    """

    def __init__(self, config: GATConfig):
        """
        Args:
            config: GAT 설정
        """
        super().__init__()

        self.config = config

        # 레이어 생성
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = config.node_input_dim

        for i in range(config.num_layers):
            # 마지막 레이어는 head 평균
            is_last = (i == config.num_layers - 1)

            layer = EdgeAwareGATConv(
                in_dim=in_dim,
                out_dim=config.hidden_dim if not is_last else config.hidden_dim // config.num_heads,
                edge_dim=config.edge_input_dim,
                heads=config.num_heads,
                dropout=config.dropout
            )
            self.layers.append(layer)

            out_dim = config.hidden_dim * config.num_heads if not is_last else config.hidden_dim
            self.norms.append(nn.LayerNorm(out_dim))

            in_dim = out_dim

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: 노드 features (N, node_input_dim)
            edge_index: 엣지 연결 (2, E)
            edge_attr: 엣지 features (E, edge_input_dim)

        Returns:
            노드 embeddings (N, hidden_dim)
        """
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x, 0.2)
                x = F.dropout(x, p=self.config.dropout, training=self.training)

        return x
