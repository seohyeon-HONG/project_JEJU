import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedPCDimensionFilter(nn.Module):
    """
    감성 차원별 특화 필터
    """

    def __init__(self, num_pc_dims: int = 7, hidden_dim: int = 128, dropout_rate: float = 0.25):
        super(EnhancedPCDimensionFilter, self).__init__()

        self.pc_filters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            for _ in range(num_pc_dims)
        ])

        # 차원별 스케일 파라미터
        self.scale_params = nn.Parameter(torch.ones(num_pc_dims) * 3.0)

        # 차원별 중요도 게이트
        self.importance_gates = nn.Parameter(torch.randn(num_pc_dims))

    def forward(self, x: torch.Tensor, pc_scores: torch.Tensor) -> torch.Tensor:
        # 온도 조정으로 PC 점수 차별화
        temperature = 0.5
        sharpened_pc_scores = pc_scores / temperature

        batch_size = x.size(0)
        filtered_outputs = []

        # 감성 축별 필터링
        for i, filter_layer in enumerate(self.pc_filters):
            # 각 차원별 필터 적용
            filtered = filter_layer(x)

            # 배치별 PC 점수로 가중치 적용
            pc_weights = sharpened_pc_scores[:, i].unsqueeze(1)

            # 스케일 및 중요도 게이트를 통한 차별화 강화
            importance_factor = torch.sigmoid(self.importance_gates[i]) * 2.0  # 게이트 확장
            scaled_weights = pc_weights * (self.scale_params[i] * importance_factor)

            # 가중치의 제곱으로 차이 강화
            weighted_output = filtered * (scaled_weights ** 2)
            filtered_outputs.append(weighted_output)

        combined_output = torch.stack(filtered_outputs, dim=0)  # [7, B, 128]

        max_values, _ = torch.max(combined_output, dim=0)
        alpha = 0.7
        final_output = alpha * max_values + (1 - alpha) * combined_output.sum(dim=0)

        return final_output