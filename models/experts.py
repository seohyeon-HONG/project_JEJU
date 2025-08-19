import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLayer(nn.Module):
    """
    다양한 페르소나 유형 대응 전문가 레이어
    """

    def __init__(self, input_dim: int = 128, num_pc_dims: int = 7, dropout_rate: float = 0.25):
        super(ExpertLayer, self).__init__()

        # 공통 전문가 네트워크
        self.shared_expert = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, input_dim)
        )

        # 페르소나별 전문가 네트워크
        self.persona_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.LayerNorm(input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim // 2, input_dim)
            )
            for _ in range(num_pc_dims)
        ])

        self.gate_network = nn.Sequential(
            nn.Linear(num_pc_dims, num_pc_dims + 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor, pc_scores: torch.Tensor) -> torch.Tensor:
        shared_out = self.shared_expert(x)

        persona_outs = [expert(x) for expert in self.persona_experts]

        gates = self.gate_network(pc_scores)

        combined_out = gates[:, 0].unsqueeze(1) * shared_out
        for i, persona_out in enumerate(persona_outs):
            combined_out = combined_out + gates[:, i + 1].unsqueeze(1) * persona_out

        return combined_out