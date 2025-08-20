import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EmotionEncoder(nn.Module):
    """
    감성 벡터 정제를 위한 인코더
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, emotion_dim: int = 128, dropout_rate: float = 0.25):
        super(EmotionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 39 -> 256
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),  # 256 -> 256
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, emotion_dim)  # 256 -> 128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AttractionEmotionProjector(nn.Module):
    """
    관광지 임베딩을 감성 공간으로 투영
    """

    def __init__(self, attraction_dim: int, emotion_dim: int = 128, dropout_rate: float = 0.25):
        super(AttractionEmotionProjector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(attraction_dim, attraction_dim // 2),
            nn.LayerNorm(attraction_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(attraction_dim // 2, emotion_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class PersonaExpansionLayer(nn.Module):
    """
    페르소나 벡터 확장
    """

    def __init__(self, input_dim: int, expansion_factor: int = 1, dropout_rate: float = 0.25):
        
        super(PersonaExpansionLayer, self).__init__()
        self.expanded_dim = input_dim  

        self.expansion = nn.Sequential(
            nn.Linear(input_dim, self.expanded_dim),
            nn.LayerNorm(self.expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Residual connection
        self.residual_proj = nn.Linear(input_dim,
                                       self.expanded_dim) if input_dim != self.expanded_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        expanded = self.expansion(x)

        return expanded + residual
