import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# 다른 모듈에서 import (실제 구현 시)
# from .encoders import EmotionEncoder, AttractionEmotionProjector, PersonaExpansionLayer
# from .attention_modules import BidirectionalCrossAttention, AttractionToEmotionAttention  
# from .filters import EnhancedPCDimensionFilter
# from .experts import ExpertLayer


class EmotionInteractionModule(nn.Module):
    """
    감성 차원 간 상호작용 모듈
    """

    def __init__(self, emotion_dim: int = 128, num_heads: int = 4, dropout_rate: float = 0.25):
        super(EmotionInteractionModule, self).__init__()
        assert emotion_dim % num_heads == 0, f"Emotion dimension ({emotion_dim}) must be divisible by num_heads ({num_heads})"

        self.self_attention = nn.MultiheadAttention(
            embed_dim=emotion_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.norm = nn.LayerNorm(emotion_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(emotion_dim, emotion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emotion_dim * 2, emotion_dim)
        )

        self.norm2 = nn.LayerNorm(emotion_dim)

    def forward(self, x: torch.Tensor, pc_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = x.unsqueeze(1)

        attn_out, attention_weights = self.self_attention(x_seq, x_seq, x_seq)
        x_normed = self.norm(x + attn_out.squeeze(1))

        ff_out = self.feedforward(x_normed)
        output = self.norm2(x_normed + ff_out)

        weighted_pc_scores = pc_scores

        return output, weighted_pc_scores


class EmotionPersonaRecommender(nn.Module):
    """
    감성 기반 페르소나 추천 시스템 메인 모델
    """

    def __init__(
            self,
            persona_dim: int,
            attraction_dim: int,
            pc_dim: int = 7,
            hidden_dim: int = 256,
            emotion_dim: int = 128,
            num_heads: int = 4,
            dropout: float = 0.25,
            use_emotion_matching: bool = True
    ):
        super(EmotionPersonaRecommender, self).__init__()

        self.pc_dim = pc_dim
        self.use_emotion_matching = use_emotion_matching
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim

        assert emotion_dim % num_heads == 0, f"Emotion dimension ({emotion_dim}) must be divisible by num_heads ({num_heads})"
        assert hidden_dim % num_heads == 0, f"Hidden dimension ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.emotion_interaction = EmotionInteractionModule(
            emotion_dim=emotion_dim,
            num_heads=num_heads
        )

        self.output_layer = nn.Sequential(
            nn.Linear(emotion_dim, emotion_dim // 2),
            nn.LayerNorm(emotion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emotion_dim // 2, 1)
        )

        self.register_buffer('pc_dim_importance', torch.zeros(pc_dim))
        self.importance_count = 0

    def forward(self, persona: torch.Tensor, attraction: torch.Tensor, pc_scores: torch.Tensor,
                add_noise: bool = False) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        batch_size = persona.size(0)

        if self.training and add_noise:
            persona = persona + torch.randn_like(persona) * 0.01
            attraction = attraction + torch.randn_like(attraction) * 0.01

        persona_interacted, weighted_pc_scores = self.emotion_interaction(persona, pc_scores)

        if self.use_emotion_matching:
            similarity = F.cosine_similarity(persona, attraction, dim=1)
            scaled_similarity = (similarity + 1) / 2
            final_output = scaled_similarity
        else:
            final_output = self.output_layer(persona_interacted).squeeze(-1)

        if self.training:
            self.pc_dim_importance += weighted_pc_scores.mean(dim=0).detach()
            self.importance_count += 1

        filter_activation = torch.mean(torch.abs(persona_interacted - persona)).item()

        return final_output, weighted_pc_scores, filter_activation, None
