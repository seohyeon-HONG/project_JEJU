import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BidirectionalCrossAttention(nn.Module):
    """
    페르소나와 관광지 간 양방향 어텐션
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout_rate: float = 0.25):
        super(BidirectionalCrossAttention, self).__init__()

        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # 페르소나 -> 관광지 어텐션
        self.persona_to_attraction = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # 관광지 -> 페르소나 어텐션
        self.attraction_to_persona = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, persona_emb: torch.Tensor, attraction_emb: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        if persona_emb.dim() == 2:
            persona_emb = persona_emb.unsqueeze(1)
        if attraction_emb.dim() == 2:
            attraction_emb = attraction_emb.unsqueeze(1)

        scaled_persona = persona_emb / self.temperature
        scaled_attraction = attraction_emb / self.temperature

        p2a_attn, p2a_weights = self.persona_to_attraction(
            query=persona_emb,
            key=scaled_attraction,
            value=attraction_emb
        )
        persona_new = self.norm1(persona_emb + p2a_attn)

        a2p_attn, a2p_weights = self.attraction_to_persona(
            query=attraction_emb,
            key=scaled_persona,
            value=persona_emb
        )
        attraction_new = self.norm2(attraction_emb + a2p_attn)

        persona_new = persona_new.squeeze(1)
        attraction_new = attraction_new.squeeze(1)

        combined = torch.cat([persona_new, attraction_new], dim=1)
        gate = self.fusion_gate(combined)
        fused_emb = gate * persona_new + (1 - gate) * attraction_new

        return fused_emb, p2a_weights, a2p_weights


class AttractionToEmotionAttention(nn.Module):
    """
    관광지 임베딩을 감성 차원에 맞게 변환
    """

    def __init__(self, attraction_dim: int, emotion_dim: int = 128, num_pc_dims: int = 7, num_heads: int = 4,
                 dropout_rate: float = 0.25):
        super(AttractionToEmotionAttention, self).__init__()

        assert emotion_dim % num_heads == 0, f"Emotion dimension ({emotion_dim}) must be divisible by num_heads ({num_heads})"

        self.feature_extractor = nn.Sequential(
            nn.Linear(attraction_dim, emotion_dim),
            nn.LayerNorm(emotion_dim),
            nn.GELU()
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=emotion_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.emotion_projection = nn.Sequential(
            nn.Linear(emotion_dim, emotion_dim // 2),
            nn.LayerNorm(emotion_dim // 2),
            nn.GELU(),
            nn.Linear(emotion_dim // 2, num_pc_dims)
        )


        self.pc_gates = nn.Parameter(torch.ones(num_pc_dims) * 2.0)

    def forward(self, attraction_emb: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(attraction_emb)

        features = features.unsqueeze(1)

        attn_out, _ = self.attention(features, features, features)
        attn_out = attn_out.squeeze(1)

        emotion_scores = self.emotion_projection(attn_out)

        gated_scores = emotion_scores * torch.sigmoid(self.pc_gates)


        return gated_scores
