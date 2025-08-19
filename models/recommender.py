import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .components.encoders import EmotionEncoder, AttractionEmotionProjector, PersonaExpansionLayer
from .components.attention_modules import BidirectionalCrossAttention, AttractionToEmotionAttention
from .components.filters import EnhancedPCDimensionFilter
from .components.experts import ExpertLayer


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

        # 1. Emotion Encoder
        self.emotion_encoder = EmotionEncoder(
            input_dim=persona_dim,
            hidden_dim=hidden_dim,
            emotion_dim=emotion_dim
        )

        # 2. Attraction Emotion Projector
        self.attraction_projector = AttractionEmotionProjector(
            attraction_dim=attraction_dim,
            emotion_dim=emotion_dim
        )

        # 3. Persona Expansion Layer
        self.persona_expansion = PersonaExpansionLayer(
            input_dim=emotion_dim,
            expansion_factor=2
        )

        # 4. Emotion Interaction Module
        self.emotion_interaction = EmotionInteractionModule(
            emotion_dim=emotion_dim,
            num_heads=num_heads
        )

        # 5. AttractionToEmotionAttention
        self.attraction_to_emotion = AttractionToEmotionAttention(
            attraction_dim=attraction_dim,
            emotion_dim=emotion_dim,
            num_pc_dims=pc_dim
        )

        # 6. EnhancedPCDimensionFilter
        self.pc_dimension_filter = EnhancedPCDimensionFilter(
            num_pc_dims=pc_dim,
            hidden_dim=emotion_dim
        )

        # 7. Expert Layer
        self.expert_layer = ExpertLayer(
            input_dim=emotion_dim,
            num_pc_dims=pc_dim
        )

        # 8. Bidirectional Cross Attention
        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=emotion_dim,
            num_heads=num_heads
        )

        # 9. 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(emotion_dim, emotion_dim // 2),
            nn.LayerNorm(emotion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emotion_dim // 2, 1)
        )

        # PC 차원 중요도 추적용
        self.register_buffer('pc_dim_importance', torch.zeros(pc_dim))
        self.importance_count = 0

    def forward(self, persona: torch.Tensor, attraction: torch.Tensor, pc_scores: torch.Tensor,
                add_noise: bool = False) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        batch_size = persona.size(0)

        # 훈련 중 약간의 노이즈 추가
        if self.training and add_noise:
            persona = persona + torch.randn_like(persona) * 0.01
            attraction = attraction + torch.randn_like(attraction) * 0.01

        # 1. 감성 인코딩
        persona_emotion = self.emotion_encoder(persona)

        # 2. 페르소나 확장
        expanded_persona = self.persona_expansion(persona_emotion)

        # 3. 감성 차원 간 상호작용
        persona_interacted, weighted_pc_scores = self.emotion_interaction(expanded_persona, pc_scores)

        # 4. 관광지 감성 투영
        attraction_emotion = self.attraction_projector(attraction)

        # 5. 관광지 감성 차원 맵핑
        attraction_pc_scores = self.attraction_to_emotion(attraction)

        # 6. PC 차원별 필터 적용
        filtered_persona = self.pc_dimension_filter(persona_interacted, weighted_pc_scores)

        # 7. 양방향 크로스 어텐션
        fused_emb, p2a_weights, a2p_weights = self.cross_attention(filtered_persona, attraction_emotion)

        # 8. 전문가 레이어 적용
        expert_out = self.expert_layer(fused_emb, weighted_pc_scores)

        # 9. 추천 점수 계산
        if self.use_emotion_matching:
            persona_pc = weighted_pc_scores
            attraction_pc = attraction_pc_scores

            similarity = F.cosine_similarity(persona_pc, attraction_pc, dim=1)
            scaled_similarity = (similarity + 1) / 2
            final_output = scaled_similarity
        else:
            final_output = self.output_layer(expert_out).squeeze(-1)

        # PC 차원 중요도 누적
        if self.training:
            self.pc_dim_importance += weighted_pc_scores.mean(dim=0).detach()
            self.importance_count += 1

        # 필터 활성화 정도 측정
        filter_activation = torch.mean(torch.abs(filtered_persona - persona_interacted)).item()

        return final_output, weighted_pc_scores, filter_activation, p2a_weights