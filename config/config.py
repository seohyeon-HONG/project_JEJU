import random
import numpy as np
import torch


class Config:
    """감성 기반 페르소나 추천 시스템 설정"""

    # ===== 설정 파라미터 =====
    # 쉽게 변경할 수 있는 모델 파라미터들(중앙 제어 타워)

    # 기본 설정
    SEED = 42
    DATA_PATH = "/content/drive/MyDrive/Project_JEJU/dataset"
    EMBEDDINGS_PATH = "/content/drive/MyDrive/Project_JEJU/modeling/embeddings_pkl"
    OUTPUT_PATH = "/content/drive/MyDrive/Project_JEJU/models/0412"

    # 모델 아키텍처 관련
    NUM_PC_DIMS = 7  # PCA 차원 수
    HIDDEN_DIM = 256  # 히든 레이어 차원
    NUM_HEADS = 4  # 어텐션 헤드 수
    PERSONA_EXPANSION_FACTOR = 2  # 페르소나 확장 계수
    EMOTION_EMBEDDING_DIM = 128  # 감성 임베딩 차원

    # 학습 관련
    NUM_EPOCHS = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0003
    DROPOUT_RATE = 0.25  # 과적합 방지
    WEIGHT_DECAY = 2e-4  # 가중치 감쇠를 줘서 파라미터 폭주를 막는다.

    # 대조학습 관련(Emotion Encoder와 Attraction Projector 사전 훈련에 쓰인다.)
    INTERNAL_CONTRAST_WEIGHT = 0.3  # 내부 대조 가중치 (LIGHT)
    EXTERNAL_CONTRAST_WEIGHT = 0.7  # 외부 대조 가중치 (STRONG)
    CONTRASTIVE_TEMPERATURE = 0.5  # 대조학습 온도 파라미터

    # 복합 손실 함수 가중치(감성 기반 예측 모델의 해석 가능성과 다양성 확보를 위함)
    MSE_WEIGHT = 0.5  # 회귀 손실 가중치
    BPR_WEIGHT = 0.3  # 랭킹 손실 가중치
    REG_WEIGHT = 0.1  # 정규화 가중치
    FILTER_REG_WEIGHT = 0.1  # 필터 활성화 정규화 가중치

    # 평가 관련
    TOP_K_VALUES = [5, 10]  # 평가에 사용할 K 값들(5개 미만이면 평가에서 제외)
    MIN_VISITS = 5  # 사용자별 최소 방문 수
    N_SPLITS = 3  # 교차 검증 분할 수


# 시드 설정
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True