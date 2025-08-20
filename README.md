# Project JEJU

감성 프로필 기반 멀티모달 제주도 여행지 추천 시스템 프로젝트입니다.

## 프로젝트 구조
```
Project_JEJU/
├── README.md
├── requirements.txt
├── config/
│   └── config.py                    # 전역 설정 파라미터
├── data/
│   ├── processing.py                # TargetScoreNormalizer
│   ├── persona_processor.py         # PersonaProcessor 
│   └── sample/                      # 샘플 데이터
├── models/
│   ├── encoders.py                  # EmotionEncoder, AttractionEmotionProjector
│   ├── attention_modules.py         # BidirectionalCrossAttention
│   ├── filters.py                   # EnhancedPCDimensionFilter
│   ├── experts.py                   # ExpertLayer
│   ├── losses.py                    # ContrastiveLoss, RecommendationLoss
│   └── recommender.py               # EmotionPersonaRecommender 
├── training/
│   ├── pretrain.py                  # 감성 인코더/투영기 사전훈련
│   └── trainer.py                   # 메인 모델 학습 및 평가
└── evaluation/
    └── evaluator.py                 # 성능 평가 지표
```
