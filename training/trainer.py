import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from utils.config import Config


def train_and_evaluate(
        filtered_df,
        traveler_features,
        attraction_features,
        traveler_pc_scores,
        pc_meanings,
        pretrained_encoder=None,
        pretrained_projector=None,
        n_splits=Config.N_SPLITS,
        batch_size=Config.BATCH_SIZE,
        n_epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
):
    """ 모델 학습 및 평가 함수 (원본 코드)
    """
    print(f"모델 학습 시작")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 사용자 ID 기준으로 폴드 분할
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
    unique_travelers = filtered_df['TRAVELER_ID'].unique()

    # 결과 저장
    fold_metrics = []
    fold_pc_weights = []
    best_models = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_travelers)):
        print(f"\n폴드 {fold + 1}/{n_splits}")

        # 훈련/테스트 사용자 분할
        train_travelers = unique_travelers[train_idx]
        test_travelers = unique_travelers[test_idx]

        # 사용자 ID 기준으로 데이터프레임 분할
        train_df = filtered_df[filtered_df['TRAVELER_ID'].isin(train_travelers)]
        test_df = filtered_df[filtered_df['TRAVELER_ID'].isin(test_travelers)]

        print(f"학습 샘플: {len(train_df)}, 테스트 샘플: {len(test_df)}")

        # 데이터셋 및 데이터로더 생성
        from data.dataset import EmotionPersonaDataset

        train_dataset = EmotionPersonaDataset(
            train_df, traveler_features, attraction_features, traveler_pc_scores
        )
        test_dataset = EmotionPersonaDataset(
            test_df, traveler_features, attraction_features, traveler_pc_scores
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        # 모델 초기화
        from models.recommender import EmotionPersonaRecommender

        persona_dim = next(iter(traveler_features.values())).shape[0]
        attraction_dim = next(iter(attraction_features.values())).shape[0]

        model = EmotionPersonaRecommender(
            persona_dim=persona_dim,
            attraction_dim=attraction_dim,
            pc_dim=Config.NUM_PC_DIMS,
            hidden_dim=Config.HIDDEN_DIM,
            emotion_dim=Config.EMOTION_EMBEDDING_DIM,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT_RATE,
            use_emotion_matching=True
        ).to(device)

        # 사전 훈련된 인코더와 투영기 가중치 로드 (제공된 경우)
        if pretrained_encoder is not None:
            model.emotion_encoder.load_state_dict(pretrained_encoder.state_dict())
            print("사전 훈련된 Emotion Encoder 가중치 로드 완료")

        if pretrained_projector is not None:
            model.attraction_projector.load_state_dict(pretrained_projector.state_dict())
            print("사전 훈련된 Attraction Projector 가중치 로드 완료")

        # 손실 함수
        from models.losses import RecommendationLoss

        criterion = RecommendationLoss(
            mse_weight=Config.MSE_WEIGHT,
            bpr_weight=Config.BPR_WEIGHT,
            reg_weight=Config.REG_WEIGHT,
            filter_reg_weight=Config.FILTER_REG_WEIGHT
        )

        # 옵티마이저 - 더 낮은 학습률 사용 & 작은 가중치 감쇠
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr * 0.1,  # 학습률 감소
            weight_decay=weight_decay * 0.1,  # 가중치 감쇠 감소
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 학습률 스케줄러 - 더 부드러운 변화
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 0.1,  # 최대 학습률 감소
            steps_per_epoch=len(train_loader),
            epochs=n_epochs,
            pct_start=0.3,
            div_factor=10.0,  # 더 작은 초기 학습률 감소율
            final_div_factor=100.0  # 더 작은 최종 학습률 감소율
        )

        # 조기 종료 설정
        best_ndcg = 0
        best_model_state = None
        patience = 5
        patience_counter = 0

        # 손실 컴포넌트 추적
        loss_history = {
            'total': [],
            'mse': [],
            'bpr': [],
            'diff_reg': [],
            'filter_reg': []
        }

        # 에포크별 학습
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0
            mse_losses = []
            bpr_losses = []
            diff_reg_losses = []
            filter_reg_losses = []
            filter_activations = []

            # 학습 루프
            for batch_idx, batch in enumerate(train_loader):
                try:
                    persona = batch['persona'].to(device)
                    attraction = batch['attraction'].to(device)
                    pc_scores = batch['pc_scores'].to(device)
                    scores = batch['score'].to(device)

                    # 예측
                    predictions, pc_weights, filter_activation, _ = model(persona, attraction, pc_scores)

                    # NaN 체크
                    if torch.isnan(predictions).any() or torch.isnan(pc_weights).any():
                        print(f"배치 {batch_idx}에서 NaN 발견 - 건너뛰기")
                        continue

                    # 손실 계산
                    loss, loss_components = criterion(predictions, scores, filter_activation, pc_weights)

                    # NaN 손실 체크
                    if torch.isnan(loss).any():
                        print(f"배치 {batch_idx}에서 NaN 손실 발견 - 건너뛰기")
                        continue

                    # 역전파 및 최적화
                    optimizer.zero_grad()
                    loss.backward()

                    # 그래디언트 클리핑 - 더 낮은 임계값
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # 임계값 0.5 -> 0.1로 낮춤

                    optimizer.step()
                    scheduler.step()

                    # 손실 및 활성화 추적
                    train_loss += loss.item()
                    mse_losses.append(loss_components['mse_loss'])
                    bpr_losses.append(loss_components['bpr_loss'])
                    diff_reg_losses.append(loss_components['diff_reg_loss'])
                    filter_reg_losses.append(loss_components['filter_reg'])
                    filter_activations.append(filter_activation)

                except Exception as e:
                    print(f"배치 {batch_idx} 처리 중 오류 발생: {e}")
                    continue

            # 빈 손실 리스트 체크
            if not mse_losses:
                print("경고: 이번 에포크에서 유효한 배치가 없습니다.")
                continue

            # 에포크 평균 손실
            train_loss /= max(1, len(mse_losses))  # 0으로 나누기 방지
            avg_mse = np.mean(mse_losses) if mse_losses else 0
            avg_bpr = np.mean(bpr_losses) if bpr_losses else 0
            avg_diff_reg = np.mean(diff_reg_losses) if diff_reg_losses else 0
            avg_filter_reg = np.mean(filter_reg_losses) if filter_reg_losses else 0
            avg_filter_activation = np.mean(filter_activations) if filter_activations else 0

            # 손실 히스토리 저장
            loss_history['total'].append(train_loss)
            loss_history['mse'].append(avg_mse)
            loss_history['bpr'].append(avg_bpr)
            loss_history['diff_reg'].append(avg_diff_reg)
            loss_history['filter_reg'].append(avg_filter_reg)

            # 검증 (5 에포크마다 또는 마지막 에포크)
            if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                try:
                    from evaluation.evaluator import evaluate_model
                    metrics, _, _, _, _, epoch_pc_weights = evaluate_model(model, test_loader, device)
                    ndcg5 = metrics.get('ndcg@5', 0)

                    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss:.4f}, "
                          f"MSE: {avg_mse:.4f}, BPR: {avg_bpr:.4f}, "
                          f"필터 활성화: {avg_filter_activation:.6f}, NDCG@5: {ndcg5:.4f}")

                    # PC 차원 가중치 출력
                    if epoch_pc_weights is not None:
                        print("PC 차원 중요도 가중치:")
                        for i, weight in enumerate(epoch_pc_weights):
                            pc_name = f"PC{i + 1}"
                            meaning = pc_meanings.get(pc_name, "")
                            weight_val = weight.item() if not torch.isnan(weight).any() else "NaN"
                            print(f"  {pc_name} ({meaning}): {weight_val}")

                    # 최고 성능 모델 저장
                    if ndcg5 > best_ndcg:
                        best_ndcg = ndcg5
                        best_model_state = copy.deepcopy(model.state_dict())
                        best_pc_weights = epoch_pc_weights
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # 조기 종료
                    if patience_counter >= patience:
                        print(f"조기 종료: {patience} 에포크 동안 성능 향상 없음")
                        break

                except Exception as e:
                    print(f"평가 중 오류 발생: {e}")
                    continue

        # 최고 성능 모델로 복원
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # 최종 평가
        try:
            from evaluation.evaluator import evaluate_model
            metrics, user_metrics, user_predictions, user_ground_truth, user_attraction_ids, final_pc_weights = evaluate_model(
                model, test_loader, device
            )

            print(f"\n폴드 {fold + 1} 최종 결과:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            print("\nPC 차원별 최종 중요도:")
            for i, weight in enumerate(final_pc_weights):
                pc_name = f"PC{i + 1}"
                meaning = pc_meanings.get(pc_name, "")
                weight_val = weight.item() if not torch.isnan(weight).any() else "NaN"
                print(f"  {pc_name} ({meaning}): {weight_val}")

            # 모델의 누적 PC 차원 중요도 분석
            if model.importance_count > 0:
                accumulated_importance = model.pc_dim_importance / model.importance_count
                print("\n학습 과정에서의 PC 차원 누적 중요도:")
                for i, imp in enumerate(accumulated_importance):
                    pc_name = f"PC{i + 1}"
                    meaning = pc_meanings.get(pc_name, "")
                    imp_val = imp.item() if not torch.isnan(imp).any() else "NaN"
                    print(f"  {pc_name} ({meaning}): {imp_val}")

            # 결과 저장
            fold_metrics.append(metrics)
            fold_pc_weights.append(final_pc_weights)
            best_models.append((model, user_predictions, user_ground_truth, user_attraction_ids))

        except Exception as e:
            print(f"최종 평가 중 오류 발생: {e}")
            continue

        # 손실 그래프 그리기
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history['total'], label='Total Loss')
        plt.plot(loss_history['mse'], label='MSE Loss')
        plt.plot(loss_history['bpr'], label='BPR Loss')
        plt.plot(loss_history['diff_reg'], label='Diff Reg')
        plt.plot(loss_history['filter_reg'], label='Filter Reg')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold + 1} Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"fold_{fold + 1}_loss.png")
        plt.close()

    # 전체 폴드의 평균 성능 계산
    avg_metrics = {}
    for metric in [f'ndcg@{k}' for k in Config.TOP_K_VALUES] + \
                  [f'precision@{k}' for k in Config.TOP_K_VALUES] + \
                  [f'recall@{k}' for k in Config.TOP_K_VALUES]:
        values = [fold[metric] for fold in fold_metrics if metric in fold]
        avg_metrics[metric] = np.mean(values) if values else 0

    # 평균 PC 차원 가중치 계산
    avg_pc_weights = torch.zeros(Config.NUM_PC_DIMS, device=device)
    for weights in fold_pc_weights:
        avg_pc_weights += weights
    avg_pc_weights /= len(fold_pc_weights)

    # 최종 출력
    print("\n===== 전체 성능 =====")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n===== PC 차원별 평균 중요도 =====")
    for i, weight in enumerate(avg_pc_weights):
        pc_name = f"PC{i + 1}"
        meaning = pc_meanings.get(pc_name, "")
        print(f"  {pc_name} ({meaning}): {weight.item():.4f}")

    # 최고 성능 모델 선택
    best_model_idx = np.argmax([
        fold_metrics[fold_idx].get('ndcg@5', 0) for fold_idx in range(len(fold_metrics))
    ])
    best_model, _, _, _ = best_models[best_model_idx]

    return best_model, avg_metrics, fold_metrics, fold_pc_weights, best_models