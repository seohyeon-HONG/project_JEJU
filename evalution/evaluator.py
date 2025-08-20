import torch
import numpy as np
from collections import defaultdict

from utils.config import Config


def relative_ndcg(y_true, y_pred, k=5, percentile=70):
    """사용자 기준 상위 평가 아이템에 대한 NDCG"""
    if len(y_true) < 3:
        return 0.0

    threshold = np.percentile(y_true, percentile)
    # 이진 관련성 점수 생성 (threshold 이상이면 1, 아니면 0)
    binary_relevance = (y_true >= threshold).astype(int)

    # 상위 k개 아이템 인덱스
    top_k_indices = np.argsort(y_pred)[-k:][::-1]  # 내림차순

    # DCG 계산
    dcg = 0
    for i, idx in enumerate(top_k_indices):
        if i < k:
            dcg += (2 ** binary_relevance[idx] - 1) / np.log2(i + 2)

    # IDCG 계산
    ideal_indices = np.argsort(binary_relevance)[-k:][::-1]  
    idcg = 0
    for i, idx in enumerate(ideal_indices):
        if i < k:
            idcg += (2 ** binary_relevance[idx] - 1) / np.log2(i + 2)

    # NDCG 계산
    if idcg > 0:
        return dcg / idcg
    else:
        return 0.0


def relative_precision_at_k(y_true, y_pred, k=5, percentile=70):
    """상위 k개 아이템 중 사용자 기준 상위 평가 아이템의 비율"""
    if len(y_true) < 3:
        return 0.0

    threshold = np.percentile(y_true, percentile)
    top_k_indices = np.argsort(y_pred)[-k:][::-1]  
    relevance = y_true[top_k_indices] >= threshold
    return np.sum(relevance) / k


def relative_recall_at_k(y_true, y_pred, k=10, percentile=70):
    """사용자 기준 상위 평가 아이템 중 상위 k개에 포함된 아이템의 비율"""
    if len(y_true) < 3:
        return 0.0

    threshold = np.percentile(y_true, percentile)
    top_k_indices = np.argsort(y_pred)[-k:][::-1]  # 내림차순
    relevance_in_top_k = y_true[top_k_indices] >= threshold
    total_relevant = np.sum(y_true >= threshold)

    if total_relevant > 0:
        return np.sum(relevance_in_top_k) / total_relevant
    return 0.0


def evaluate_model(model, test_dataloader, device):
    """모델 평가 함수"""
    model.eval()

    # 예측 및 실제 점수 수집
    user_predictions = defaultdict(list)
    user_ground_truth = defaultdict(list)
    user_attraction_ids = defaultdict(list)

    pc_weights_sum = None
    pc_weights_count = 0

    with torch.no_grad():
        for batch in test_dataloader:
            persona = batch['persona'].to(device)
            attraction = batch['attraction'].to(device)
            pc_scores = batch['pc_scores'].to(device)
            scores = batch['score'].to(device)
            traveler_ids = batch['traveler_id']
            attraction_ids = batch['attraction_id']

            predictions, pc_weights, _, _ = model(persona, attraction, pc_scores, add_noise=False)

            for i in range(len(traveler_ids)):
                t_id = traveler_ids[i]
                a_id = attraction_ids[i]
                user_predictions[t_id].append(predictions[i].item())
                user_ground_truth[t_id].append(scores[i].item())
                user_attraction_ids[t_id].append(a_id)

            if pc_weights_sum is None:
                pc_weights_sum = pc_weights.sum(dim=0)
            else:
                pc_weights_sum += pc_weights.sum(dim=0)
            pc_weights_count += len(pc_weights)

    for u_id in user_predictions:
        user_predictions[u_id] = np.array(user_predictions[u_id])
        user_ground_truth[u_id] = np.array(user_ground_truth[u_id])

    user_metrics = {}
    for u_id in user_predictions:
        if len(user_predictions[u_id]) >= Config.MIN_VISITS:  
            y_pred = user_predictions[u_id]
            y_true = user_ground_truth[u_id]

            metrics_dict = {}
            for k in Config.TOP_K_VALUES:
                metrics_dict.update({
                    f'ndcg@{k}': relative_ndcg(y_true, y_pred, k=k),
                    f'precision@{k}': relative_precision_at_k(y_true, y_pred, k=k),
                    f'recall@{k}': relative_recall_at_k(y_true, y_pred, k=k),
                })

            user_metrics[u_id] = metrics_dict

    avg_metrics = {}
    for metric in [f'ndcg@{k}' for k in Config.TOP_K_VALUES] + \
                  [f'precision@{k}' for k in Config.TOP_K_VALUES] + \
                  [f'recall@{k}' for k in Config.TOP_K_VALUES]:
        values = [m[metric] for m in user_metrics.values() if metric in m]
        avg_metrics[metric] = np.mean(values) if values else 0

    avg_pc_weights = pc_weights_sum / pc_weights_count if pc_weights_count > 0 else None

    print(f"\n평가 완료: {len(user_metrics)} 사용자, 평균 NDCG@5: {avg_metrics.get('ndcg@5', 0):.4f}")


    return avg_metrics, user_metrics, user_predictions, user_ground_truth, user_attraction_ids, avg_pc_weights
