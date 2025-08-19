import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class ContrastiveLoss(nn.Module):
    """
    대조학습을 위한 손실 함수
    """

    def __init__(self, temperature: float = 0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """사전 훈련된 임베딩의 분포를 정규화"""
        normalized = (embeddings - embeddings.mean(dim=0)) / (embeddings.std(dim=0) + 1e-8)
        return normalized

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 입력 임베딩 정규화 적용
        embeddings = self.normalize_embeddings(embeddings)

        # 기존 L2 정규화 유지 (단위 벡터로 변환)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 온도 조정된 유사도 행렬 계산
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature

        # 대각선 마스킹 (자기 자신과의 유사도 제외)
        mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        mask = 1 - mask

        # 라벨 기반 마스크 생성
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        label_matrix = label_matrix.float() * mask

        # negative 마스크
        neg_mask = (~label_matrix.bool()).float() * mask

        # 로그-소프트맥스 계산
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # 내부/외부 대조 손실 계산
        internal_loss = torch.sum(label_matrix * -log_prob, dim=1) / (label_matrix.sum(dim=1) + 1e-8)
        internal_loss = internal_loss.mean()

        neg_sim = sim_matrix * neg_mask
        hard_negatives = torch.max(neg_sim, dim=1)[0]
        external_loss = torch.mean(F.softplus(hard_negatives))

        # 가중 손실 계산 (config에서 가져올 값들)
        internal_weight = 0.3
        external_weight = 0.7
        loss = internal_weight * internal_loss + external_weight * external_loss

        return loss, {"internal": internal_loss.item(), "external": external_loss.item()}


class RecommendationLoss(nn.Module):
    """
    추천 시스템을 위한 복합 손실 함수
    """

    def __init__(
            self,
            mse_weight: float,
            bpr_weight: float,
            reg_weight: float,
            filter_reg_weight: float
    ):
        super(RecommendationLoss, self).__init__()
        self.mse_weight = mse_weight
        self.bpr_weight = bpr_weight
        self.reg_weight = reg_weight
        self.filter_reg_weight = filter_reg_weight
        self.mse = nn.MSELoss(reduction='none')  # 요소별 손실 계산

    @staticmethod
    def safe_entropy(weights: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """수치적으로 안정적인 엔트로피 계산"""
        normalized_weights = F.softmax(weights, dim=1)
        safe_weights = torch.clamp(normalized_weights, min=epsilon)
        entropy = -torch.sum(safe_weights * torch.log(safe_weights), dim=1).mean()
        return entropy

    @staticmethod
    def compute_bpr_loss_debug_verbose(valid_pred: torch.Tensor, valid_targ: torch.Tensor,
                                       margin: float = 0.2, temperature: float = 0.1) -> Tuple[torch.Tensor, int]:
        bpr_loss = 0
        num_pairs = 0
        diff_list = []
        pair_losses = []

        batch_size = valid_pred.shape[0]
        valid_indices = torch.arange(batch_size, device=valid_pred.device)

        for i_idx in valid_indices:
            for j_idx in valid_indices:
                if valid_targ[i_idx] > valid_targ[j_idx]:
                    diff = valid_pred[i_idx] - valid_pred[j_idx]
                    diff_list.append(diff.item())

                    clamped_diff = torch.clamp(diff, min=-10.0, max=10.0)
                    pair_loss = -torch.log(torch.sigmoid(clamped_diff) + 1e-10)
                    pair_losses.append(pair_loss.item())

                    bpr_loss += pair_loss
                    num_pairs += 1

        if num_pairs > 0:
            bpr_loss = bpr_loss / num_pairs
            print("\n=== BPR 손실 디버깅 정보 ===")
        else:
            bpr_loss = torch.tensor(0.0, device=valid_pred.device, requires_grad=True)
            print("유효한 pair가 없습니다.")

        return bpr_loss, num_pairs

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                filter_activation: float = None, pc_weights: torch.Tensor = None) -> Tuple[
        torch.Tensor, Dict[str, float]]:
        batch_size = predictions.size(0)

        # NaN 체크 및 처리
        valid_mask = ~torch.isnan(predictions) & ~torch.isnan(targets)
        if not valid_mask.any():
            return (
                torch.tensor(0.0, device=predictions.device, requires_grad=True),
                {'mse_loss': 0.0, 'bpr_loss': 0.0, 'diff_reg_loss': 0.0, 'filter_reg': 0.0}
            )

        valid_pred = predictions[valid_mask]
        valid_targ = targets[valid_mask]

        if len(valid_pred) == 0:
            return (
                torch.tensor(0.0, device=predictions.device, requires_grad=True),
                {'mse_loss': 0.0, 'bpr_loss': 0.0, 'diff_reg_loss': 0.0, 'filter_reg': 0.0}
            )

        # MSE 손실 계산
        element_losses = self.mse(valid_pred, valid_targ)
        element_losses = torch.clamp(element_losses, max=10.0)
        mse_loss = element_losses.mean()

        # BPR 손실 계산
        bpr_loss, num_pairs = self.compute_bpr_loss_debug_verbose(valid_pred, valid_targ)

        # 페르소나 차별화 정규화
        diff_reg_loss = 0
        if pc_weights is not None:
            try:
                pc_entropy = self.safe_entropy(pc_weights)
                diff_reg_loss = -0.05 * pc_entropy
                print(f"엔트로피: {pc_entropy.item():.4f}, 정규화 손실: {diff_reg_loss.item():.4f}")
            except Exception as e:
                print(f"엔트로피 계산 오류: {e}")
                diff_reg_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # 필터 정규화
        filter_reg = 0
        if filter_activation is not None:
            filter_reg = torch.max(
                torch.tensor(0.0, device=predictions.device),
                torch.tensor(0.05, device=predictions.device) - filter_activation
            )

        print("\n=== RecommendationLoss 디버깅 정보 ===")
        print(f"MSE Loss: {mse_loss.item():.6f}")
        print(f"BPR Loss: {bpr_loss.item()}")
        print(f"DiffReg Loss: {diff_reg_loss.item()}")
        print(f"Filter Reg: {filter_reg.item() if isinstance(filter_reg, torch.Tensor) else filter_reg}")

        total_loss = (self.mse_weight * mse_loss +
                      self.bpr_weight * bpr_loss +
                      self.reg_weight * diff_reg_loss +
                      self.filter_reg_weight * filter_reg)

        print(f"Total Loss: {total_loss.item():.6f}")

        loss_components = {
            'mse_loss': mse_loss.item(),
            'bpr_loss': bpr_loss.item(),
            'diff_reg_loss': diff_reg_loss.item(),
            'filter_reg': filter_reg.item() if isinstance(filter_reg, torch.Tensor) else filter_reg
        }

        return total_loss, loss_components


class TripletLoss(nn.Module):
    """
    트리플렛 손실 (사전 훈련용)
    """

    def __init__(self, margin: float = 0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class WeightedMSELoss(nn.Module):
    """
    가중치가 적용된 MSE 손실
    """

    def __init__(self, reduction: str = 'mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        squared_errors = (predictions - targets) ** 2
        weighted_errors = squared_errors * weights

        if self.reduction == 'mean':
            return weighted_errors.mean()
        elif self.reduction == 'sum':
            return weighted_errors.sum()
        else:
            return weighted_errors


class FocalLoss(nn.Module):
    """
    Focal Loss (불균형 데이터용)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE loss 계산
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')

        # Focal weight 계산
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Focal loss
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def main():
    """손실 함수 테스트"""
    batch_size = 32
    embed_dim = 128
    num_classes = 10

    # 테스트 데이터
    embeddings = torch.randn(batch_size, embed_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    predictions = torch.randn(batch_size)
    targets = torch.randn(batch_size)
    pc_weights = torch.randn(batch_size, 7)

    print("=== 손실 함수 테스트 ===")

    # 1. ContrastiveLoss 테스트
    contrastive_loss = ContrastiveLoss(temperature=0.5)
    cont_loss, cont_components = contrastive_loss(embeddings, labels)
    print(f"ContrastiveLoss: {cont_loss.item():.4f}")
    print(f"  Internal: {cont_components['internal']:.4f}")
    print(f"  External: {cont_components['external']:.4f}")

    # 2. RecommendationLoss 테스트
    rec_loss = RecommendationLoss(
        mse_weight=0.5, bpr_weight=0.3, reg_weight=0.1, filter_reg_weight=0.1
    )
    total_loss, loss_components = rec_loss(predictions, targets, 0.02, pc_weights)
    print(f"\nRecommendationLoss: {total_loss.item():.4f}")
    for comp, value in loss_components.items():
        print(f"  {comp}: {value:.4f}")

    # 3. TripletLoss 테스트
    triplet_loss = TripletLoss(margin=0.2)
    anchor = torch.randn(batch_size, embed_dim)
    positive = torch.randn(batch_size, embed_dim)
    negative = torch.randn(batch_size, embed_dim)
    trip_loss = triplet_loss(anchor, positive, negative)
    print(f"\nTripletLoss: {trip_loss.item():.4f}")

    print("\n손실 함수 테스트 완료!")


if __name__ == "__main__":
    main()