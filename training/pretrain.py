import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any

from ..models.components.encoders import EmotionEncoder, AttractionEmotionProjector
from ..models.losses import ContrastiveLoss, TripletLoss


def pretrain_emotion_encoder(
        persona_df_pca: pd.DataFrame,
        traveler_features: Dict,
        pc_meanings: Dict,
        device: torch.device,
        output_path: str,
        n_epochs: int = 10,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 2e-4,
        contrastive_temperature: float = 0.5
) -> EmotionEncoder:
    """Emotion Encoder 사전 훈련 함수"""
    print("Emotion Encoder 사전 훈련 시작...")

    persona_dim = next(iter(traveler_features.values())).shape[0]
    num_pc_dims = len(pc_meanings)
    hidden_dim = 256
    emotion_dim = 128

    encoder = EmotionEncoder(
        input_dim=persona_dim,
        hidden_dim=hidden_dim,
        emotion_dim=emotion_dim
    ).to(device)

    pc_columns = [f'PC{i + 1}' for i in range(num_pc_dims)]
    data = []
    labels = []

    for idx, row in persona_df_pca.iterrows():
        t_id = row['TRAVELER_ID']
        if t_id in traveler_features:
            features = traveler_features[t_id]
            data.append(features)

            pc_scores = row[pc_columns].values
            dominant_pc = np.argmax(np.abs(pc_scores))
            labels.append(dominant_pc)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)

    tensor_data = torch.tensor(data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensor_data, tensor_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    contrastive_loss_fn = ContrastiveLoss(temperature=contrastive_temperature)
    optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        encoder.train()
        epoch_loss = 0
        epoch_internal_loss = 0
        epoch_external_loss = 0

        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            embeddings = encoder(batch_data)

            loss, loss_components = contrastive_loss_fn(embeddings, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_internal_loss += loss_components['internal']
            epoch_external_loss += loss_components['external']

        avg_loss = epoch_loss / len(dataloader)
        avg_internal = epoch_internal_loss / len(dataloader)
        avg_external = epoch_external_loss / len(dataloader)

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}, "
              f"Internal: {avg_internal:.4f}, External: {avg_external:.4f}")

    os.makedirs(output_path, exist_ok=True)
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'config': {
            'persona_dim': persona_dim,
            'hidden_dim': hidden_dim,
            'emotion_dim': emotion_dim
        }
    }, os.path.join(output_path, "emotion_encoder_pretrained.pt"))

    print("Emotion Encoder 사전 훈련 완료 및 저장!")
    return encoder


def pretrain_attraction_projector(
        filtered_df: pd.DataFrame,
        attraction_features: Dict,
        traveler_pc_scores: Dict,
        device: torch.device,
        output_path: str,
        n_epochs: int = 10,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 2e-4
) -> AttractionEmotionProjector:
    """Attraction Emotion Projector 사전 훈련 함수"""
    print("Attraction Emotion Projector 사전 훈련 시작...")

    attraction_dim = next(iter(attraction_features.values())).shape[0]
    num_pc_dims = next(iter(traveler_pc_scores.values())).shape[0]
    emotion_dim = 128

    projector = AttractionEmotionProjector(
        attraction_dim=attraction_dim,
        emotion_dim=emotion_dim
    ).to(device)

    triplets = []

    user_attractions = defaultdict(list)
    for _, row in filtered_df.iterrows():
        t_id = row['TRAVELER_ID']
        a_id = int(row['UNIQUE_VISIT_ID'])
        score = row['TARGET_SCORE_FINAL'] if 'TARGET_SCORE_FINAL' in filtered_df.columns else row['TARGET_SCORE']

        if t_id in traveler_pc_scores and a_id in attraction_features:
            user_attractions[t_id].append((a_id, score))

    for t_id, attractions in user_attractions.items():
        if len(attractions) < 3:
            continue

        attractions.sort(key=lambda x: x[1], reverse=True)

        top_20_percent = int(len(attractions) * 0.2) or 1
        positives = [a[0] for a in attractions[:top_20_percent]]
        negatives = [a[0] for a in attractions[-top_20_percent:]] if len(attractions) >= 5 else []

        if not negatives:  
            other_users = list(user_attractions.keys())
            other_users.remove(t_id)
            if other_users:
                random_user = random.choice(other_users)
                if user_attractions[random_user]:
                    negatives = [user_attractions[random_user][0][0]]  

        for pos in positives:
            for neg in negatives:
                triplets.append((positives[0], pos, neg, t_id))

    print(f"생성된 트리플렛 수: {len(triplets)}")

    class TripletDataset(Dataset):
        def __init__(self, triplets, attraction_features, traveler_pc_scores):
            self.triplets = triplets
            self.attraction_features = attraction_features
            self.traveler_pc_scores = traveler_pc_scores

        def __len__(self):
            return len(self.triplets)

        def __getitem__(self, idx):
            anchor_id, pos_id, neg_id, t_id = self.triplets[idx]
            anchor_vec = self.attraction_features[anchor_id]
            pos_vec = self.attraction_features[pos_id]
            neg_vec = self.attraction_features[neg_id]
            pc_scores = self.traveler_pc_scores[t_id]

            return {
                'anchor': torch.tensor(anchor_vec, dtype=torch.float32),
                'positive': torch.tensor(pos_vec, dtype=torch.float32),
                'negative': torch.tensor(neg_vec, dtype=torch.float32),
                'pc_scores': torch.tensor(pc_scores, dtype=torch.float32)
            }

    triplet_dataset = TripletDataset(triplets, attraction_features, traveler_pc_scores)
    triplet_loader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True)

    triplet_loss_fn = TripletLoss()
    mse_loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(projector.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        projector.train()
        epoch_triplet_loss = 0
        epoch_mse_loss = 0
        epoch_total_loss = 0

        for batch in triplet_loader:
            anchor_vec = batch['anchor'].to(device)
            pos_vec = batch['positive'].to(device)
            neg_vec = batch['negative'].to(device)
            pc_scores = batch['pc_scores'].to(device)

            optimizer.zero_grad()

            anchor_emotion = projector(anchor_vec)
            pos_emotion = projector(pos_vec)
            neg_emotion = projector(neg_vec)

            triplet_loss = triplet_loss_fn(anchor_emotion, pos_emotion, neg_emotion)

            projection_layer = torch.nn.Linear(emotion_dim, num_pc_dims).to(device)

            anchor_pc_pred = projection_layer(anchor_emotion)

            mse_loss = mse_loss_fn(anchor_pc_pred, pc_scores)

            loss = triplet_loss + 0.5 * mse_loss
            loss.backward()
            optimizer.step()

            epoch_triplet_loss += triplet_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_total_loss += loss.item()

        avg_triplet_loss = epoch_triplet_loss / len(triplet_loader)
        avg_mse_loss = epoch_mse_loss / len(triplet_loader)
        avg_total_loss = epoch_total_loss / len(triplet_loader)

        print(f"Epoch {epoch + 1}/{n_epochs}, Total Loss: {avg_total_loss:.4f}, "
              f"Triplet: {avg_triplet_loss:.4f}, MSE: {avg_mse_loss:.4f}")

    os.makedirs(output_path, exist_ok=True)
    torch.save({
        'model_state_dict': projector.state_dict(),
        'config': {
            'attraction_dim': attraction_dim,
            'emotion_dim': emotion_dim
        }
    }, os.path.join(output_path, "attraction_projector_pretrained.pt"))

    print("Attraction Emotion Projector 사전 훈련 완료 및 저장!")

    return projector
