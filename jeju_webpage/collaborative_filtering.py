import pandas as pd
import numpy as np
import joblib
import os
from sklearn.neighbors import NearestNeighbors
from clustering import predict_user_cluster

# 사용자 데이터 로드
visit_area_data = pd.read_csv(r"E:\AI_HUB\data\concatenated\tn_visit_area_info_concatened_WR.csv")
traveller_data = pd.read_csv(r"E:\AI_HUB\data\concatenated\tn_traveller_concatenated_cleaned.csv")
travel_data = pd.read_csv(r"E:\AI_HUB\data\concatenated\tn_travel_concatenated.csv")

# 경로 설정
MODEL_PATH = "cluster_model.pkl"
DATA_FOLDER = "jeju_webpage/data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# K-Means 모델 로드
if os.path.exists(MODEL_PATH):
    kmeans = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"'{MODEL_PATH}' 파일이 존재하지 않습니다. 같은 디렉토리에 있는지 확인하세요.")


# 클러스터 정보 추가 (없을 경우 예측하여 추가)
if "cluster" not in traveller_data.columns:
    traveller_data["cluster"] = kmeans.predict(traveller_data.drop(columns=["TRAVELER_ID"]))


# 사용자 기반 협업 필터링 (User-Based)
def recommend_users_based(user_cluster, traveller_data, k=5):
    cluster_users = traveller_data[traveller_data["cluster"] == user_cluster]
    user_features = cluster_users.drop(columns=["TRAVELER_ID", "cluster"])

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(user_features)

    distances, indices = knn.kneighbors(user_features)
    recommended_users = cluster_users.iloc[indices[0]][["TRAVELER_ID", "cluster"]]
    return recommended_users


# 아이템 기반 협업 필터링 (Item-Based) + WR 보완
def recommend_items_based(user_cluster, travel_data, visit_area_data, k=5):

    # 해당 클러스터의 여행 ID 찾기
    cluster_travel_ids = travel_data[travel_data["cluster"] == user_cluster]["TRAVEL_ID"].values

    # 방문지 데이터에서 해당 클러스터의 여행지 필터링
    cluster_visits = visit_area_data[visit_area_data["TRAVEL_ID"].isin(cluster_travel_ids)]

    # WR의 기존 최소/최대 값
    WR_min, WR_max = cluster_visits["WR"].min(), cluster_visits["WR"].max()

    # 최근접 이웃 모델로 관광지 추천 (W 기반)
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(cluster_visits[["W"]])

    distances, indices = knn.kneighbors(cluster_visits[["W"]])

    # 추천된 관광지 리스트 및 유사도 점수
    recommended_items = cluster_visits.iloc[indices[0]][["VISIT_AREA_NM", "W", "WR"]]

    # WR을 1~5 범위로 변환하여 보정
    recommended_items["Adjusted_WR"] = (recommended_items["WR"] - WR_min) / (WR_max - WR_min) * 4 + 1

    # 최종 추천 점수 계산 (W 60% + 변환된 WR 40%)
    recommended_items["Final_Score"] = recommended_items["W"] * 0.6 + recommended_items["Adjusted_WR"] * 0.4
    recommended_items = recommended_items.sort_values(by="Final_Score", ascending=False)

    return recommended_items[["VISIT_AREA_NM", "Final_Score"]].values.tolist()


# 신규 사용자 클러스터 할당 및 추천
def collaborative_filtering(nickname, gender, age, companion, stay_duration, duration_min, duration_max):
    try:
        # 사용자 클러스터 예측 (clustering.py에서 불러옴)
        user_cluster = predict_user_cluster(nickname, gender, age, companion, stay_duration, duration_min, duration_max)

        # 통합 협업 필터링 수행
        recommended_items = recommend_items_based(user_cluster, travel_data, visit_area_data, k=5)

        return recommended_items

    except Exception as e:
        raise RuntimeError(f"협업 필터링 중 오류 발생: {e}")


# 신규 사용자 예제로 테스트 실행
new_user_recommendations = collaborative_filtering("jeju_lover", "Female", "20s", "2인 가족 여행", 1, 2, 14)

# 중복 제거 후 최종 추천 리스트 생성
unique_recommendations = {}
for name, score in new_user_recommendations:
    if name not in unique_recommendations:
        unique_recommendations[name] = score

# 추천 결과 출력 (중복 제거 후)
print("신규 사용자 추천 관광지:")
for name, score in sorted(unique_recommendations.items(), key=lambda x: x[1], reverse=True):
    print(f" - {name}: 추천 점수 {score:.3f}")

























