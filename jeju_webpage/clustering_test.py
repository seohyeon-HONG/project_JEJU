from project_JEJU.jeju_webpage.clustering import predict_user_cluster, preprocess_user_input
import joblib

# 기존 데이터에서 구한 min/max 값 (예제)
duration_min = 1  # 기존 데이터에서 최소 체류 기간
duration_max = 14  # 기존 데이터에서 최대 체류 기간

# KMeans 모델 로드 (t-SNE 없이 학습된 모델 사용)
kmeans = joblib.load("cluster_model.pkl")

# 테스트 사용자 입력
test_nickname = "여행자1"
test_gender = "Female"
test_age = "60s"
test_companion = "나홀로 여행"
test_stay_duration = 3

# 사용자 입력을 기존 데이터와 동일한 형식으로 변환
test_user_data = preprocess_user_input(
    test_nickname, test_gender, test_age, test_companion, test_stay_duration, duration_min, duration_max
)

print("변환된 사용자 입력 데이터:")
print(test_user_data)

# 클러스터 예측 실행
test_result = predict_user_cluster(
    test_nickname, test_gender, test_age, test_companion, test_stay_duration, duration_min, duration_max
)

print(f"예측된 클러스터: {test_result}")


