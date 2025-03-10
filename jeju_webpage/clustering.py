import pandas as pd
import joblib
import os

MODEL_PATH = "cluster_model.pkl"
DATA_FOLDER = "jeju_webpage/data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# K-Means 모델 로드
if os.path.exists(MODEL_PATH):
    kmeans = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"❌ '{MODEL_PATH}' 파일이 존재하지 않습니다. 같은 디렉토리에 있는지 확인하세요.")

# 기존 데이터 로드 (사용자 및 여행 기록)
traveller_data_path = r"E:\AI_HUB\data\concatenated\tn_traveller_concatenated_cleaned.csv"
travel_data_path = r"E:\AI_HUB\data\concatenated\tn_travel_concatenated.csv"

traveller_data = pd.read_csv(traveller_data_path)
travel_data = pd.read_csv(travel_data_path)

# 기존 사용자들의 클러스터 추가 (없으면 KMeans 예측)
if "cluster" not in traveller_data.columns:
    feature_columns = [col for col in traveller_data.columns if col not in ["TRAVELER_ID"]]
    traveller_data["cluster"] = kmeans.predict(traveller_data[feature_columns])

# 여행 기록 데이터에도 클러스터 추가 (TRAVELER_ID 기반 매핑)
if "TRAVELER_ID" in travel_data.columns and "cluster" not in travel_data.columns:
    travel_data = travel_data.merge(traveller_data[["TRAVELER_ID", "cluster"]], on="TRAVELER_ID", how="left")

# 기존 사용자 데이터 저장 (업데이트된 클러스터 포함)
traveller_data.to_csv(traveller_data_path, index=False)
travel_data.to_csv(travel_data_path, index=False)

def preprocess_user_input(nickname, gender, age, companion, stay_duration, duration_min, duration_max):
    """ 신규 사용자 입력을 기존 데이터 형식으로 변환하는 함수 """

    # 기존 데이터에서 사용한 동반자 유형 리스트
    companion_types = [
        "2인 가족 여행", "2인 여행 (가족 외)", "기타",
        "나홀로 여행", "부모 동반 여행", "자녀 동반 여행"
    ]

    # 성별 & 연령대 인코딩
    gender_map = {"Male": 0, "Female": 1}
    age_map = {"20s": 0, "30s": 1, "40s": 2, "50s": 3, "60s": 4}

    # 체류 기간 정규화 (ZeroDivisionError 방지)
    duration_range = duration_max - duration_min
    duration_normalized = (stay_duration - duration_min) / duration_range if duration_range != 0 else 0.5

    # 동반자 유형 원핫 인코딩
    companion_df = pd.DataFrame([[companion]], columns=["ACCOMPANY_CATEGORY"])
    companion_one_hot = pd.get_dummies(companion_df, columns=["ACCOMPANY_CATEGORY"])

    # 기존 데이터와 동일한 컬럼 유지
    expected_columns = ["ACCOMPANY_CATEGORY_" + c for c in companion_types]
    for col in expected_columns:
        if col not in companion_one_hot:
            companion_one_hot[col] = 0

    # 성별, 연령대, 정규화된 체류 기간을 DataFrame으로 변환
    user_data = pd.DataFrame([[gender_map.get(gender, -1), age_map.get(age, -1), duration_normalized]],
                             columns=["GENDER", "AGE_GRP", "Duration"])

    # 원핫 인코딩된 동반자 유형 추가
    user_data = pd.concat([user_data, companion_one_hot], axis=1)

    # 컬럼 순서 맞추기
    final_columns = ["GENDER", "AGE_GRP", "Duration"] + expected_columns
    user_data = user_data[final_columns]

    return user_data

def predict_user_cluster(nickname, gender, age, companion, stay_duration, duration_min, duration_max):
    """ 신규 사용자 클러스터 예측 및 저장 """
    try:
        # 신규 사용자 입력 전처리
        user_data = preprocess_user_input(nickname, gender, age, companion, stay_duration, duration_min, duration_max)

        # K-Means 클러스터 예측
        cluster = kmeans.predict(user_data)[0]

        # 사용자 정보 저장
        file_path = os.path.join(DATA_FOLDER, "user_info.csv")
        user_df = pd.DataFrame([{
            "nickname": nickname,
            "gender": gender,
            "age": age,
            "companion": companion,
            "stay_duration": stay_duration,
            "cluster": cluster
        }])

        if os.path.exists(file_path):
            user_df.to_csv(file_path, mode='a', header=False, index=False, encoding="utf-8-sig")
        else:
            user_df.to_csv(file_path, index=False, encoding="utf-8-sig")

        return cluster

    except Exception as e:
        raise RuntimeError(f"❌ 클러스터 예측 중 오류 발생: {e}")
