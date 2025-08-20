import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from typing import Optional, Tuple, Dict


class TargetScoreNormalizer:
    """
    타겟 점수 정규화
    전역 정규화 / 페르소나별 적응적 정규화
    """

    def __init__(self,
                 w_weight: float = 0.6,
                 wr_weight: float = 0.4,
                 use_persona_adaptation: bool = True,
                 top_pc_dims: int = 3):

        self.w_weight = w_weight
        self.wr_weight = wr_weight
        self.use_persona_adaptation = use_persona_adaptation
        self.top_pc_dims = top_pc_dims

        self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        self.minmax_scaler = MinMaxScaler()

        self.is_fitted = False

    def compute_target_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['TARGET_SCORE'] = self.w_weight * df['W'] + self.wr_weight * df['WR']
        return df

    def global_normalize(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        전역 정규화: Yeo-Johnson 변환 + MinMax 스케일링

        Args:
            scores: 정규화할 점수 배열

        Returns:
            Yeo-Johnson 변환 결과, MinMax 스케일링 결과
        """
        scores_reshaped = scores.reshape(-1, 1)

        # Yeo-Johnson 변환
        yeo_transformed = self.power_transformer.fit_transform(scores_reshaped)

        # MinMax 스케일링
        minmax_transformed = self.minmax_scaler.fit_transform(yeo_transformed)

        self.is_fitted = True

        return yeo_transformed.flatten(), minmax_transformed.flatten()

    def _compute_persona_intensities(self, persona_df_pca: pd.DataFrame) -> Dict[str, float]:
        """
        페르소나별 감성 강도 계산

        Args:
            persona_df_pca: PCA 변환된 페르소나 데이터프레임

        Returns:
            사용자 ID별 감성 강도 딕셔너리
        """
        pc_columns = [f'PC{i + 1}' for i in range(self.top_pc_dims)]

        persona_intensities = {}
        for _, row in persona_df_pca.iterrows():
            user_id = row['TRAVELER_ID']
            pc_values = row[pc_columns].values

            # 감성 강도는 0.5~2.0 범위로 제한
            intensity = 1.0 + np.clip(np.mean(np.abs(pc_values)), 0.0, 1.0)
            persona_intensities[user_id] = intensity

        return persona_intensities

    def _s_curve_transform(self, score: float, intensity: float) -> float:
        """
        S-커브 변환 함수

        Args:
            score: 변환할 점수 (0-1 범위)
            intensity: 감성 강도

        Returns:
            변환된 점수
        """
        # 0.5를 중심으로 거리 계산
        distance = score - 0.5

        # S-커브 적용 (시그모이드 기반)
        steepness = intensity * 5.0  # 기울기 조절 파라미터
        transformed = 0.5 + distance * (2.0 / (1.0 + np.exp(-steepness * distance)))

        # 범위 제한
        return np.clip(transformed, 0.0, 1.0)

    def persona_adaptive_normalize(self,
                                   df: pd.DataFrame,
                                   persona_df_pca: pd.DataFrame) -> pd.DataFrame:
        """
        페르소나별 적응적 정규화 수행

        Args:
            df: 정규화할 데이터프레임 (TARGET_SCORE_GLOBAL 컬럼 필요)
            persona_df_pca: PCA 변환된 페르소나 데이터프레임

        Returns:
            적응적 정규화가 적용된 데이터프레임
        """
        df = df.copy()

        # 페르소나별 감성 강도 계산
        persona_intensities = self._compute_persona_intensities(persona_df_pca)

        # 조정된 점수 계산
        adjusted_scores = []
        for _, row in df.iterrows():
            user_id = row['TRAVELER_ID']
            global_score = row['TARGET_SCORE_GLOBAL']
            intensity = persona_intensities.get(user_id, 1.0)

            # S-커브 변환 적용
            adjusted = self._s_curve_transform(global_score, intensity)
            adjusted_scores.append(adjusted)

        df['TARGET_SCORE_ADJUSTED'] = adjusted_scores
        return df

    def normalize_target_scores(self,
                                df: pd.DataFrame,
                                persona_df_pca: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        타겟 점수 정규화 메인 함수

        Args:
            df: 정규화할 데이터프레임 (W, WR 컬럼 필요)
            persona_df_pca: 페르소나별 적응적 정규화를 위한 PCA 데이터프레임

        Returns:
            정규화된 타겟 점수가 포함된 데이터프레임
        """
        print("타겟 점수 정규화 중...")

        # 1. TARGET_SCORE 계산
        df = self.compute_target_score(df)

        # 2. 전역 정규화
        scores = df['TARGET_SCORE'].values
        yeo_scores, global_scores = self.global_normalize(scores)

        df['TARGET_SCORE_YEO'] = yeo_scores
        df['TARGET_SCORE_GLOBAL'] = global_scores

        # 3. 페르소나별 적응적 정규화 (선택적)
        if self.use_persona_adaptation and persona_df_pca is not None:
            print("페르소나별 적응적 정규화 수행 중...")
            df = self.persona_adaptive_normalize(df, persona_df_pca)
            df['TARGET_SCORE_FINAL'] = df['TARGET_SCORE_ADJUSTED']
        else:
            df['TARGET_SCORE_FINAL'] = df['TARGET_SCORE_GLOBAL']

        # 4. 결과 출력
        self._print_normalization_results(df)

        return df


class DataFilter:
    """
    데이터 필터링을 담당하는 클래스
    """

    @staticmethod
    def filter_valid_logs(df: pd.DataFrame,
                          traveler_features: Dict,
                          attraction_features: Dict) -> pd.DataFrame:
        """
        유효한 데이터만 필터링 (페르소나와 관광지 특성이 모두 있는 경우)

        Args:
            df: 필터링할 데이터프레임
            traveler_features: 사용자 특성 딕셔너리
            attraction_features: 관광지 특성 딕셔너리

        Returns:
            필터링된 데이터프레임
        """
        print("유효한 데이터 필터링 중...")

        filtered_logs = []
        for _, row in df.iterrows():
            t_id = row['TRAVELER_ID']
            a_id = int(row['UNIQUE_VISIT_ID'])

            if t_id in traveler_features and a_id in attraction_features:
                filtered_logs.append(row)

        filtered_df = pd.DataFrame(filtered_logs)

        print(f"필터링 완료: {len(df)} → {len(filtered_df)} 행")
        return filtered_df

    @staticmethod
    def filter_min_visits(df: pd.DataFrame, min_visits: int = 5) -> pd.DataFrame:
        """
        최소 방문 수 이상의 사용자만 필터링

        Args:
            df: 필터링할 데이터프레임
            min_visits: 최소 방문 수

        Returns:
            필터링된 데이터프레임
        """
        user_visit_counts = df['TRAVELER_ID'].value_counts()
        valid_users = user_visit_counts[user_visit_counts >= min_visits].index

        filtered_df = df[df['TRAVELER_ID'].isin(valid_users)]

        print(f"최소 방문 수 ({min_visits}회) 필터링: {len(df)} → {len(filtered_df)} 행")
        return filtered_df
