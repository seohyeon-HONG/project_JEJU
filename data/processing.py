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
        """
        scores_reshaped = scores.reshape(-1, 1)

        # Yeo-Johnson 변환
        yeo_transformed = self.power_transformer.fit_transform(scores_reshaped)

        # MinMax 스케일링
        minmax_transformed = self.minmax_scaler.fit_transform(yeo_transformed)

        self.is_fitted = True

        return yeo_transformed.flatten(), minmax_transformed.flatten()

    def _print_normalization_results(self, df: pd.DataFrame):
        """정규화 결과 출력"""
        print("\n정규화 전후 점수 분포:")
        print(df[['TARGET_SCORE', 'TARGET_SCORE_YEO', 'TARGET_SCORE_FINAL']].describe())

    def persona_adaptive_normalize(self,
                                   df: pd.DataFrame,
                                   persona_df_pca: pd.DataFrame) -> pd.DataFrame:
        """
        페르소나별 적응적 정규화 수행
        """
        df = df.copy()

        # 주요 PC 차원 선택
        pc_columns = [f'PC{i+1}' for i in range(self.top_pc_dims)]

        # 각 사용자의 주요 PC 차원 값 추출
        user_pc_values = {}
        for _, row in persona_df_pca.iterrows():
            user_id = row['TRAVELER_ID']
            pc_values = row[pc_columns].values
            user_pc_values[user_id] = pc_values

        # 감성 강도 계산 (PC 값의 절대값 평균)
        persona_intensities = {}
        for user_id, pc_values in user_pc_values.items():
            # 감성 강도는 0.5~2.0 범위로 제한
            intensity = 1.0 + np.clip(np.mean(np.abs(pc_values)), 0.0, 1.0)
            persona_intensities[user_id] = intensity

        # S-커브 변환 함수 정의
        def s_curve_transform(score, intensity):
            distance = score - 0.5
            steepness = intensity * 5.0
            transformed = 0.5 + distance * (2.0 / (1.0 + np.exp(-steepness * distance)))
            return np.clip(transformed, 0.0, 1.0)

        adjusted_scores = []
        for _, row in df.iterrows():
            user_id = row['TRAVELER_ID']
            global_score = row['TARGET_SCORE_GLOBAL']
            intensity = persona_intensities.get(user_id, 1.0)

            adjusted = s_curve_transform(global_score, intensity)
            adjusted_scores.append(adjusted)

        df['TARGET_SCORE_ADJUSTED'] = adjusted_scores
        return df

    def normalize_target_scores(self,
                                df: pd.DataFrame,
                                persona_df_pca: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        타겟 점수 정규화 메인 함수
        """
        # 1. TARGET_SCORE 계산
        df = self.compute_target_score(df)

        # 2. 전역 정규화
        scores = df['TARGET_SCORE'].values
        yeo_scores, global_scores = self.global_normalize(scores)

        df['TARGET_SCORE_YEO'] = yeo_scores
        df['TARGET_SCORE_GLOBAL'] = global_scores

        # 3. 페르소나별 적응적 정규화 
        if self.use_persona_adaptation and persona_df_pca is not None:
            df = self.persona_adaptive_normalize(df, persona_df_pca)
            df['TARGET_SCORE_FINAL'] = df['TARGET_SCORE_ADJUSTED']
        else:
            df['TARGET_SCORE_FINAL'] = df['TARGET_SCORE_GLOBAL']

        # 4. 결과 출력
        self._print_normalization_results(df)

        return df

