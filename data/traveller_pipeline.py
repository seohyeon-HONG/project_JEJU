import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple


class PersonaProcessor:

    def __init__(self, n_components: int = 7, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)

        # PC 차원별 의미 정의
        self.pc_meanings = {
            'PC1': 'SNS 인생샷, 과시, 사진촬영 중요',
            'PC2': '테마파크/놀이시설/동식물원, 교육/체험 프로그램',
            'PC3': '정신육체적 휴식, 휴식형, 진정감',
            'PC4': '익숙한 지역, 담일 도시, 유명 관광지',
            'PC5': '야외 스포츠, 유흥오락, 일상탈출, 쇼핑',
            'PC6': '건강/운동, 신체 등 특별한 목적',
            'PC7': '일상탈출, 신규 여행지 발굴'
        }

        self.is_fitted = False

    def select_persona_features(self, persona_df: pd.DataFrame) -> pd.DataFrame:
        """
        페르소나 특성 중 필요한 컬럼만 선택
        """
        selected_columns = [col for col in persona_df.columns if (
                col == 'TRAVELER_ID' or
                col.startswith('STYLE_AXIS_') or
                col.startswith('MOTIVE_') or
                col.startswith('MISSION_W_')
        )]

        return persona_df[selected_columns].copy()

    def fit_transform(self, persona_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
        """
        PCA 학습 및 변환 수행
        """
        print("페르소나 차원 축소 중...")

        selected_df = self.select_persona_features(persona_df)

        # ID 열 제외한 감성 특성만 추출
        id_cols = ['TRAVELER_ID']
        feature_cols = [col for col in selected_df.columns if col not in id_cols]
        X = selected_df[feature_cols].copy()

        # 표준화
        X_scaled = self.scaler.fit_transform(X)

        # PCA 차원 축소
        X_pca = self.pca.fit_transform(X_scaled)

        # PCA 결과를 데이터프레임으로 변환
        pca_columns = [f'PC{i + 1}' for i in range(self.n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns)

        # 원본 데이터와 PCA 결과 결합
        result_df = pd.concat([
            selected_df.reset_index(drop=True),
            pca_df
        ], axis=1)

        # 사용자별 PC 점수 딕셔너리 생성
        traveler_pc_scores = {}
        for _, row in result_df.iterrows():
            traveler_id = row['TRAVELER_ID']
            pc_scores = row[pca_columns].values.astype(np.float32)
            traveler_pc_scores[traveler_id] = pc_scores

        self.is_fitted = True

        # 결과 출력
        self._print_pca_results()

        return result_df, traveler_pc_scores

    def transform(self, persona_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
        """
        이미 학습된 PCA로 새로운 데이터 변환

        Args:
            persona_df: 변환할 페르소나 데이터프레임

        Returns:
            변환된 데이터프레임과 사용자별 PC 점수 딕셔너리
        """
        if not self.is_fitted:
            raise ValueError("PCA가 아직 학습되지 않았습니다. fit_transform을 먼저 호출하세요.")

        selected_df = self.select_persona_features(persona_df)

        id_cols = ['TRAVELER_ID']
        feature_cols = [col for col in selected_df.columns if col not in id_cols]
        X = selected_df[feature_cols].copy()

        # 표준화 및 PCA 변환
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        # 결과 정리
        pca_columns = [f'PC{i + 1}' for i in range(self.n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns)

        result_df = pd.concat([
            selected_df.reset_index(drop=True),
            pca_df
        ], axis=1)

        # 사용자별 PC 점수 딕셔너리 생성
        traveler_pc_scores = {}
        for _, row in result_df.iterrows():
            traveler_id = row['TRAVELER_ID']
            pc_scores = row[pca_columns].values.astype(np.float32)
            traveler_pc_scores[traveler_id] = pc_scores

        return result_df, traveler_pc_scores

    def _print_pca_results(self):
        """PCA 결과 출력"""
        explained_variance_ratio = self.pca.explained_variance_ratio_
        print(f"PCA 설명 비율: {explained_variance_ratio}")
        print(f"누적 설명 비율: {np.sum(explained_variance_ratio):.4f}")

        print("\nPC 차원별 의미:")
        for i in range(self.n_components):
            pc_col = f'PC{i + 1}'
            meaning = self.pc_meanings.get(pc_col, '')
            variance = explained_variance_ratio[i]
            print(f"{pc_col}: {meaning} (설명력: {variance:.3f})")

    def save_processor(self, save_path: str):
        """
        학습된 PCA 프로세서 저장

        Args:
            save_path: 저장 경로
        """
        if not self.is_fitted:
            raise ValueError("PCA가 아직 학습되지 않았습니다.")

        processor_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'n_components': self.n_components,
            'pc_meanings': self.pc_meanings,
            'explained_variance_ratio': self.pca.explained_variance_ratio_
        }

        with open(save_path, 'wb') as f:
            pickle.dump(processor_data, f)

        print(f"PCA 프로세서가 {save_path}에 저장되었습니다.")

    def load_processor(self, load_path: str):
        """
        저장된 PCA 프로세서 로드

        Args:
            load_path: 로드 경로
        """
        with open(load_path, 'rb') as f:
            processor_data = pickle.load(f)

        self.scaler = processor_data['scaler']
        self.pca = processor_data['pca']
        self.n_components = processor_data['n_components']
        self.pc_meanings = processor_data['pc_meanings']
        self.is_fitted = True

        print(f"PCA 프로세서가 {load_path}에서 로드되었습니다.")
        print(f"설명 비율: {processor_data['explained_variance_ratio']}")
