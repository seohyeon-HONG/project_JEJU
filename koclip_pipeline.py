import os
import torch
import numpy as np
import pickle
import pandas as pd
import re
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
import warnings

warnings.filterwarnings("ignore")


class KoCLIPEmbeddingProcessor:
    def __init__(self, model_name: str = "koclip/koclip-base-pt", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

        print(f"사용 중인 디바이스: {self.device}")

    def load_model(self) -> None:
        print("KoCLIP 모델 로딩 중...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        print("KoCLIP 모델 로딩 완료")

    def extract_texts(self, text_str: str) -> List[str]:
        if not isinstance(text_str, str):
            return []

        match = re.search(r'\[(.*)\]', text_str)
        if not match:
            return []

        texts = re.findall(r'\'([^\']+)\'', match.group(1))
        return list(set(texts))  # 중복 제거

    def extract_paths(self, path_str: str) -> List[str]:
        if not isinstance(path_str, str):
            return []

        match = re.search(r'\[(.*)\]', path_str)
        if not match:
            return []

        paths = re.findall(r'\'([^\']+)\'', match.group(1))
        return list(set(paths))  # 중복 제거

    def get_koclip_embeddings(self, texts: List[str], image_paths: List[str],
                              max_chars: int = 500) -> Dict[str, List[np.ndarray]]:
        text_embeddings = []
        image_embeddings = []

        # 텍스트 처리
        processed_texts = [text[:max_chars] for text in texts]
        for text in processed_texts:
            try:
                text_inputs = self.processor(text=text, return_tensors="pt",
                                             padding=True).to(self.device)
                with torch.no_grad():
                    text_features = self.model.get_text_features(**text_inputs)
                    text_emb = text_features.cpu().numpy()[0].copy()
                    text_embeddings.append(text_emb)
            except Exception as e:
                print(f"텍스트 처리 오류 ({text[:30]}...): {e}")
                continue

        # 이미지 처리
        valid_images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                valid_images.append(img)
            except Exception as e:
                print(f"이미지 로드 오류 ({img_path}): {e}")

        for img in valid_images:
            try:
                image_inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**image_inputs)
                    image_emb = image_features.cpu().numpy()[0].copy()
                    image_embeddings.append(image_emb)
            except Exception as e:
                print(f"이미지 처리 오류: {e}")
                continue

        return {
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings
        }

    def generate_enhanced_multimodal_embeddings(self, text_embeddings: List[np.ndarray],
                                                image_embeddings: List[np.ndarray]) -> np.ndarray:
        if len(text_embeddings) == 0 or len(image_embeddings) == 0:
            return None

        text_tensor = torch.tensor(np.array(text_embeddings)).to(self.device)
        image_tensor = torch.tensor(np.array(image_embeddings)).to(self.device)

        text_tensor = text_tensor / text_tensor.norm(dim=1, keepdim=True)
        image_tensor = image_tensor / image_tensor.norm(dim=1, keepdim=True)

        similarity = torch.mm(text_tensor, image_tensor.t())
        similarity = similarity.cpu().numpy()

        enhanced_embeddings = []

        # 각 텍스트에 대해 이미지 임베딩의 가중 평균 계산
        for i, text_emb in enumerate(text_embeddings):
            weights = similarity[i]
            weights = np.exp(weights) / np.sum(np.exp(weights))

            weighted_image_emb = np.zeros_like(image_embeddings[0])
            for j, img_emb in enumerate(image_embeddings):
                weighted_image_emb += weights[j] * img_emb

            combined_emb = np.concatenate([text_emb, weighted_image_emb])
            enhanced_embeddings.append(combined_emb)

        final_embedding = np.mean(enhanced_embeddings, axis=0)
        return final_embedding

    def process_dataset(self, df: pd.DataFrame, text_col: str = 'REVIEW_TEXTS',
                        image_col: str = 'IMAGE_PATHS', uid_col: str = 'UNIQUE_VISIT_ID',
                        name_col: str = 'VISIT_AREA_NM') -> Dict[int, Dict[str, Any]]:
        if self.model is None or self.processor is None:
            self.load_model()

        results = {}

        print("KoCLIP 임베딩 생성 시작")
        for i, row in tqdm(df.iterrows(), total=len(df), desc="임베딩 생성"):
            if i % 10 == 0:
                print(f"처리 중: {i}/{len(df)}")

            uid = row[uid_col]
            place_name = row[name_col]

            review_texts = self.extract_texts(row[text_col])
            image_paths = self.extract_paths(row[image_col])

            if not review_texts or not image_paths:
                print(f"행 {i}: 텍스트 또는 이미지 없음, 건너뜀")
                continue

            valid_paths = [p for p in image_paths if os.path.exists(p)]
            if not valid_paths:
                print(f"행 {i}: 유효한 이미지 경로 없음, 건너뜀")
                continue

            try:
                embeddings = self.get_koclip_embeddings(review_texts, valid_paths)

                if not embeddings["text_embeddings"] or not embeddings["image_embeddings"]:
                    print(f"행 {i}: 임베딩 생성 실패, 건너뜀")
                    continue

                enhanced_embedding = self.generate_enhanced_multimodal_embeddings(
                    embeddings["text_embeddings"],
                    embeddings["image_embeddings"]
                )

                if enhanced_embedding is not None:
                    results[uid] = {
                        'UNIQUE_VISIT_ID': uid,
                        'VISIT_AREA_NM': place_name,
                        'TEXT_EMBEDDINGS': embeddings["text_embeddings"],
                        'IMAGE_EMBEDDINGS': embeddings["image_embeddings"],
                        'ENHANCED_EMBEDDINGS': enhanced_embedding,
                        'NUM_TEXTS': len(embeddings["text_embeddings"]),
                        'NUM_IMAGES': len(embeddings["image_embeddings"])
                    }

            except Exception as e:
                print(f"오류 발생 (행 {i}): {e}")
                continue

        return results

    def save_embeddings(self, embeddings: Dict[int, Dict[str, Any]],
                        output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)

        print(f"임베딩이 '{output_path}'에 저장되었습니다. (총 {len(embeddings)}개 항목)")

    def load_embeddings(self, file_path: str) -> Dict[int, Dict[str, Any]]:
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings



def main():
    # 설정
    CONFIG = {
        'multi_embedding_df_path': "/content/drive/MyDrive/Project_JEJU/dataset/embedding_data.csv",
        'output_path': "/content/drive/MyDrive/Project_JEJU/modeling/embeddings_pkl/enhanced_koclip_embeddings.pkl",
        'model_name': "koclip/koclip-base-pt",
        'max_chars': 500
    }

    # 프로세서 초기화
    processor = KoCLIPEmbeddingProcessor(model_name=CONFIG['model_name'])

    # 데이터 로드
    print("데이터 로드 중...")
    df = pd.read_csv(CONFIG['multi_embedding_df_path'])
    print(f"데이터 로드 완료: {len(df)}개 행")

    # 임베딩 생성
    embeddings = processor.process_dataset(df)

    # 결과 저장
    processor.save_embeddings(embeddings, CONFIG['output_path'])


if __name__ == "__main__":
    main()