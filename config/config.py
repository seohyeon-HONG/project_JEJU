import random
import numpy as np
import torch


class Config:
    SEED = 42
    DATA_PATH = "/content/drive/MyDrive/Project_JEJU/dataset"
    EMBEDDINGS_PATH = "/content/drive/MyDrive/Project_JEJU/modeling/embeddings_pkl"
    OUTPUT_PATH = "/content/drive/MyDrive/Project_JEJU/models/0412"

    NUM_PC_DIMS = 7  
    HIDDEN_DIM = 256  
    NUM_HEADS = 4  
    PERSONA_EXPANSION_FACTOR = 2  
    EMOTION_EMBEDDING_DIM = 128 

    NUM_EPOCHS = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0003
    DROPOUT_RATE = 0.25  
    WEIGHT_DECAY = 2e-4 

    INTERNAL_CONTRAST_WEIGHT = 0.3  # 내부 대조 가중치 (LIGHT)
    EXTERNAL_CONTRAST_WEIGHT = 0.7  # 외부 대조 가중치 (STRONG)
    CONTRASTIVE_TEMPERATURE = 0.5  

    MSE_WEIGHT = 0.5  
    BPR_WEIGHT = 0.3  
    REG_WEIGHT = 0.1  
    FILTER_REG_WEIGHT = 0.1  

    TOP_K_VALUES = [5, 10]  
    MIN_VISITS = 5  
    N_SPLITS = 3  


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
