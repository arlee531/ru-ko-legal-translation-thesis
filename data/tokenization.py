"""
NLLB 모델을 위한 토큰화 처리
Tokenization Process for NLLB Model

본 스크립트는 논문 5.1.1에서 기술된 토큰화 과정을 담당하며,
러시아어-한국어 법률 문서의 NLLB 모델 입력 형태 변환을 수행합니다.

Author: [Ahreum Lee]
Date: 2025-09-11
"""

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalTextTokenizer:
    """법률 텍스트 전용 토큰화 클래스"""
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.tokenizer = None
        self.source_lang = "rus_Cyrl"  # 러시아어
        self.target_lang = "kor_Hang"  # 한국어
        
        # 토큰화 설정
        self.max_length = 256  # 법률 문서의 긴 문장 고려
        self.padding = True
        self.truncation = True
