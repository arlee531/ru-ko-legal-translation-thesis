"""
러시아어-한국어 법률 번역 모델 선택 과정
Model Selection Process for Russian-Korean Legal Translation

본 스크립트는 논문 5.1.2 "모델 선택 및 학습 환경"에서 기술된 
여러 모델 시도 과정과 시행착오를 재현합니다.

Author: [Ahreum Lee]
Date: 2025-09-11
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel
import torch
import gc
from typing import Optional, Tuple

def clear_memory():
    """GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
