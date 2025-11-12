"""
러시아어-한국어 법률 번역 모델 선택 과정
Model Selection Process for Russian-Korean Legal Translation

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
