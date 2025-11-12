"""
mBART-50 Fine-tuning for Russian-Korean Legal Translation
러시아어-한국어 법률 번역을 위한 mBART-50 미세조정
"""

# ============================================
# 1. 환경 설정 및 라이브러리 설치
# ============================================

!pip install transformers datasets sentencepiece accelerate evaluate sacrebleu

# ============================================
# 2. 라이브러리 임포트
# ============================================

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import torch

# ============================================
# 3. 데이터 로딩
# ============================================

# CSV 파일 로드
train_df = pd.read_csv('train_v2.csv')
dev_df = pd.read_csv('dev_v2.csv')
test_df = pd.read_csv('test_v2.csv')

# 컬럼명 표준화
train_df = train_df.rename(columns={'ru': 'source_text', 'ko': 'target_text'})
dev_df = dev_df.rename(columns={'ru': 'source_text', 'ko': 'target_text'})
test_df = test_df.rename(columns={'ru': 'source_text', 'ko': 'target_text'})

# Hugging Face Dataset 형식으로 변환
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': dev_dataset,
    'test': test_dataset
})

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(dev_dataset)}")
print(f"Test size: {len(test_dataset)}")

# ============================================
# 4. 모델 및 토크나이저 로드
# ============================================

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# 언어 코드 설정
src_lang = "ru_RU"
tgt_lang = "ko_KR"
tokenizer.src_lang = src_lang

# ============================================
# 5. 전처리 함수 정의
# ============================================

max_length = 256

def preprocess_function(examples):
    """
    러시아어 소스와 한국어 타겟 텍스트를 토큰화
    """
    # 소스 텍스트 토큰화
    model_inputs = tokenizer(
        examples['source_text'],
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    
    # 타겟 텍스트 토큰화 (언어 코드 강제 설정)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['target_text'],
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
    
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# ============================================
# 6. 데이터셋 전처리 적용
# ============================================

tokenized_datasets = dataset_dict.map(
    preprocess_function,
    batched=True,
    remove_columns=['source_text', 'target_text']
)

# ============================================
# 7. Data Collator 설정
# ============================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# ============================================
# 8. 학습 설정
# ============================================

training_args = Seq2SeqTrainingArguments(
    output_dir="./mbart-legal-ru-ko",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    warmup_steps=300,
    logging_steps=100,
    save_steps=1000,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=256,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    report_to="none"
)

# ============================================
# 9. Trainer 초기화
# ============================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# ============================================
# 10. 학습 실행
# ============================================

print("=" * 50)
print("학습 시작")
print("=" * 50)

trainer.train()

print("=" * 50)
print("학습 완료")
print("=" * 50)

# ============================================
# 11. 모델 저장
# ============================================

trainer.save_model("./mbart-legal-ru-ko-final")
tokenizer.save_pretrained("./mbart-legal-ru-ko-final")

print("모델이 './mbart-legal-ru-ko-final'에 저장되었습니다.")

# ============================================
# 12. 테스트 번역 예시
# ============================================

def translate(text, src_lang="ru_RU", tgt_lang="ko_KR"):
    """
    러시아어를 한국어로 번역하는 함수
    """
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
    
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=256
    )
    
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

# 테스트 예시
test_text = "Статья 1. Российская Федерация - Россия есть демократическое федеративное правовое государство с республиканской формой правления."
translation = translate(test_text)
print(f"\n원문: {test_text}")
print(f"번역: {translation}")
