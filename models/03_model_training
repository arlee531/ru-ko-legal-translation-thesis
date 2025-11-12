#!/usr/bin/env python3
"""
Legal Translation Model Training Script
러시아어-한국어 법률 번역 모델 훈련 스크립트

이 스크립트는 준비된 법률 도메인 데이터셋을 사용하여
러시아어-한국어 번역 모델을 훈련합니다.
"""

import os
import json
import torch
import warnings
from datetime import datetime
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import numpy as np
import evaluate

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning)

class LegalTranslationTrainer:
    """법률 번역 모델 훈련 클래스"""
    
    def __init__(self, config_path="config.json"):
        """
        초기화
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Training device: {self.device}")
        
        # 모델 및 토크나이저 초기화
        self.tokenizer = None
        self.model = None
        self.datasets = None
        self.trainer = None
        
    def load_config(self, config_path):
        """설정 파일 로드"""
        default_config = {
            "model": {
                "name": "facebook/nllb-200-distilled-600M",
                "source_lang": "rus_Cyrl",
                "target_lang": "kor_Hang"
            },
            "data": {
                "train_path": "data/processed/train.json",
                "validation_path": "data/processed/validation.json",
                "test_path": "data/processed/test.json",
                "max_length": 128
            },
            "training": {
                "output_dir": "./models/trained/nllb-legal-ru-ko",
                "num_train_epochs": 3,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "learning_rate": 1e-5,
                "warmup_steps": 300,
                "weight_decay": 0.01,
                "logging_steps": 100,
                "save_steps": 500,
                "eval_steps": 500,
                "fp16": True,
                "early_stopping_patience": 3
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 기본 설정과 사용자 설정 병합
                for key in user_config:
                    if key in default_config:
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
        
        return default_config
    
    def load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        model_name = self.config["model"]["name"]
        print(f"📦 Loading model: {model_name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            src_lang=self.config["model"]["source_lang"],
            tgt_lang=self.config["model"]["target_lang"]
        )
        
        # 모델 로드
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        print(f"✅ Model loaded successfully")
        print(f"   - Tokenizer vocab size: {len(self.tokenizer)}")
        print(f"   - Model parameters: {self.model.num_parameters():,}")
        
    def load_datasets(self):
        """데이터셋 로드 및 전처리"""
        print("📊 Loading datasets...")
        
        # 데이터 로드
        datasets = {}
        for split in ["train", "validation", "test"]:
            path = self.config["data"][f"{split}_path"]
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                datasets[split] = Dataset.from_dict(data)
                print(f"   - {split}: {len(datasets[split])} examples")
            else:
                print(f"⚠️  Warning: {path} not found, skipping {split} set")
        
        self.datasets = DatasetDict(datasets)
        
        # 토크나이징
        print("🔤 Tokenizing datasets...")
        self.datasets = self.datasets.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.datasets["train"].column_names
        )
        
        print("✅ Datasets prepared successfully")
        
    def tokenize_function(self, examples):
        """토크나이징 함수"""
        max_length = self.config["data"]["max_length"]
        
        # 소스 언어 설정
        self.tokenizer.src_lang = self.config["model"]["source_lang"]
        
        # 입력 토크나이징
        model_inputs = self.tokenizer(
            examples["russian"],
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # 타겟 토크나이징
        self.tokenizer.tgt_lang = self.config["model"]["target_lang"]
        labels = self.tokenizer(
            examples["korean"],
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def compute_metrics(self, eval_pred):
        """평가 메트릭 계산"""
        predictions, labels = eval_pred
        
        # 패딩 토큰 제거
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # 디코딩
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # BLEU 스코어 계산
        bleu = evaluate.load("bleu")
        result = bleu.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        
        return {
            "bleu": result["bleu"],
            "precisions": result["precisions"]
        }
    
    def setup_trainer(self):
        """트레이너 설정"""
        print("🎯 Setting up trainer...")
        
        # 훈련 인수 설정
        training_args = TrainingArguments(
            output_dir=self.config["training"]["output_dir"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            warmup_steps=self.config["training"]["warmup_steps"],
            weight_decay=self.config["training"]["weight_decay"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            evaluation_strategy="steps",
            save_strategy="steps",
            fp16=self.config["training"]["fp16"],
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            save_total_limit=3,
            predict_with_generate=True,
            generation_max_length=self.config["data"]["max_length"]
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # 콜백 리스트 준비
        callbacks = []
        
        # 조기 종료 콜백
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config["training"]["early_stopping_patience"]
        )
        callbacks.append(early_stopping)
        
        # 성능 모니터링 콜백 (사용 가능한 경우)
        if MONITORING_AVAILABLE:
            print("🔍 Setting up performance monitoring...")
            performance_monitor = create_performance_monitor(
                log_dir="./logs",
                checkpoint_dir="./checkpoints",
                final_model_path="./nllb-legal-ru-ko-final",
                logging_steps=self.config["training"]["logging_steps"],
                save_steps=self.config["training"]["save_steps"]
            )
            callbacks.append(performance_monitor)
            print("✅ Performance monitoring enabled")
            print("   - 200스텝마다 학습 진행 상황 기록")
            print("   - 1,000스텝마다 모델 체크포인트 저장")
            print("   - 워밍업 및 학습률 스케줄링 모니터링")
            print("   - 최종 모델을 './nllb-legal-ru-ko-final'에 저장")
        else:
            print("⚠️  Performance monitoring disabled - module not available")
        
        # 트레이너 초기화
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        print("✅ Trainer setup complete")
        
    def train(self):
        """모델 훈련"""
        print("🚀 Starting training...")
        print("-" * 60)
        
        # 훈련 정보 출력
        total_steps = len(self.datasets["train"]) // (
            self.config["training"]["per_device_train_batch_size"] * 
            self.config["training"]["gradient_accumulation_steps"]
        ) * self.config["training"]["num_train_epochs"]
        
        print(f"📊 Training Information:")
        print(f"   - Total examples: {len(self.datasets['train'])}")
        print(f"   - Batch size: {self.config['training']['per_device_train_batch_size']}")
        print(f"   - Gradient accumulation: {self.config['training']['gradient_accumulation_steps']}")
        print(f"   - Epochs: {self.config['training']['num_train_epochs']}")
        print(f"   - Total steps: {total_steps}")
        print(f"   - Learning rate: {self.config['training']['learning_rate']}")
        print("-" * 60)
        
        # 훈련 시작
        start_time = datetime.now()
        train_result = self.trainer.train()
        end_time = datetime.now()
        
        # 훈련 결과 출력
        training_time = end_time - start_time
        print(f"✅ Training completed!")
        print(f"   - Training time: {training_time}")
        print(f"   - Final loss: {train_result.training_loss:.4f}")
        
        return train_result
    
    def evaluate_model(self):
        """모델 평가"""
        if "test" not in self.datasets:
            print("⚠️  No test dataset available for evaluation")
            return None
            
        print("📊 Evaluating model on test set...")
        eval_result = self.trainer.evaluate(eval_dataset=self.datasets["test"])
        
        print(f"📈 Test Results:")
        for key, value in eval_result.items():
            if isinstance(value, float):
                print(f"   - {key}: {value:.4f}")
            else:
                print(f"   - {key}: {value}")
                
        return eval_result
    
    def save_model(self, path=None):
        """모델 저장"""
        if path is None:
            path = self.config["training"]["output_dir"]
            
        print(f"💾 Saving model to: {path}")
        
        # 디렉토리 생성
        os.makedirs(path, exist_ok=True)
        
        # 모델과 토크나이저 저장
        self.trainer.save_model(path)
        self.tokenizer.save_pretrained(path)
        
        # 설정 파일 저장
        config_path = os.path.join(path, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
            
        print("✅ Model saved successfully")
        
    def test_translation(self, test_sentences=None):
        """번역 테스트"""
        if test_sentences is None:
            test_sentences = [
                "Стороны договора обязуются выполнять свои обязательства.",
                "Данный закон вступает в силу с момента его опубликования.",
                "Российская Федерация является демократическим государством.",
                "Нарушение условий договора влечет за собой ответственность.",
                "Судебное разбирательство проводится в открытом заседании."
            ]
        
        print("🔍 Testing translation quality...")
        print("-" * 60)
        
        for i, text in enumerate(test_sentences, 1):
            translation = self.translate_text(text)
            print(f"{i}. Russian: {text}")
            print(f"   Korean:  {translation}")
            print()
    
    def translate_text(self, text):
        """단일 텍스트 번역"""
        # 토크나이징
        self.tokenizer.src_lang = self.config["model"]["source_lang"]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config["data"]["max_length"],
            truncation=True
        ).to(self.device)
        
        # 번역 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                    self.config["model"]["target_lang"]
                ),
                max_length=self.config["data"]["max_length"],
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # 디코딩
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation

def main():
    """메인 함수"""
    print("🎯 Legal Translation Model Training")
    print("=" * 60)
    
    # 트레이너 초기화
    trainer = LegalTranslationTrainer()
    
    try:
        # 1. 모델 및 토크나이저 로드
        trainer.load_model_and_tokenizer()
        
        # 2. 데이터셋 로드
        trainer.load_datasets()
        
        # 3. 트레이너 설정
        trainer.setup_trainer()
        
        # 4. 훈련 실행
        train_result = trainer.train()
        
        # 5. 모델 평가
        eval_result = trainer.evaluate_model()
        
        # 6. 모델 저장
        trainer.save_model()
        
        # 7. 번역 테스트
        trainer.test_translation()
        
        print("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
