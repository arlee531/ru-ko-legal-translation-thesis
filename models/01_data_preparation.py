"""
데이터 준비 (파일 업로드 및 전처리)
Data Preparation with File Upload (Loading and Preprocessing)
- 파일 업로드 기능 추가
- CSV 파일 로드
- 컬럼명 표준화
- 데이터 품질 검증
- Hugging Face Dataset 형식 변환

Author: [Ahreum Lee]
Date: 2025-09-11
Modified: 2025-09-17 (파일 업로드 기능 추가)
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from typing import Dict, Tuple, Any, List
import logging
import os

# 코랩 환경 확인 및 업로드 기능 추가
try:
    from google.colab import files
    IN_COLAB = True
    print("Google Colab 환경이 감지되었습니다.")
except ImportError:
    IN_COLAB = False
    print("로컬 환경에서 실행됩니다.")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_files():
    """파일 업로드 기능"""
    if IN_COLAB:
        print("필요한 CSV 파일들을 업로드해주세요:")
        print("- train_v2.csv (훈련 데이터)")
        print("- dev_v2.csv (검증 데이터)")  
        print("- test_v2.csv (테스트 데이터)")
        print("\n파일 선택 버튼을 클릭하여 업로드를 시작하세요.")
        
        uploaded = files.upload()
        
        print("\n업로드 완료된 파일들:")
        for filename in uploaded.keys():
            print(f"- {filename}")
        
        return list(uploaded.keys())
    else:
        print("로컬 환경에서는 파일이 현재 디렉토리에 있어야 합니다.")
        current_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        print(f"현재 디렉토리의 CSV 파일들: {current_files}")
        return current_files

def check_uploaded_files():
    """업로드된 파일 확인 및 자동 매핑"""
    print("\n업로드된 파일 확인 중...")
    
    # 현재 디렉토리의 모든 파일 확인
    all_files = os.listdir('.')
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    print(f"발견된 CSV 파일들: {csv_files}")
    
    # 파일명 자동 매핑
    file_mapping = {}
    
    for file in csv_files:
        if 'train' in file.lower():
            file_mapping['train'] = file
        elif 'dev' in file.lower() or 'val' in file.lower():
            file_mapping['dev'] = file
        elif 'test' in file.lower():
            file_mapping['test'] = file
    
    print(f"\n자동 매핑 결과:")
    for split, filename in file_mapping.items():
        print(f"- {split}: {filename}")
    
    # 누락된 파일 확인
    required_splits = ['train', 'dev', 'test']
    missing_files = [split for split in required_splits if split not in file_mapping]
    
    if missing_files:
        print(f"\n누락된 파일: {missing_files}")
        print("파일명에 'train', 'dev', 'test'가 포함되도록 해주세요.")
        
        # 수동 매핑 옵션
        if len(csv_files) >= 3:
            print(f"\n수동 매핑을 시도합니다:")
            print(f"첫 번째 파일을 train으로 사용: {csv_files[0]}")
            print(f"두 번째 파일을 dev로 사용: {csv_files[1]}")
            print(f"세 번째 파일을 test로 사용: {csv_files[2]}")
            
            file_mapping = {
                'train': csv_files[0],
                'dev': csv_files[1], 
                'test': csv_files[2]
            }
    
    return file_mapping

class LegalDataLoader:
    """법률 도메인 데이터 로딩 및 전처리 클래스"""
    
    def __init__(self):
        self.train_df = None
        self.dev_df = None
        self.test_df = None
        self.dataset_dict = None
        
        # 데이터 통계
        self.data_stats = {}
    
    def load_csv_files(self, file_mapping: Dict[str, str] = None) -> bool:
        """CSV 파일들을 로드합니다."""
        try:
            logger.info("CSV 파일 로딩 시작")
            
            # 파일 매핑이 없으면 기본값 사용
            if file_mapping is None:
                file_mapping = {
                    "train": "train_v2.csv",
                    "dev": "dev_v2.csv", 
                    "test": "test_v2.csv"
                }
            
            # 파일 존재 확인
            for split, path in file_mapping.items():
                if not os.path.exists(path):
                    logger.error(f"{split} 파일을 찾을 수 없습니다: {path}")
                    return False
            
            # 파일 로딩
            logger.info(f"훈련 데이터 로딩: {file_mapping['train']}")
            self.train_df = pd.read_csv(file_mapping['train'])
            
            logger.info(f"검증 데이터 로딩: {file_mapping['dev']}")
            self.dev_df = pd.read_csv(file_mapping['dev'])
            
            logger.info(f"테스트 데이터 로딩: {file_mapping['test']}")
            self.test_df = pd.read_csv(file_mapping['test'])
            
            logger.info("모든 CSV 파일 로딩 완료")
            return True
            
        except Exception as e:
            logger.error(f"CSV 파일 로딩 실패: {e}")
            return False
    
    def analyze_raw_data(self):
        """원시 데이터 구조 분석"""
        logger.info("원시 데이터 분석 시작")
        
        datasets = {"train": self.train_df, "dev": self.dev_df, "test": self.test_df}
        
        for split, df in datasets.items():
            logger.info(f"\n{split.upper()} 데이터 분석:")
            logger.info(f"  행 수: {len(df):,}")
            logger.info(f"  열 수: {len(df.columns)}")
            logger.info(f"  컬럼명: {list(df.columns)}")
            
            # 결측값 확인
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                logger.warning(f"  결측값 발견:")
                for col, count in missing_values.items():
                    if count > 0:
                        logger.warning(f"    {col}: {count}개")
            else:
                logger.info("  결측값: 없음")
            
            # 샘플 데이터 확인
            if len(df) > 0:
                logger.info(f"  첫 번째 샘플:")
                for col in df.columns:
                    sample_text = str(df.iloc[0][col])
                    if len(sample_text) > 50:
                        sample_text = sample_text[:50] + "..."
                    logger.info(f"    {col}: {sample_text}")
    
    def standardize_columns(self) -> bool:
        """컬럼명을 Hugging Face 표준으로 변환"""
        try:
            logger.info("컬럼명 표준화 시작")
            
            # 각 데이터프레임에서 필요한 컬럼만 추출하고 이름 변경
            datasets = {"train": self.train_df, "dev": self.dev_df, "test": self.test_df}
            
            for split, df in datasets.items():
                # 컬럼 확인 및 자동 매핑
                columns = df.columns.tolist()
                logger.info(f"{split} 데이터 컬럼: {columns}")
                
                # 러시아어/한국어 컬럼 자동 감지
                ru_col = None
                ko_col = None
                
                for col in columns:
                    col_lower = col.lower()
                    if 'ru' in col_lower or 'russian' in col_lower or 'source' in col_lower:
                        ru_col = col
                    elif 'ko' in col_lower or 'korean' in col_lower or 'target' in col_lower:
                        ko_col = col
                
                if ru_col is None or ko_col is None:
                    # 컬럼명 자동 감지 실패시 순서로 추정
                    if len(columns) >= 2:
                        ru_col = columns[0] if ru_col is None else ru_col
                        ko_col = columns[1] if ko_col is None else ko_col
                        logger.warning(f"{split}: 컬럼 자동감지 실패, 순서로 추정 - ru: {ru_col}, ko: {ko_col}")
                    else:
                        logger.error(f"{split} 데이터에 충분한 컬럼이 없습니다.")
                        return False
                
                # 새로운 데이터프레임 생성
                clean_df = df[[ru_col, ko_col]].copy()
                clean_df.columns = ['source_text', 'target_text']
                
                # 원본 데이터프레임 교체
                if split == "train":
                    self.train_df = clean_df
                elif split == "dev":
                    self.dev_df = clean_df
                elif split == "test":
                    self.test_df = clean_df
                
                logger.info(f"{split} 데이터 컬럼명 변경: {ru_col}→source_text, {ko_col}→target_text")
            
            logger.info("컬럼명 표준화 완료")
            return True
            
        except Exception as e:
            logger.error(f"컬럼명 표준화 실패: {e}")
            return False
    
    def clean_data(self, min_length: int = 5) -> bool:
        """데이터 품질 정제"""
        try:
            logger.info("데이터 정제 시작")
            
            datasets = {"train": self.train_df, "dev": self.dev_df, "test": self.test_df}
            cleaned_datasets = {}
            
            for split, df in datasets.items():
                initial_count = len(df)
                
                # 결측값 제거
                clean_df = df.dropna()
                after_na_removal = len(clean_df)
                
                # 짧은 문장 제거
                clean_df = clean_df[
                    (clean_df['source_text'].str.len() > min_length) & 
                    (clean_df['target_text'].str.len() > min_length)
                ]
                final_count = len(clean_df)
                
                cleaned_datasets[split] = clean_df
                
                logger.info(f"{split} 데이터 정제 결과:")
                logger.info(f"  초기: {initial_count:,}개")
                logger.info(f"  결측값 제거 후: {after_na_removal:,}개")
                logger.info(f"  짧은 문장 제거 후: {final_count:,}개")
                logger.info(f"  제거된 데이터: {initial_count - final_count:,}개")
            
            # 정제된 데이터프레임 할당
            self.train_df = cleaned_datasets["train"]
            self.dev_df = cleaned_datasets["dev"]
            self.test_df = cleaned_datasets["test"]
            
            logger.info("데이터 정제 완료")
            return True
            
        except Exception as e:
            logger.error(f"데이터 정제 실패: {e}")
            return False
    
    def analyze_text_statistics(self):
        """텍스트 길이 및 통계 분석"""
        logger.info("텍스트 통계 분석 시작")
        
        datasets = {"train": self.train_df, "dev": self.dev_df, "test": self.test_df}
        self.data_stats = {}
        
        for split, df in datasets.items():
            # 러시아어 텍스트 길이
            ru_lengths = df['source_text'].str.len()
            # 한국어 텍스트 길이  
            ko_lengths = df['target_text'].str.len()
            
            stats = {
                "count": len(df),
                "russian": {
                    "min": ru_lengths.min(),
                    "max": ru_lengths.max(),
                    "mean": ru_lengths.mean(),
                    "median": ru_lengths.median(),
                    "std": ru_lengths.std()
                },
                "korean": {
                    "min": ko_lengths.min(),
                    "max": ko_lengths.max(),
                    "mean": ko_lengths.mean(),
                    "median": ko_lengths.median(),
                    "std": ko_lengths.std()
                }
            }
            
            self.data_stats[split] = stats
            
            logger.info(f"\n{split.upper()} 텍스트 통계:")
            logger.info(f"  문장 쌍 수: {stats['count']:,}")
            logger.info(f"  러시아어 길이 - 평균: {stats['russian']['mean']:.1f}, "
                       f"최소: {stats['russian']['min']}, 최대: {stats['russian']['max']}")
            logger.info(f"  한국어 길이 - 평균: {stats['korean']['mean']:.1f}, "
                       f"최소: {stats['korean']['min']}, 최대: {stats['korean']['max']}")
    
    def convert_to_datasets(self) -> bool:
        """Hugging Face Datasets 형식으로 변환"""
        try:
            logger.info("Hugging Face Dataset 형식 변환 시작")
            
            # 인덱스 리셋
            train_clean = self.train_df.reset_index(drop=True)
            dev_clean = self.dev_df.reset_index(drop=True)
            test_clean = self.test_df.reset_index(drop=True)
            
            # DatasetDict 생성
            self.dataset_dict = DatasetDict({
                'train': Dataset.from_pandas(train_clean),
                'validation': Dataset.from_pandas(dev_clean),
                'test': Dataset.from_pandas(test_clean)
            })
            
            logger.info("Dataset 변환 완료:")
            for split, dataset in self.dataset_dict.items():
                logger.info(f"  {split}: {len(dataset):,}개")
                logger.info(f"    컬럼: {dataset.column_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Dataset 변환 실패: {e}")
            return False
    
    def save_processed_data(self, output_dir: str = "./processed_data"):
        """전처리된 데이터 저장"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # CSV 형태로 저장
            self.train_df.to_csv(f"{output_dir}/processed_train.csv", index=False)
            self.dev_df.to_csv(f"{output_dir}/processed_dev.csv", index=False)
            self.test_df.to_csv(f"{output_dir}/processed_test.csv", index=False)
            
            # 통계 정보 저장
            import json
            with open(f"{output_dir}/data_statistics.json", 'w', encoding='utf-8') as f:
                json.dump(self.data_stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"전처리된 데이터 저장 완료: {output_dir}")
            
        except Exception as e:
            logger.error(f"데이터 저장 실패: {e}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """데이터셋 정보 반환"""
        if self.dataset_dict is None:
            return {"error": "데이터셋이 로딩되지 않았습니다."}
        
        return {
            "train_size": len(self.dataset_dict['train']),
            "validation_size": len(self.dataset_dict['validation']),
            "test_size": len(self.dataset_dict['test']),
            "total_size": len(self.dataset_dict['train']) + len(self.dataset_dict['validation']) + len(self.dataset_dict['test']),
            "columns": self.dataset_dict['train'].column_names,
            "statistics": self.data_stats
        }

def main():
    """메인 실행 함수"""
    print("러시아어-한국어 법률 번역 데이터 준비")
    print("="*80)
    
    # 1. 파일 업로드 (코랩인 경우)
    if IN_COLAB:
        uploaded_files = upload_files()
        if not uploaded_files:
            print("파일 업로드가 필요합니다.")
            return None
    
    # 2. 업로드된 파일 확인 및 매핑
    file_mapping = check_uploaded_files()
    if len(file_mapping) < 3:
        print("필요한 모든 파일이 준비되지 않았습니다.")
        return None
    
    # 3. 데이터 로더 초기화
    loader = LegalDataLoader()
    
    # 4. CSV 파일 로딩
    if not loader.load_csv_files(file_mapping):
        print("CSV 파일 로딩 실패")
        return None
    
    # 5. 원시 데이터 분석
    loader.analyze_raw_data()
    
    # 6. 컬럼명 표준화
    if not loader.standardize_columns():
        print("컬럼명 표준화 실패")
        return None
    
    # 7. 데이터 정제
    if not loader.clean_data():
        print("데이터 정제 실패")
        return None
    
    # 8. 텍스트 통계 분석
    loader.analyze_text_statistics()
    
    # 9. Hugging Face Dataset 변환
    if not loader.convert_to_datasets():
        print("Dataset 변환 실패")
        return None
    
    # 10. 전처리된 데이터 저장
    loader.save_processed_data()
    
    # 11. 최종 정보 출력
    info = loader.get_dataset_info()
    
    print("\n" + "="*80)
    print("데이터 준비 완료")
    print("="*80)
    print(f"훈련 데이터: {info['train_size']:,}개")
    print(f"검증 데이터: {info['validation_size']:,}개")
    print(f"테스트 데이터: {info['test_size']:,}개")
    print(f"총합: {info['total_size']:,}개")
    print(f"컬럼: {info['columns']}")
    print("\n다음 단계: 02_tokenization.py")
    
    return loader

if __name__ == "__main__":
    # 실행
    data_loader = main()
    
    if data_loader:
        print("\n데이터 준비가 성공적으로 완료되었습니다.")
        print("이제 토큰화 단계로 진행할 수 있습니다.")
    else:
        print("\n데이터 준비 실패. 파일 경로와 형식을 확인해주세요.")
