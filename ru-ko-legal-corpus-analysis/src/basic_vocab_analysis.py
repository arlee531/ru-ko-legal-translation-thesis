"""
Basic Vocabulary Statistics Analysis
러-한 법률 병렬 코퍼스 기본 어휘 통계 분석

Author: [Your Name]
Date: 2025
Description: TTR, 고빈도 어휘, 어휘 다양성 등 기본 통계 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# 한글 폰트 설정 (matplotlib 한글 깨짐 방지)
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

def load_corpus(file_path):
    """
    병렬 코퍼스 CSV 파일 로딩
    
    Args:
        file_path (str): CSV 파일 경로
        
    Returns:
        pd.DataFrame: 로딩된 데이터프레임 (id, ru, ko 컬럼)
    """
    df = pd.read_csv(file_path)
    print(f"데이터 로딩 완료: {len(df):,}개 문장쌍")
    return df

def tokenize_text(text, language='ko'):
    """
    언어별 토큰화
    
    Args:
        text (str): 입력 텍스트
        language (str): 'ko' 또는 'ru'
        
    Returns:
        list: 토큰 리스트
    """
    if language == 'ko':
        # 한국어: 공백 기준 분리
        tokens = text.split()
    elif language == 'ru':
        # 러시아어: 단어 경계 기준 분리
        tokens = re.findall(r'\b\w+\b', text)
    else:
        raise ValueError("language는 'ko' 또는 'ru'여야 합니다")
    
    return tokens

def calculate_basic_statistics(tokens):
    """
    기본 어휘 통계 계산
    
    Args:
        tokens (list): 토큰 리스트
        
    Returns:
        dict: 기본 통계 정보
    """
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    type_token_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
    token_lengths = [len(token) for token in tokens]
    
    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'type_token_ratio': type_token_ratio,
        'avg_token_length': np.mean(token_lengths) if token_lengths else 0,
        'token_lengths': token_lengths
    }

def analyze_frequency_distribution(tokens, top_n=100):
    """
    빈도 분포 분석
    
    Args:
        tokens (list): 토큰 리스트
        top_n (int): 상위 N개 어휘
        
    Returns:
        dict: 빈도 분석 결과
    """
    freq_counter = Counter(tokens)
    
    # 고빈도 어휘
    top_words = freq_counter.most_common(top_n)
    
    # 저빈도 어휘 (hapax legomena - 1회만 등장)
    hapax_count = sum(1 for count in freq_counter.values() if count == 1)
    hapax_ratio = hapax_count / len(freq_counter) if len(freq_counter) > 0 else 0
    
    return {
        'frequency_dist': freq_counter,
        'top_words': top_words,
        'hapax_count': hapax_count,
        'hapax_ratio': hapax_ratio,
        'vocabulary_size': len(freq_counter)
    }

def print_basic_summary(ko_stats, ru_stats, ko_freq, ru_freq, sentence_count):
    """
    기본 통계 요약 출력
    """
    print("=" * 60)
    print("러-한 법률 코퍼스 기본 어휘 통계 분석 결과")
    print("=" * 60)
    
    print(f"\n📄 코퍼스 규모:")
    print(f"  총 문장쌍: {sentence_count:,}개")
    
    print(f"\n🔤 어휘 통계:")
    print(f"  한국어:")
    print(f"    총 토큰: {ko_stats['total_tokens']:,}개")
    print(f"    고유 어휘: {ko_stats['unique_tokens']:,}개")
    print(f"    TTR: {ko_stats['type_token_ratio']:.4f}")
    print(f"    평균 토큰 길이: {ko_stats['avg_token_length']:.2f}자")
    
    print(f"  러시아어:")
    print(f"    총 토큰: {ru_stats['total_tokens']:,}개")
    print(f"    고유 어휘: {ru_stats['unique_tokens']:,}개")
    print(f"    TTR: {ru_stats['type_token_ratio']:.4f}")
    print(f"    평균 토큰 길이: {ru_stats['avg_token_length']:.2f}자")
    
    print(f"\n🔍 저빈도 어휘 (hapax legomena):")
    print(f"  한국어: {ko_freq['hapax_count']:,}개 ({ko_freq['hapax_ratio']:.2%})")
    print(f"  러시아어: {ru_freq['hapax_count']:,}개 ({ru_freq['hapax_ratio']:.2%})")

def print_top_words(freq_analysis, language, top_n=100):
    """
    고빈도 어휘 출력
    
    Args:
        freq_analysis (dict): 빈도 분석 결과
        language (str): 언어명 (출력용)
        top_n (int): 출력할 상위 어휘 수
    """
    print(f"\n🔝 {language} 고빈도 어휘 Top {top_n}:")
    print("-" * 50)
    
    for i, (word, count) in enumerate(freq_analysis['top_words'][:top_n], 1):
        print(f"{i:3d}. {word:<20} ({count:,}회)")

def plot_comparison_charts(ko_stats, ru_stats):
    """
    언어별 비교 차트 생성
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    languages = ['Korean', 'Russian']
    
    # 토큰 수 비교
    token_counts = [ko_stats['total_tokens'], ru_stats['total_tokens']]
    axes[0].bar(languages, token_counts, color=['skyblue', 'lightcoral'])
    axes[0].set_title('Total Tokens')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(token_counts):
        axes[0].text(i, v + max(token_counts)*0.01, f'{v:,}', ha='center')
    
    # 고유 어휘 수 비교
    unique_counts = [ko_stats['unique_tokens'], ru_stats['unique_tokens']]
    axes[1].bar(languages, unique_counts, color=['skyblue', 'lightcoral'])
    axes[1].set_title('Unique Vocabulary')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(unique_counts):
        axes[1].text(i, v + max(unique_counts)*0.01, f'{v:,}', ha='center')
    
    # TTR 비교
    ttr_values = [ko_stats['type_token_ratio'], ru_stats['type_token_ratio']]
    axes[2].bar(languages, ttr_values, color=['skyblue', 'lightcoral'])
    axes[2].set_title('Type-Token Ratio')
    axes[2].set_ylabel('Ratio')
    for i, v in enumerate(ttr_values):
        axes[2].text(i, v + max(ttr_values)*0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_token_length_distribution(ko_stats, ru_stats):
    """
    토큰 길이 분포 히스토그램
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(ko_stats['token_lengths'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Korean Token Length Distribution')
    plt.xlabel('Token Length (characters)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(ru_stats['token_lengths'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Russian Token Length Distribution')
    plt.xlabel('Token Length (characters)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_zipf_distribution(freq_analysis, language):
    """
    Zipf 분포 시각화
    """
    frequencies = list(freq_analysis['frequency_dist'].values())
    frequencies.sort(reverse=True)
    ranks = range(1, len(frequencies) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, 'o-', alpha=0.7, markersize=3)
    plt.title(f'{language} Word Frequency Distribution (Zipf\'s Law)')
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.grid(True, alpha=0.3)
    plt.show()

def main_analysis(csv_file_path):
    """
    메인 분석 함수
    
    Args:
        csv_file_path (str): CSV 파일 경로
        
    Returns:
        tuple: (df, ko_stats, ru_stats, ko_freq, ru_freq)
    """
    # 1. 데이터 로딩
    df = load_corpus(csv_file_path)
    
    # 2. 텍스트 합치기
    korean_text = ' '.join(df['ko'].astype(str))
    russian_text = ' '.join(df['ru'].astype(str))
    
    # 3. 토큰화
    print("\n토큰화 진행 중...")
    ko_tokens = tokenize_text(korean_text, 'ko')
    ru_tokens = tokenize_text(russian_text, 'ru')
    
    # 4. 기본 통계 계산
    print("기본 통계 계산 중...")
    ko_stats = calculate_basic_statistics(ko_tokens)
    ru_stats = calculate_basic_statistics(ru_tokens)
    
    # 5. 빈도 분석
    print("빈도 분석 중...")
    ko_freq = analyze_frequency_distribution(ko_tokens)
    ru_freq = analyze_frequency_distribution(ru_tokens)
    
    # 6. 결과 출력
    print_basic_summary(ko_stats, ru_stats, ko_freq, ru_freq, len(df))
    print_top_words(ko_freq, "한국어", 100)
    print_top_words(ru_freq, "러시아어", 100)
    
    # 7. 시각화
    print("\n시각화 생성 중...")
    plot_comparison_charts(ko_stats, ru_stats)
    plot_token_length_distribution(ko_stats, ru_stats)
    plot_zipf_distribution(ko_freq, "Korean")
    plot_zipf_distribution(ru_freq, "Russian")
    
    return df, ko_stats, ru_stats, ko_freq, ru_freq

# 사용 예시
if __name__ == "__main__":
    # 파일 경로 설정
    csv_file_path = "ru_ko_legal_corpus_10000.csv"
    
    # 분석 실행
    df, ko_stats, ru_stats, ko_freq, ru_freq = main_analysis(csv_file_path)
    
    print("\n✅ 기본 어휘 통계 분석 완료!")
