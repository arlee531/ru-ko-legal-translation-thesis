"""
담화표지 5범주 분포 분석 코드
러-한 법률 코퍼스에서 담화표지 범주별 분포 분석

Author: [Ahreum Lee]
Date: 2025
Description: 5범주 담화표지의 코퍼스 내 분포 분석 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict

# 영어 라벨로 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

def build_discourse_marker_patterns():
    """5범주 담화표지 정규식 패턴 구축"""
    
    patterns = {
        'ko': {
            '접속_표지': [r'및', r'또는', r'그리고', r'그러나', r'그런데', r'따라서', r'또한', r'아울러'],
            '조건양보_표지': [r'다만', r'단', r'만약', r'경우', r'하지만', r'비록', r'할지라도'],
            '법률절차_표지': [r'에\s*따라', r'에\s*의하여', r'에\s*관하여', r'에\s*대한', r'에\s*의거하여', r'에\s*근거하여'],
            '규범양상_표지': [r'할\s*수\s*있다', r'해야\s*한다', r'하지\s*아니한다', r'정한다', r'규정한다', r'금지한다'],
            '참조_표지': [r'제\s*\d+\s*조', r'제\s*\d+\s*항', r'해당', r'상기', r'본', r'동', r'같은']
        },
        'ru': {
            '접속_표지': [r'и', r'или', r'но', r'а\s+также', r'однако', r'поэтому', r'кроме\s+того'],
            '조건양보_표지': [r'если', r'за\s+исключением', r'несмотря\s+на', r'хотя'],
            '법률절차_표지': [r'в\s+соответствии\s+с', r'согласно', r'в\s+случае', r'в\s+отношении', r'на\s+основании'],
            '규범양상_표지': [r'может', r'должен', r'вправе', r'обязан', r'запрещается', r'устанавливает'],
            '참조_표지': [r'статья', r'пункт', r'настоящ\w+', r'указанн\w+', r'данн\w+', r'соответствующ\w+']
        }
    }
    
    return patterns

def find_markers_in_sentence(sentence, patterns, language):
    """문장에서 담화표지 검색"""
    
    found_markers = defaultdict(list)
    
    for category, pattern_list in patterns[language].items():
        for pattern in pattern_list:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            if matches:
                found_markers[category].extend(matches)
    
    return found_markers

def analyze_discourse_marker_distribution(csv_file_path):
    """담화표지 분포 분석 메인 함수"""
    
    print("담화표지 분포 분석 시작...")
    
    # 데이터 로딩
    df = pd.read_csv(csv_file_path)
    print(f"데이터 로딩 완료: {len(df):,}개 문장쌍")
    
    # 패턴 구축
    patterns = build_discourse_marker_patterns()
    
    # 결과 저장용 딕셔너리
    results = {
        'ko': defaultdict(lambda: {'sentences_with_marker': 0, 'total_occurrences': 0, 'examples': []}),
        'ru': defaultdict(lambda: {'sentences_with_marker': 0, 'total_occurrences': 0, 'examples': []})
    }
    
    total_sentences = len(df)
    
    # 각 문장 분석
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"진행률: {idx/total_sentences*100:.1f}%")
        
        # 한국어 분석
        ko_sentence = str(row['ko'])
        ko_markers = find_markers_in_sentence(ko_sentence, patterns, 'ko')
        
        for category, markers in ko_markers.items():
            if markers:
                results['ko'][category]['sentences_with_marker'] += 1
                results['ko'][category]['total_occurrences'] += len(markers)
                if len(results['ko'][category]['examples']) < 5:
                    results['ko'][category]['examples'].append((ko_sentence, markers))
        
        # 러시아어 분석
        ru_sentence = str(row['ru'])
        ru_markers = find_markers_in_sentence(ru_sentence, patterns, 'ru')
        
        for category, markers in ru_markers.items():
            if markers:
                results['ru'][category]['sentences_with_marker'] += 1
                results['ru'][category]['total_occurrences'] += len(markers)
                if len(results['ru'][category]['examples']) < 5:
                    results['ru'][category]['examples'].append((ru_sentence, markers))
    
    # 비율 계산
    for lang in ['ko', 'ru']:
        for category in results[lang]:
            results[lang][category]['sentence_ratio'] = \
                results[lang][category]['sentences_with_marker'] / total_sentences * 100
    
    return results, total_sentences

def print_results(results, total_sentences):
    """결과 출력"""
    
    print(f"\n{'='*80}")
    print("5범주 담화표지 분포 분석 결과")
    print(f"{'='*80}")
    print(f"총 분석 문장: {total_sentences:,}개")
    
    for lang_name, lang_code in [('한국어', 'ko'), ('러시아어', 'ru')]:
        print(f"\n🔍 {lang_name} 담화표지 분포:")
        print("-" * 60)
        
        # 결과 정렬 (비율 기준 내림차순)
        sorted_categories = sorted(
            results[lang_code].items(), 
            key=lambda x: x[1]['sentence_ratio'], 
            reverse=True
        )
        
        for category, data in sorted_categories:
            category_name = category.replace('_', '/').upper()
            print(f"\n📂 {category_name}:")
            print(f"  문장 수: {data['sentences_with_marker']:,}개")
            print(f"  문장 비율: {data['sentence_ratio']:.2f}%")
            print(f"  총 출현 횟수: {data['total_occurrences']:,}회")
            
            if data['examples']:
                print(f"  예시:")
                for i, (sentence, markers) in enumerate(data['examples'][:3], 1):
                    short_sentence = sentence[:80] + "..." if len(sentence) > 80 else sentence
                    print(f"    {i}. {short_sentence}")
                    print(f"       → 발견된 표지: {', '.join(markers)}")

def create_comparison_table(results, total_sentences):
    """언어별 비교 표 생성"""
    
    categories = ['접속_표지', '조건양보_표지', '법률절차_표지', '규범양상_표지', '참조_표지']
    category_names = ['접속 표지', '조건/양보 표지', '법률절차 표지', '규범양상 표지', '참조 표지']
    
    comparison_data = []
    
    for i, cat in enumerate(categories):
        ko_data = results['ko'][cat]
        ru_data = results['ru'][cat]
        
        comparison_data.append({
            '범주': category_names[i],
            '한국어_문장수': ko_data['sentences_with_marker'],
            '한국어_비율': f"{ko_data['sentence_ratio']:.2f}%",
            '러시아어_문장수': ru_data['sentences_with_marker'],
            '러시아어_비율': f"{ru_data['sentence_ratio']:.2f}%",
            '차이': f"{ko_data['sentence_ratio'] - ru_data['sentence_ratio']:+.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print(f"\n{'='*100}")
    print("담화표지 범주별 언어 간 비교표")
    print(f"{'='*100}")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

def create_visualization(results, total_sentences):
    """분포 시각화 (영어 라벨)"""
    
    # 데이터 준비
    categories = ['접속_표지', '조건양보_표지', '법률절차_표지', '규범양상_표지', '참조_표지']
    category_names = ['Conjunctive', 'Conditional/Concessive', 'Legal Procedural', 'Deontic Modal', 'Reference']
    
    ko_ratios = [results['ko'][cat]['sentence_ratio'] for cat in categories]
    ru_ratios = [results['ru'][cat]['sentence_ratio'] for cat in categories]
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    x = np.arange(len(categories))
    width = 0.35
    
    # 막대 그래프
    bars1 = ax1.bar(x - width/2, ko_ratios, width, label='Korean', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, ru_ratios, width, label='Russian', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Discourse Marker Categories')
    ax1.set_ylabel('Sentence Ratio (%)')
    ax1.set_title('Distribution Comparison of Discourse Marker Categories')
    ax1.set_xticks(x)
    ax1.set_xticklabels(category_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 파이 차트 (한국어)
    ko_values = [results['ko'][cat]['sentences_with_marker'] for cat in categories]
    ax2.pie(ko_values, labels=category_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Korean Discourse Marker Distribution')
    
    plt.tight_layout()
    plt.show()

def main():
    """메인 실행 함수"""
    
    # CSV 파일 경로 설정
    csv_file_path = "ru_ko_legal_corpus_10000.csv"
    
    # 분석 실행
    results, total_sentences = analyze_discourse_marker_distribution(csv_file_path)
    
    # 결과 출력
    print_results(results, total_sentences)
    
    # 비교표 생성
    comparison_df = create_comparison_table(results, total_sentences)
    
    # 시각화
    create_visualization(results, total_sentences)
    
    print("\n✅ 담화표지 분포 분석 완료!")
    
    return results, comparison_df

# 실행
if __name__ == "__main__":
    results, comparison_df = main()
