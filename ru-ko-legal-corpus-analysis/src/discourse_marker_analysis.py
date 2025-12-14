"""
담화표지 5범주 분포 분석 코드
러-한 법률 코퍼스에서 담화표지 범주별 분포 분석

import pandas as pd
import re
from collections import defaultdict

# 1. 파일 업로드 및 데이터 로딩 (코랩에서 실행)
print("파일을 업로드하세요...")
from google.colab import files
uploaded = files.upload()

# 업로드된 파일명 확인
filename = list(uploaded.keys())[0]
print(f"업로드된 파일: {filename}")

# CSV 파일 로딩
df = pd.read_csv(filename)
print(f"데이터 로딩 완료: {len(df)}개 문장")
print(f"컬럼: {list(df.columns)}")
print(f"첫 3개 샘플:")
print(df.head(3))

# 2. 표 17 기준 담화 표지 사전 정의
discourse_markers = {
    "접속 표지": {
        "ru": ["и", "или", "а также", "поэтому", "кроме того", "в том числе", "при этом"],
        "ko": ["및", "또는", "그리고", "따라서", "또한", "아울러", "혹은"]
    },
    "조건/대립 표지": {
        "ru": ["если", "однако", "за исключением", "несмотря на", "хотя", "но", "за исключением случаев", "в случае", "если", "при условии"],
        "ko": ["다만", "단", "만약", "경우", "그러나", "그런데", "하지만", "비록", "할지라도", "이 경우", "이때", "하는/한/할 경우"]
    },
    "근거/지시 표지": {
        "ru": ["в соответствии с", "согласно", "в случае/в случаях", "в отношении", "на основании"],
        "ko": ["에 따라", "에 의하여", "에 관하여", "에 대한", "에 의거하여", "에 근거하여"]
    },
    "규범 양상 표지": {
        "ru": ["может", "должен", "вправе", "обязан", "запрещается", "устанавливает", "подлежит/подлежат"],
        "ko": ["할 수 있다/수 없다", "해야 한다", "하지 아니한다", "정한다", "규정한다", "금지한다", "얻는다"]
    },
    "참조 표지": {
        "ru": ["статья/статьи", "пункт", "подпункт", "настоящий", "указанный", "данный", "соответствующий", "настоящий", "вышеуказанный/нижеуказанный"],
        "ko": ["제 N 장", "제 N 조", "제 N 항", "제 N 목", "제 N 호", "해당", "상기", "본 항", "본 조", "본 법", "동", "같은", "이 법", "이 조", "이 항"]
    }
}

# 3. 정규식 패턴 정의 (표 17의 실제 표현 기준)
patterns = {
    "ru": {
        # 접속 표지
        "и": r'\bи\b',
        "или": r'\bили\b', 
        "а также": r'а\s+также',
        "поэтому": r'\bпоэтому\b',
        "кроме того": r'кроме\s+того',
        "в том числе": r'в\s+том\s+числе',
        "при этом": r'при\s+этом',
        
        # 조건/대립 표지  
        "если": r'\bесли\b',
        "однако": r'\bоднако\b',
        "за исключением": r'за\s+исключением',
        "несмотря на": r'несмотря\s+на',
        "хотя": r'\bхотя\b',
        "но": r'\bно\b',
        "за исключением случаев": r'за\s+исключением\s+случаев',
        "в случае": r'в\s+случае',
        "при условии": r'при\s+условии',
        
        # 근거/지시 표지
        "в соответствии с": r'в\s+соответствии\s+с',
        "согласно": r'\bсогласно\b',
        "в случаях": r'в\s+случаях', 
        "в отношении": r'в\s+отношении',
        "на основании": r'на\s+основании',
        
        # 규범 양상 표지
        "может": r'\bможет\b',
        "должен": r'должн[а-я]+',
        "вправе": r'\bвправе\b',
        "обязан": r'обязан[а-я]*',
        "запрещается": r'\bзапрещается\b',
        "устанавливает": r'\bустанавливает\b',
        "подлежит": r'\bподлежит\b',
        "подлежат": r'\bподлежат\b',
        
        # 참조 표지
        "статья": r'\bстатья\b',
        "статьи": r'\bстатьи\b',
        "пункт": r'пункт[а-я]*',
        "подпункт": r'подпункт[а-я]*',
        "настоящий": r'настоящ[а-я]+',
        "указанный": r'указанн[а-я]+',
        "данный": r'данн[а-я]+',
        "соответствующий": r'соответствующ[а-я]+',
        "вышеуказанный": r'вышеуказанн[а-я]+',
        "нижеуказанный": r'нижеуказанн[а-я]+'
    },
    
    "ko": {
        # 접속 표지
        "및": r'\b및\b',
        "또는": r'\b또는\b',
        "그리고": r'\b그리고\b', 
        "따라서": r'\b따라서\b',
        "또한": r'\b또한\b',
        "아울러": r'\b아울러\b',
        "혹은": r'\b혹은\b',
        
        # 조건/대립 표지
        "다만": r'\b다만\b',
        "단": r'\b단[\s,]',
        "만약": r'\b만약\b',
        "경우": r'\b경우\b',
        "그러나": r'\b그러나나\b',
        "그런데": r'\b그런데\b', 
        "하지만": r'\b하지만\b',
        "비록": r'\b비록\b',
        "할지라도": r'\b할지라도\b',
        "이 경우": r'이\s*경우',
        "이때": r'\b이때\b',
        "하는 경우": r'하는\s*경우',
        "한 경우": r'한\s*경우',
        "할 경우": r'할\s*경우',
        
        # 근거/지시 표지
        "에 따라": r'에\s*따라',
        "에 의하여": r'에\s*의하여',
        "에 관하여": r'에\s*관하여',
        "에 대한": r'에\s*대한',
        "에 의거하여": r'에\s*의거하여',
        "에 근거하여": r'에\s*근거하여',
        
        # 규범 양상 표지
        "할 수 있다": r'할\s*수\s*있다',
        "수 없다": r'수\s*없다',
        "해야 한다": r'해야\s*한다',
        "하지 아니한다": r'하지\s*아니한다',
        "정한다": r'정한다',
        "규정한다": r'규정한다',
        "금지한다": r'금지한다',
        "얻는다": r'얻는다',
        
        # 참조 표지
        "제": r'제\s*\d+',
        "조": r'\d+\s*조',
        "항": r'\d+\s*항',
        "목": r'\d+\s*목',
        "호": r'\d+\s*호',
        "해당": r'\b해당\b',
        "상기": r'\b상기\b',
        "본 항": r'본\s*항',
        "본 조": r'본\s*조',
        "본 법": r'본\s*법',
        "동": r'동\s*[조항목호]',
        "같은": r'\b같은\b',
        "이 법": r'이\s*법',
        "이 조": r'이\s*조',
        "이 항": r'이\s*항'
    }
}

# 4. 전체 텍스트 결합
ru_text = ' '.join(df['ru'].fillna('').astype(str))
ko_text = ' '.join(df['ko'].fillna('').astype(str))

print(f"러시아어 텍스트 길이: {len(ru_text):,}자")
print(f"한국어 텍스트 길이: {len(ko_text):,}자")

# 5. 빈도 분석 함수
def analyze_markers(text, patterns_dict, language):
    """텍스트에서 담화 표지 빈도 분석"""
    results = {}
    
    for marker, pattern in patterns_dict[language].items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        results[marker] = len(matches)
    
    return results

# 6. 분석 실행
print("\n" + "="*80)
print("담화 표지 빈도 분석 시작...")
print("="*80)

ru_frequency = analyze_markers(ru_text, patterns, 'ru')
ko_frequency = analyze_markers(ko_text, patterns, 'ko')

# 7. 범주별 결과 정리 및 출력
category_mapping = {
    # 접속 표지
    "и": "접속 표지", "или": "접속 표지", "а также": "접속 표지",
    "поэтому": "접속 표지", "кроме того": "접속 표지", "в том числе": "접속 표지", "при этом": "접속 표지",
    "및": "접속 표지", "또는": "접속 표지", "그리고": "접속 표지",
    "따라서": "접속 표지", "또한": "접속 표지", "아울러": "접속 표지", "혹은": "접속 표지",
    
    # 조건/대립 표지
    "если": "조건/대립 표지", "однако": "조건/대립 표지", "за исключением": "조건/대립 표지", "несмотря на": "조건/대립 표지", 
    "хотя": "조건/대립 표지", "но": "조건/대립 표지", "за исключением случаев": "조건/대립 표지", "в случае": "조건/대립 표지", "при условии": "조건/대립 표지",
    "다만": "조건/대립 표지", "단": "조건/대립 표지", "만약": "조건/대립 표지", "경우": "조건/대립 표지", "그러나": "조건/대립 표지", "그런데": "조건/대립 표지",
    "하지만": "조건/대립 표지", "비록": "조건/대립 표지", "할지라도": "조건/대립 표지", "이 경우": "조건/대립 표지", 
    "이때": "조건/대립 표지", "하는 경우": "조건/대립 표지", "한 경우": "조건/대립 표지", "할 경우": "조건/대립 표지",
    
    # 근거/지시 표지
    "в соответствии с": "근거/지시 표지", "согласно": "근거/지시 표지", "в случаях": "근거/지시 표지", 
    "в отношении": "근거/지시 표지", "на основании": "근거/지시 표지",
    "에 따라": "근거/지시 표지", "에 의하여": "근거/지시 표지", "에 관하여": "근거/지시 표지", 
    "에 대한": "근거/지시 표지", "에 의거하여": "근거/지시 표지", "에 근거하여": "근거/지시 표지",
    
    # 규범 양상 표지
    "может": "규범 양상 표지", "должен": "규범 양상 표지", "вправе": "규범 양상 표지", "обязан": "규범 양상 표지",
    "запрещается": "규범 양상 표지", "устанавливает": "규범 양상 표지", "подлежит": "규범 양상 표지", "подлежат": "규범 양상 표지",
    "할 수 있다": "규범 양상 표지", "수 없다": "규범 양상 표지", "해야 한다": "규범 양상 표지", 
    "하지 아니한다": "규범 양상 표지", "정한다": "규범 양상 표지", "규정한다": "규범 양상 표지", 
    "금지한다": "규범 양상 표지", "얻는다": "규범 양상 표지",
    
    # 참조 표지
    "статья": "참조 표지", "статьи": "참조 표지", "пункт": "참조 표지", "подпункт": "참조 표지",
    "настоящий": "참조 표지", "указанный": "참조 표지", "данный": "참조 표지", "соответствующий": "참조 표지",
    "вышеуказанный": "참조 표지", "нижеуказанный": "참조 표지",
    "제": "참조 표지", "조": "참조 표지", "항": "참조 표지", "목": "참조 표지", "호": "참조 표지",
    "해당": "참조 표지", "상기": "참조 표지", "본 항": "참조 표지", "본 조": "참조 표지", "본 법": "참조 표지",
    "동": "참조 표지", "같은": "참조 표지", "이 법": "참조 표지", "이 조": "참조 표지", "이 항": "참조 표지"
}

# 8. 범주별 통계 계산
category_stats = defaultdict(lambda: {"ru": defaultdict(int), "ko": defaultdict(int)})

for marker, count in ru_frequency.items():
    if marker in category_mapping:
        category = category_mapping[marker]
        category_stats[category]["ru"][marker] = count

for marker, count in ko_frequency.items():
    if marker in category_mapping:
        category = category_mapping[marker]
        category_stats[category]["ko"][marker] = count

# 9. 결과 출력
print("\n📊 담화 표지 범주별 분석 결과")
print("="*80)

for category in ["접속 표지", "조건/대립 표지", "근거/지시 표지", "규범 양상 표지", "참조 표지"]:
    print(f"\n🔹 {category}")
    print("-" * 60)
    
    # 러시아어
    ru_total = sum(category_stats[category]["ru"].values())
    print(f"러시아어 (총 {ru_total}회):")
    if ru_total > 0:
        ru_sorted = sorted(category_stats[category]["ru"].items(), key=lambda x: x[1], reverse=True)
        for marker, count in ru_sorted:
            if count > 0:
                percentage = (count / ru_total) * 100
                print(f"  {marker}: {count}회 ({percentage:.1f}%)")
    else:
        print("  (해당 표현 없음)")
    
    # 한국어
    ko_total = sum(category_stats[category]["ko"].values())
    print(f"\n한국어 (총 {ko_total}회):")
    if ko_total > 0:
        ko_sorted = sorted(category_stats[category]["ko"].items(), key=lambda x: x[1], reverse=True)
        for marker, count in ko_sorted:
            if count > 0:
                percentage = (count / ko_total) * 100
                print(f"  {marker}: {count}회 ({percentage:.1f}%)")
    else:
        print("  (해당 표현 없음)")
    
    # 비교
    ratio = ko_total / ru_total if ru_total > 0 else 0
    print(f"\n📈 비교: KO/RU = {ratio:.2f}")

# 10. 전체 통계 요약
print("\n" + "="*80)
print("📋 전체 통계 요약")
print("="*80)

ru_grand_total = sum(ru_frequency.values())
ko_grand_total = sum(ko_frequency.values())
total_sentences = len(df)

print(f"총 문장 수: {total_sentences:,}개")
print(f"러시아어 담화 표지 총계: {ru_grand_total:,}회 (평균 {ru_grand_total/total_sentences:.2f}개/문장)")
print(f"한국어 담화 표지 총계: {ko_grand_total:,}회 (평균 {ko_grand_total/total_sentences:.2f}개/문장)")
print(f"한국어/러시아어 비율: {ko_grand_total/ru_grand_total:.2f}")

# 11. 상위 표현 순위
print(f"\n📈 상위 담화 표지 (빈도순)")
print("-" * 50)

print("러시아어 TOP 10:")
ru_top = sorted([(k, v) for k, v in ru_frequency.items() if v > 0], key=lambda x: x[1], reverse=True)[:10]
for i, (marker, count) in enumerate(ru_top, 1):
    print(f"  {i:2d}. {marker}: {count:,}회")

print("\n한국어 TOP 10:")
ko_top = sorted([(k, v) for k, v in ko_frequency.items() if v > 0], key=lambda x: x[1], reverse=True)[:10]
for i, (marker, count) in enumerate(ko_top, 1):
    print(f"  {i:2d}. {marker}: {count:,}회")

# 12. 데이터프레임으로 정리된 결과 생성
print("\n📊 정리된 결과 테이블 생성 중...")

# 결과를 데이터프레임으로 변환
results_data = []

for category in ["접속 표지", "조건/대립 표지", "근거/지시 표지", "규범 양상 표지", "참조 표지"]:
    # 러시아어 데이터
    for marker, count in category_stats[category]["ru"].items():
        if count > 0:
            results_data.append({
                "범주": category,
                "언어": "러시아어", 
                "표현": marker,
                "빈도": count,
                "범주내비율": f"{(count/sum(category_stats[category]['ru'].values()))*100:.1f}%" if sum(category_stats[category]["ru"].values()) > 0 else "0%",
                "전체비율": f"{(count/total_sentences)*100:.2f}%"
            })
    
    # 한국어 데이터
    for marker, count in category_stats[category]["ko"].items():
        if count > 0:
            results_data.append({
                "범주": category,
                "언어": "한국어",
                "표현": marker, 
                "빈도": count,
                "범주내비율": f"{(count/sum(category_stats[category]['ko'].values()))*100:.1f}%" if sum(category_stats[category]["ko"].values()) > 0 else "0%",
                "전체비율": f"{(count/total_sentences)*100:.2f}%"
            })

results_df = pd.DataFrame(results_data)

print("\n최종 결과 테이블:")
print(results_df.to_string(index=False))

# CSV 파일로 저장
results_df.to_csv('discourse_markers_analysis_results.csv', index=False, encoding='utf-8')
print(f"\n✅ 결과가 'discourse_markers_analysis_results.csv' 파일로 저장되었습니다.")

# 13. 범주별 요약 표 생성 (요청된 양식)
print("\n" + "="*80)
print("📋 범주별 요약 표 (논문용)")
print("="*80)

summary_data = []
total_sentences = len(df)

for category in ["접속 표지", "조건/대립 표지", "근거/지시 표지", "규범 양상 표지", "참조 표지"]:
    ru_total = sum(category_stats[category]["ru"].values())
    ko_total = sum(category_stats[category]["ko"].values())
    
    # 문장 비율 계산 (해당 범주가 나타나는 문장의 비율)
    ru_sentence_ratio = (ru_total / total_sentences) * 100
    ko_sentence_ratio = (ko_total / total_sentences) * 100
    
    # 차이 계산 (한국어 - 러시아어)
    difference = ko_sentence_ratio - ru_sentence_ratio
    
    summary_data.append({
        "범주": category,
        "러시아어": f"{total_sentences}개\n({ru_sentence_ratio:.2f}%)",
        "한국어": f"{total_sentences}개\n({ko_sentence_ratio:.2f}%)", 
        "차이": f"{difference:+.2f}%"
    })

summary_df = pd.DataFrame(summary_data)

print("\n📊 범주별 분포 요약표:")
print("-" * 60)
print(f"{'범주':<12} {'러시아어':<15} {'한국어':<15} {'차이':<10}")
print("-" * 60)

for _, row in summary_df.iterrows():
    ru_info = row['러시아어'].replace('\n', ' ')
    ko_info = row['한국어'].replace('\n', ' ')
    print(f"{row['범주']:<12} {ru_info:<15} {ko_info:<15} {row['차이']:<10}")

# 14. 더 정확한 문장 단위 분석을 위한 추가 계산
print("\n" + "="*80)
print("📈 정확한 문장 단위 분포 분석")
print("="*80)

# 각 문장에서 담화 표지가 나타나는지 확인
def count_sentences_with_markers(df, patterns, language, category_markers):
    """각 범주별로 담화 표지가 포함된 문장 수 계산"""
    category_sentence_counts = defaultdict(int)
    
    for idx, row in df.iterrows():
        text = str(row[language]) if pd.notna(row[language]) else ""
        
        for category, markers in category_markers.items():
            lang_markers = markers.get(language, [])
            has_marker = False
            
            for marker in lang_markers:
                if marker in patterns[language]:
                    pattern = patterns[language][marker]
                    if re.search(pattern, text, re.IGNORECASE):
                        has_marker = True
                        break
            
            if has_marker:
                category_sentence_counts[category] += 1
    
    return category_sentence_counts

# 문장 단위 분석 실행
ru_sentence_counts = count_sentences_with_markers(df, patterns, 'ru', discourse_markers)
ko_sentence_counts = count_sentences_with_markers(df, patterns, 'ko', discourse_markers)

print("\n📊 담화 표지가 포함된 문장 수 (더 정확한 분석):")
print("-" * 70)
print(f"{'범주':<15} {'러시아어':<20} {'한국어':<20} {'차이'}")
print("-" * 70)

final_summary = []

for category in ["접속 표지", "조건/대립 표지", "근거/지시 표지", "규범 양상 표지", "참조 표지"]:
    ru_sentences = ru_sentence_counts[category]
    ko_sentences = ko_sentence_counts[category]
    
    ru_percentage = (ru_sentences / total_sentences) * 100
    ko_percentage = (ko_sentences / total_sentences) * 100
    
    difference = ko_percentage - ru_percentage
    
    print(f"{category:<15} {ru_sentences:>4}개 ({ru_percentage:>5.2f}%) {ko_sentences:>4}개 ({ko_percentage:>5.2f}%) {difference:>+6.2f}%")
    
    final_summary.append({
        "범주": category,
        "러시아어_문장수": ru_sentences,
        "러시아어_비율": f"{ru_percentage:.2f}%",
        "한국어_문장수": ko_sentences, 
        "한국어_비율": f"{ko_percentage:.2f}%",
        "차이": f"{difference:+.2f}%"
    })

# 15. 최종 요약 테이블 (논문용 형식)
print("\n" + "="*80)
print("📋 최종 논문용 표 형식")
print("="*80)

final_df = pd.DataFrame(final_summary)

print("\n표 형식 (복사해서 사용 가능):")
print("="*60)
print("범주\t\t러시아어\t\t한국어\t\t차이")
print("-"*60)

for _, row in final_df.iterrows():
    category = row['범주']
    ru_data = f"{total_sentences}개\n({row['러시아어_비율']})"
    ko_data = f"{total_sentences}개\n({row['한국어_비율']})" 
    diff = row['차이']
    
    print(f"{category}\t{ru_data.replace(chr(10), ' ')}\t{ko_data.replace(chr(10), ' ')}\t{diff}")

# CSV로 최종 요약 저장
final_df.to_csv('discourse_markers_summary_table.csv', index=False, encoding='utf-8')
print(f"\n✅ 최종 요약표가 'discourse_markers_summary_table.csv' 파일로 저장되었습니다.")

# 16. 등장 빈도 기준 러시아어-한국어 비교표
print("\n" + "="*80)
print("📊 등장 빈도 기준 러시아어-한국어 비교표")
print("="*80)

frequency_comparison = []

for category in ["접속 표지", "조건/대립 표지", "근거/지시 표지", "규범 양상 표지", "참조 표지"]:
    ru_total = sum(category_stats[category]["ru"].values())
    ko_total = sum(category_stats[category]["ko"].values())
    
    # 비율 계산 (전체 담화표지 대비)
    ru_total_markers = sum(ru_frequency.values())
    ko_total_markers = sum(ko_frequency.values())
    
    ru_percentage = (ru_total / ru_total_markers) * 100 if ru_total_markers > 0 else 0
    ko_percentage = (ko_total / ko_total_markers) * 100 if ko_total_markers > 0 else 0
    
    # 차이 계산
    difference = ko_total - ru_total
    ratio = ko_total / ru_total if ru_total > 0 else float('inf') if ko_total > 0 else 0
    
    frequency_comparison.append({
        "범주": category,
        "러시아어_빈도": ru_total,
        "러시아어_비율": f"{ru_percentage:.1f}%",
        "한국어_빈도": ko_total,
        "한국어_비율": f"{ko_percentage:.1f}%",
        "차이": difference,
        "한국어/러시아어": f"{ratio:.2f}" if ratio != float('inf') else "∞"
    })

freq_comparison_df = pd.DataFrame(frequency_comparison)

print("\n📋 등장 빈도 기준 비교표:")
print("-" * 90)
print(f"{'범주':<12} {'러시아어':<15} {'한국어':<15} {'차이':<8} {'비율(KO/RU)':<12}")
print("-" * 90)

for _, row in freq_comparison_df.iterrows():
    ru_info = f"{row['러시아어_빈도']:,}회 ({row['러시아어_비율']})"
    ko_info = f"{row['한국어_빈도']:,}회 ({row['한국어_비율']})"
    diff = f"{row['차이']:+,}회"
    ratio = row['한국어/러시아어']
    
    print(f"{row['범주']:<12} {ru_info:<15} {ko_info:<15} {diff:<8} {ratio:<12}")

# 17. 러시아어 범주별 순위표
print("\n" + "="*80)
print("🇷🇺 러시아어 담화 표지 범주별 순위")
print("="*80)

ru_category_ranking = []
for category in ["접속 표지", "조건/대립 표지", "근거/지시 표지", "규범 양상 표지", "참조 표지"]:
    total = sum(category_stats[category]["ru"].values())
    ru_category_ranking.append((category, total))

ru_category_ranking.sort(key=lambda x: x[1], reverse=True)

print("\n📊 러시아어 범주별 순위:")
print("-" * 50)
print(f"{'순위':<4} {'범주':<15} {'빈도':<10} {'비율'}")
print("-" * 50)

ru_total_all = sum(ru_frequency.values())
for rank, (category, frequency) in enumerate(ru_category_ranking, 1):
    percentage = (frequency / ru_total_all) * 100
    print(f"{rank:<4} {category:<15} {frequency:,}회{'':<5} {percentage:.1f}%")

# 18. 한국어 범주별 순위표  
print("\n" + "="*80)
print("🇰🇷 한국어 담화 표지 범주별 순위")
print("="*80)

ko_category_ranking = []
for category in ["접속 표지", "조건/대립 표지", "근거/지시 표지", "규범 양상 표지", "참조 표지"]:
    total = sum(category_stats[category]["ko"].values())
    ko_category_ranking.append((category, total))

ko_category_ranking.sort(key=lambda x: x[1], reverse=True)

print("\n📊 한국어 범주별 순위:")
print("-" * 50)
print(f"{'순위':<4} {'범주':<15} {'빈도':<10} {'비율'}")
print("-" * 50)

ko_total_all = sum(ko_frequency.values())
for rank, (category, frequency) in enumerate(ko_category_ranking, 1):
    percentage = (frequency / ko_total_all) * 100
    print(f"{rank:<4} {category:<15} {frequency:,}회{'':<5} {percentage:.1f}%")

# 19. 범주별 순위 비교 요약
print("\n" + "="*80)
print("📈 러시아어 vs 한국어 범주별 순위 비교")
print("="*80)

print("\n📊 순위 비교표:")
print("-" * 70)
print(f"{'순위':<4} {'러시아어':<20} {'빈도':<8} {'한국어':<20} {'빈도'}")
print("-" * 70)

max_ranks = max(len(ru_category_ranking), len(ko_category_ranking))
for i in range(max_ranks):
    ru_info = f"{ru_category_ranking[i][0]} ({ru_category_ranking[i][1]:,}회)" if i < len(ru_category_ranking) else "-"
    ko_info = f"{ko_category_ranking[i][0]} ({ko_category_ranking[i][1]:,}회)" if i < len(ko_category_ranking) else "-"
    
    ru_category = ru_category_ranking[i][0] if i < len(ru_category_ranking) else ""
    ru_freq = f"{ru_category_ranking[i][1]:,}" if i < len(ru_category_ranking) else ""
    ko_category = ko_category_ranking[i][0] if i < len(ko_category_ranking) else ""
    ko_freq = f"{ko_category_ranking[i][1]:,}" if i < len(ko_category_ranking) else ""
    
    print(f"{i+1:<4} {ru_category:<20} {ru_freq:<8} {ko_category:<20} {ko_freq}")

# 20. 데이터프레임으로 저장
# 빈도 비교표
freq_comparison_df.to_csv('frequency_comparison_table.csv', index=False, encoding='utf-8')

# 러시아어 순위표
ru_ranking_df = pd.DataFrame([(rank, cat, freq, f"{(freq/ru_total_all)*100:.1f}%") 
                             for rank, (cat, freq) in enumerate(ru_category_ranking, 1)],
                            columns=['순위', '범주', '빈도', '비율'])
ru_ranking_df.to_csv('russian_category_ranking.csv', index=False, encoding='utf-8')

# 한국어 순위표  
ko_ranking_df = pd.DataFrame([(rank, cat, freq, f"{(freq/ko_total_all)*100:.1f}%") 
                             for rank, (cat, freq) in enumerate(ko_category_ranking, 1)],
                            columns=['순위', '범주', '빈도', '비율'])
ko_ranking_df.to_csv('korean_category_ranking.csv', index=False, encoding='utf-8')

print(f"\n✅ 추가 분석 결과 파일들이 저장되었습니다:")
print("  - frequency_comparison_table.csv (빈도 비교표)")
print("  - russian_category_ranking.csv (러시아어 순위표)")
print("  - korean_category_ranking.csv (한국어 순위표)")

print("\n🎉 모든 분석 완료!")
