"""
Legal Term Extraction from Mid-Frequency Range
중간빈도대 법률 전문용어 추출

Author: [Your Name]
Date: 2025
Description: 통계적 방법과 패턴 매칭을 활용한 법률 전문용어 추출
"""

import re
from collections import Counter
import pandas as pd

def extract_mid_frequency_terms(frequency_dist, lang='ko'):
    """
    중간빈도 어휘 추출
    
    Args:
        frequency_dist (Counter): 전체 빈도 분포
        lang (str): 'ko' 또는 'ru'
        
    Returns:
        list: (단어, 빈도) 튜플 리스트
    """
    if lang == 'ko':
        min_freq, max_freq = 5, 150
    elif lang == 'ru':
        min_freq, max_freq = 10, 300
    else:
        raise ValueError("lang은 'ko' 또는 'ru'여야 합니다")
    
    mid_freq_terms = [(word, count) for word, count in frequency_dist.items() 
                      if min_freq <= count <= max_freq]
    
    return mid_freq_terms

def is_korean_legal_term(word):
    """
    한국어 법률 전문용어 판별
    
    Args:
        word (str): 판별할 단어
        
    Returns:
        bool: 법률 전문용어 여부
    """
    # 법률 관련 키워드
    legal_keywords = [
        # 기본 법률 용어
        '법', '조', '항', '호', '제', '규정', '조세', '세금', '관세', 
        '권리', '의무', '절차', '기관', '기구', '협정', '협약', '계약',
        
        # 국가/정부 관련
        '국가', '정부', '연방', '동맹', '회원', '주체', '기관', '청', '부', '원',
        
        # 세무 관련
        '납세', '과세', '면세', '소비세', '부가세', '특별소비세', '관세청',
        '세무', '징수', '신고', '납부', '환급',
        
        # 지식재산 관련
        '특허', '저작', '상표', '지식재산', '저작물', '발명', '실용신안',
        
        # 사법 관련
        '법원', '재판', '판결', '소송', '고발', '기소', '처벌', '형벌',
        '헌법재판소', '대법원', '고등법원', '지방법원',
        
        # 법령 관련
        '민법', '형법', '상법', '행정법', '헌법', '국제법', '조약',
        '법률', '시행령', '시행규칙', '고시', '훈령',
        
        # 절차 관련
        '통관', '신청', '허가', '승인', '등록', '신고', '보고',
        '검사', '검토', '심사', '심의', '결정', '처분'
    ]
    
    # 법률 관련 접미사
    legal_suffixes = ['법', '제', '조', '항', '호', '권', '세', '료', '금', '청', '부', '원', '소']
    
    # 법률 관련 접두사
    legal_prefixes = ['법', '조', '제', '규정', '협정', '헌법', '민법', '형법', '상법']
    
    # 1. 키워드 포함 검사
    for keyword in legal_keywords:
        if keyword in word:
            return True
    
    # 2. 접미사 검사 (2글자 이상)
    for suffix in legal_suffixes:
        if word.endswith(suffix) and len(word) > 2:
            return True
    
    # 3. 접두사 검사 (2글자 이상)
    for prefix in legal_prefixes:
        if word.startswith(prefix) and len(word) > 2:
            return True
    
    # 4. 한자어 복합어 패턴 (3글자 이상, 조사 없음)
    if len(word) >= 3 and not any(char in word for char in '을를이가에서의로부터와과도만'):
        # 기관명 패턴
        institution_patterns = [
            r'.*국$', r'.*청$', r'.*부$', r'.*원$', r'.*소$',  # 기관명
            r'.*서$', r'.*증$', r'.*장$', r'.*표$',          # 문서명
            r'.*자$', r'.*인$', r'.*사$'                    # 주체명
        ]
        for pattern in institution_patterns:
            if re.match(pattern, word):
                return True
        
        # 법률 개념 패턴
        concept_patterns = [
            r'.*권$', r'.*법$', r'.*제$', r'.*세$',         # 권리/법령/세금
            r'.*료$', r'.*금$', r'.*비$', r'.*액$'          # 비용 관련
        ]
        for pattern in concept_patterns:
            if re.match(pattern, word):
                return True
    
    return False

def is_russian_legal_term(word):
    """
    러시아어 법률 전문용어 판별
    
    Args:
        word (str): 판별할 단어
        
    Returns:
        bool: 법률 전문용어 여부
    """
    # 법률 관련 키워드 (어간 포함)
    legal_keywords = [
        # 기본 법률 용어
        'право', 'закон', 'кодекс', 'статья', 'пункт', 'правил', 'норм',
        
        # 세무 관련
        'налог', 'налож', 'платеж', 'взнос', 'сбор', 'пошлин', 'льгот',
        
        # 관세 관련
        'таможен', 'товар', 'ввоз', 'вывоз', 'импорт', 'экспорт',
        
        # 국가/정부 관련
        'федерац', 'государств', 'власт', 'орган', 'ведомство', 'служб',
        'министерств', 'комитет', 'агентств', 'инспекц',
        
        # 사법 관련
        'суд', 'решение', 'постановление', 'определение', 'приговор',
        'иск', 'дело', 'производств', 'процесс',
        
        # 지식재산 관련
        'интеллектуальн', 'патент', 'изобретен', 'полезн', 'промышленн',
        'авторск', 'товарн', 'знак', 'лицензи',
        
        # 법률행위 관련
        'договор', 'соглашение', 'сделк', 'обязательств', 'ответственност',
        'нарушени', 'штраф', 'санкци', 'взыскан',
        
        # 절차 관련
        'процедур', 'порядок', 'регистрац', 'уведомлен', 'заявлен',
        'разрешен', 'лицензирован', 'контрол', 'провер',
        
        # 문서 관련
        'документ', 'справк', 'свидетельств', 'удостоверен', 'сертификат',
        
        # 재산 관련
        'собственност', 'имущество', 'владен', 'пользован', 'распоряжен'
    ]
    
    # 법률 관련 접미사
    legal_suffixes = [
        'ство', 'ние', 'ция', 'сия', 'тель', 'щик', 'льщик', 'ник',
        'ость', 'ение', 'ание', 'ение'
    ]
    
    word_lower = word.lower()
    
    # 1. 키워드 포함 검사
    for keyword in legal_keywords:
        if keyword in word_lower:
            return True
    
    # 2. 접미사 검사 (4글자 이상)
    for suffix in legal_suffixes:
        if word_lower.endswith(suffix) and len(word) > 4:
            return True
    
    # 3. 대문자로 시작하는 고유명사 (기관명, 법령명 등)
    if word[0].isupper() and len(word) > 3:
        # 일반적인 전치사나 접속사 제외
        common_words = ['Если', 'При', 'Для', 'После', 'Перед', 'Через']
        if word not in common_words:
            return True
    
    # 4. 숫자 포함 (조항 번호 등)
    if re.search(r'\d', word) and len(word) > 1:
        return True
    
    return False

def extract_legal_terms(frequency_dist, lang='ko', top_n=20):
    """
    법률 전문용어 추출 메인 함수
    
    Args:
        frequency_dist (Counter): 전체 빈도 분포
        lang (str): 'ko' 또는 'ru'
        top_n (int): 추출할 상위 N개
        
    Returns:
        dict: 추출 결과 정보
    """
    # 1. 고빈도 어휘에서 수동 추출 (상위 100개에서)
    top_100 = frequency_dist.most_common(100)
    
    if lang == 'ko':
        manual_terms = [
            ('러시아연방', frequency_dist.get('러시아연방', 0)),
            ('관세동맹', frequency_dist.get('관세동맹', 0)),
            ('배타적', frequency_dist.get('배타적', 0)),
            ('조', frequency_dist.get('조', 0)),
            ('회원국의', frequency_dist.get('회원국의', 0)),
            ('조세', frequency_dist.get('조세', 0)),
            ('지식재산권', frequency_dist.get('지식재산권', 0)),
            ('특별소비세', frequency_dist.get('특별소비세', 0)),
            ('제1항', frequency_dist.get('제1항', 0)),
            ('세관기관이', frequency_dist.get('세관기관이', 0))
        ]
        judge_func = is_korean_legal_term
    else:  # ru
        manual_terms = [
            ('Федерации', frequency_dist.get('Федерации', 0)),
            ('Российской', frequency_dist.get('Российской', 0)),
            ('товаров', frequency_dist.get('товаров', 0)),
            ('таможенного', frequency_dist.get('таможенного', 0)),
            ('Кодекса', frequency_dist.get('Кодекса', 0)),
            ('права', frequency_dist.get('права', 0)),
            ('налога', frequency_dist.get('налога', 0)),
            ('налогового', frequency_dist.get('налогового', 0)),
            ('таможенных', frequency_dist.get('таможенных', 0)),
            ('интеллектуальной', frequency_dist.get('интеллектуальной', 0))
        ]
        judge_func = is_russian_legal_term
    
    # 2. 중간빈도대에서 자동 추출
    mid_freq_terms = extract_mid_frequency_terms(frequency_dist, lang)
    
    legal_candidates = []
    for word, count in mid_freq_terms:
        if judge_func(word):
            legal_candidates.append((word, count))
    
    # 빈도순 정렬
    legal_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 3. 최종 top_n 선정 (수동 + 자동)
    # 수동 추출된 것들 중 빈도가 0이 아닌 것들
    manual_valid = [(w, c) for w, c in manual_terms if c > 0]
    
    # 중간빈도에서 추출된 것들 중 수동 추출 목록에 없는 것들
    manual_words = set(w for w, c in manual_valid)
    auto_candidates = [(w, c) for w, c in legal_candidates if w not in manual_words]
    
    # 최종 결합
    all_terms = manual_valid + auto_candidates
    all_terms.sort(key=lambda x: x[1], reverse=True)
    
    final_terms = all_terms[:top_n]
    
    return {
        'final_terms': final_terms,
        'manual_terms': manual_valid,
        'auto_candidates': auto_candidates[:15],  # 상위 15개만
        'mid_freq_count': len(mid_freq_terms),
        'legal_candidates_count': len(legal_candidates)
    }

def print_extraction_results(results, language):
    """
    추출 결과 출력
    """
    print(f"\n{'='*60}")
    print(f"{language} 법률 전문용어 추출 결과")
    print(f"{'='*60}")
    
    print(f"중간빈도 어휘 수: {results['mid_freq_count']:,}개")
    print(f"법률 전문용어 후보: {results['legal_candidates_count']:,}개")
    
    print(f"\nTop 100에서 수동 추출 ({len(results['manual_terms'])}개):")
    for i, (word, count) in enumerate(results['manual_terms'], 1):
        print(f"  {i:2d}. {word:<20} ({count:,}회)")
    
    print(f"\n중간빈도대에서 자동 추출 상위 15개:")
    for i, (word, count) in enumerate(results['auto_candidates'], 1):
        print(f"  {i:2d}. {word:<20} ({count:,}회)")
    
    print(f"\n최종 법률 전문용어 Top 20:")
    for i, (word, count) in enumerate(results['final_terms'], 1):
        print(f"  {i:2d}. {word:<20} ({count:,}회)")

def save_results_to_csv(ko_results, ru_results, output_path="legal_terms_results.csv"):
    """
    결과를 CSV 파일로 저장
    """
    data = []
    max_len = max(len(ko_results['final_terms']), len(ru_results['final_terms']))
    
    for i in range(max_len):
        row = {}
        if i < len(ko_results['final_terms']):
            ko_term, ko_freq = ko_results['final_terms'][i]
            row['korean_rank'] = i + 1
            row['korean_term'] = ko_term
            row['korean_frequency'] = ko_freq
        
        if i < len(ru_results['final_terms']):
            ru_term, ru_freq = ru_results['final_terms'][i]
            row['russian_rank'] = i + 1
            row['russian_term'] = ru_term
            row['russian_frequency'] = ru_freq
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n결과가 {output_path}에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 이 코드는 basic_vocab_analysis.py에서 생성된 frequency distribution을 사용
    # ko_freq_dist, ru_freq_dist가 이미 생성되어 있어야 함
    
    print("법률 전문용어 추출을 시작합니다...")
    
    # 한국어 법률 전문용어 추출
    ko_results = extract_legal_terms(ko_freq_dist, 'ko', 20)
    print_extraction_results(ko_results, "한국어")
    
    # 러시아어 법률 전문용어 추출  
    ru_results = extract_legal_terms(ru_freq_dist, 'ru', 20)
    print_extraction_results(ru_results, "러시아어")
    
    # 결과 저장
    save_results_to_csv(ko_results, ru_results)
    
    print("\n법률 전문용어 추출이 완료되었습니다!")
