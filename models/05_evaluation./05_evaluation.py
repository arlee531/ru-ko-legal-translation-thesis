import pandas as pd
from sacrebleu import BLEU, CHRF
from google.colab import files

print("📂 번역 결과 파일 업로드 및 종합 평가")
print("=" * 50)

# 1. 파일 업로드
print("1️⃣ 파일 업로드를 시작합니다...")
print("다음 파일들을 업로드해주세요:")
print("- 참조번역(정답), 구글번역, 내모델번역이 포함된 CSV 파일")
print("- 또는 각각 별도의 파일들")

uploaded = files.upload()

# 2. 업로드된 파일 확인
print("\n2️⃣ 업로드된 파일 확인:")
import os
for file in os.listdir('.'):
    if file.endswith(('.csv', '.txt')):
        print(f"   - {file}")

# 3. 데이터 로딩 (CSV 파일인 경우)
print("\n3️⃣ 데이터 로딩 중...")
# 가장 큰 CSV 파일을 자동으로 선택하거나 수동으로 지정
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if csv_files:
    main_file = max(csv_files, key=lambda f: os.path.getsize(f))
    print(f"   메인 파일: {main_file}")
    df = pd.read_csv(main_file)
    
    print(f"   데이터 형태: {df.shape}")
    print(f"   컬럼명: {list(df.columns)}")
    print("\n   첫 3줄 미리보기:")
    print(df.head(3))
    
    # 4. 컬럼 매핑 (사용자가 수정 필요)
    print("\n4️⃣ 데이터 추출 중...")
    print("⚠️  아래 컬럼명을 실제 파일에 맞게 수정하세요:")
    
    # 실제 컬럼명으로 접근
    try:
        # 방법 1: 컬럼명으로 접근
        references = df['reference'].tolist()  # 정답 번역
        google_preds = df['google'].tolist()  # 구글 번역
        model_preds = df['model'].tolist()  # 내 모델
        
        print("   ✅ 컬럼명으로 데이터 추출 성공")
        
    except KeyError as e:
        print(f"   ❌ 컬럼명 오류: {e}")
        print("   📝 사용 가능한 컬럼명:", list(df.columns))
        
        # 방법 2: 인덱스로 접근 (컬럼 순서 확인 후 수정)
        print("   🔄 인덱스로 접근 시도...")
        references = df.iloc[:, 0].tolist()  # 첫 번째 컬럼
        google_preds = df.iloc[:, 1].tolist()  # 두 번째 컬럼  
        model_preds = df.iloc[:, 2].tolist()  # 세 번째 컬럼
        
        print("   ✅ 인덱스로 데이터 추출 완료")

else:
    print("   ❌ CSV 파일을 찾을 수 없습니다.")
    # 텍스트 파일로 처리하는 경우
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    if len(txt_files) >= 3:
        print("   📄 텍스트 파일들을 읽어옵니다...")
        
        # 파일명에 따라 자동 분류 또는 순서대로 읽기
        with open(txt_files[0], 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f]
        with open(txt_files[1], 'r', encoding='utf-8') as f:
            google_preds = [line.strip() for line in f]
        with open(txt_files[2], 'r', encoding='utf-8') as f:
            model_preds = [line.strip() for line in f]
    else:
        print("   ❌ 충분한 파일이 없습니다.")
        exit()

# 5. 데이터 검증
print(f"\n5️⃣ 데이터 검증:")
print(f"   참조번역 개수: {len(references)}")
print(f"   구글번역 개수: {len(google_preds)}")
print(f"   내모델번역 개수: {len(model_preds)}")

# 데이터 개수가 일치하는지 확인
if not (len(references) == len(google_preds) == len(model_preds)):
    print("   ⚠️  데이터 개수가 일치하지 않습니다!")
    min_len = min(len(references), len(google_preds), len(model_preds))
    references = references[:min_len]
    google_preds = google_preds[:min_len]
    model_preds = model_preds[:min_len]
    print(f"   🔄 {min_len}개로 맞춰서 진행합니다.")

# 6. 샘플 데이터 확인
print(f"\n6️⃣ 번역 결과 샘플 (처음 3개):")
for i in range(min(3, len(references))):
    print(f"\n   {i+1}번째:")
    print(f"   참조: {references[i]}")
    print(f"   구글: {google_preds[i]}")
    print(f"   내모델: {model_preds[i]}")

# 7. 평가 실행
print(f"\n7️⃣ BLEU/chrF 평가 실행")
print("-" * 40)

# 평가 객체 생성
bleu_evaluator = BLEU()
chrf_evaluator = CHRF()

# 다중 참조 형식으로 변환
references_multi = [references]

# 구글 번역 평가
print("\n🔵 구글 번역 성능:")
google_bleu = bleu_evaluator.corpus_score(google_preds, references_multi)
google_chrf = chrf_evaluator.corpus_score(google_preds, references_multi)

print(f"   BLEU: {google_bleu.score:.2f}")
print(f"   chrF: {google_chrf.score:.2f}")

# 내 모델 평가
print("\n🟢 내 모델 성능:")
model_bleu = bleu_evaluator.corpus_score(model_preds, references_multi)
model_chrf = chrf_evaluator.corpus_score(model_preds, references_multi)

print(f"   BLEU: {model_bleu.score:.2f}")
print(f"   chrF: {model_chrf.score:.2f}")

# 8. 비교 결과
print(f"\n8️⃣ 성능 비교 결과")
print("=" * 30)

bleu_diff = model_bleu.score - google_bleu.score
chrf_diff = model_chrf.score - google_chrf.score
bleu_pct = (bleu_diff / google_bleu.score) * 100 if google_bleu.score > 0 else 0
chrf_pct = (chrf_diff / google_chrf.score) * 100 if google_chrf.score > 0 else 0

print(f"📊 최종 결과:")
print(f"   구글번역: BLEU {google_bleu.score:.2f}, chrF {google_chrf.score:.2f}")
print(f"   내 모델:  BLEU {model_bleu.score:.2f}, chrF {model_chrf.score:.2f}")
print(f"   개선도:   BLEU {bleu_diff:+.2f}점 ({bleu_pct:+.1f}%), chrF {chrf_diff:+.2f}점 ({chrf_pct:+.1f}%)")

# 9. 상세 분석
print(f"\n9️⃣ 상세 분석:")
print(f"   내 모델 BLEU 세부점수: {[f'{p:.1f}' for p in model_bleu.precisions]}")
print(f"   내 모델 BP: {model_bleu.bp:.3f}")

# 빈 번역 검사
google_empty = sum(1 for x in google_preds if not x or str(x).strip() == '')
model_empty = sum(1 for x in model_preds if not x or str(x).strip() == '')
print(f"   빈 번역 - 구글: {google_empty}개, 내모델: {model_empty}개")

# 10. 결과 저장
results_df = pd.DataFrame({
    'reference': references,
    'google_translation': google_preds,
    'my_model': model_preds
})

# 성능 점수도 별도 저장
summary_df = pd.DataFrame({
    'system': ['Google Translate', 'My Model'],
    'BLEU': [google_bleu.score, model_bleu.score],
    'chrF': [google_chrf.score, model_chrf.score]
})

results_df.to_csv('detailed_results.csv', index=False, encoding='utf-8')
summary_df.to_csv('performance_summary.csv', index=False, encoding='utf-8')

print(f"\n💾 결과 저장 완료:")
print(f"   - detailed_results.csv: 전체 번역 결과")
print(f"   - performance_summary.csv: 성능 요약")

print(f"\n✅ 평가 완료!")
