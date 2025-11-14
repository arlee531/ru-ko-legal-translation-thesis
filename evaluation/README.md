# BLEU & ChrF Evaluation for Russian-Korean Legal Translation

This repository contains code for evaluating Russian-Korean legal translation models using BLEU and ChrF metrics.

## Overview

This evaluation script calculates BLEU and chrF scores for five translation systems:
- Google Translate (baseline)
- NLLB-200 Original
- NLLB-200 Fine-tuned
- mBART-50 Original
- mBART-50 Fine-tuned

## Requirements

```bash
pip install pandas sacrebleu
```

### Package Versions
- `pandas`: >= 1.3.0
- `sacrebleu`: 2.5.1

## Input Data Format

The script expects CSV files with the following columns:
- `id`: Sentence pair identifier
- `ru`: Russian source text
- `ko`: Korean reference translation
- `translation_*`: Model translation output (column name can vary)

### Example CSV Structure

```csv
id,ru,ko,translation_google
1,"Текст на русском","한국어 텍스트","번역된 텍스트"
2,"Другой текст","다른 텍스트","번역 결과"
```

## Usage

### Basic Usage

```python
from bleu_chrf_evaluation import calculate_scores, load_translation_files, save_results
import pandas as pd

# Define file paths
file_paths = {
    'Google Translate': 'google_translate.csv',
    'NLLB Fine-tuned': 'nllb_finetuned.csv',
    # Add more models...
}

# Load files
dataframes = load_translation_files(file_paths)

# Calculate scores
all_results = []
for model_name, df in dataframes.items():
    references = df['ko'].tolist()
    hypotheses = df['translation'].tolist()
    results = calculate_scores(references, hypotheses, model_name)
    if results:
        all_results.append(results)

# Save results
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('bleu', ascending=False)
save_results(results_df, output_dir='./results/')
```

### Command Line Usage

```bash
python bleu_chrf_evaluation.py
```

## Output Files

The script generates three output files:

1. **evaluation_results.csv**: Tabular results
2. **evaluation_results.json**: JSON format results
3. **evaluation_report.txt**: Human-readable report

### Example Output

```
======================================================================
Translation Model Evaluation Results
======================================================================

Evaluation Date: 2025-01-05 14:30:22
Total Test Samples: 1,000

======================================================================
BLEU & ChrF Scores
======================================================================

mBART Fine-tuned
----------------------------------------------------------------------
Valid Samples:  1,000 / 1,000
BLEU Score:     23.73
ChrF Score:     45.66

NLLB Fine-tuned
----------------------------------------------------------------------
Valid Samples:  1,000 / 1,000
BLEU Score:     19.46
ChrF Score:     39.00
```

## Metrics Explanation

### BLEU (Bilingual Evaluation Understudy)
- Range: 0-100 (higher is better)
- Measures n-gram overlap between hypothesis and reference translations
- Calculated at corpus level using 1-gram to 4-gram

### ChrF (Character n-gram F-score)
- Range: 0-100 (higher is better)
- Character-level evaluation metric
- Uses 1-gram to 6-gram character sequences
- More robust for morphologically rich languages

## Implementation Details

- Uses `sacrebleu` library for standardized metric calculation
- Corpus-level scoring (single score for entire dataset)
- Handles missing values and empty strings
- Filters out invalid translation pairs

## Related Research

This evaluation code is part of a doctoral dissertation on:
- Domain-specific machine translation for Russian-Korean legal texts
- Fine-tuning multilingual models (NLLB-200, mBART-50)
- Parallel corpus construction for low-resource language pairs

## License

MIT License

## Acknowledgments

- SacreBLEU library by Matt Post
- NLLB-200 by Meta AI
- mBART-50 by Facebook AI
