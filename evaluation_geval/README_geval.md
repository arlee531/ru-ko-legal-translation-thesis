# G-Eval: Translation Quality Evaluation using GPT-4o

Automated evaluation of Russian-Korean legal translations using GPT-4o for fluency and fidelity assessment.

## Overview

This tool implements G-Eval methodology to evaluate translation quality across two dimensions:
- **Fluency (유창성)**: Naturalness, grammar, and appropriateness for legal Korean text
- **Fidelity (충실성)**: Accuracy and completeness of meaning transfer from Russian source

## Features

- ✅ Dual-criterion evaluation (fluency + fidelity)
- ✅ 5-point Likert scale scoring (1-5)
- ✅ Structured JSON output with detailed analysis
- ✅ Batch processing for multiple translation systems
- ✅ Automatic retry with exponential backoff
- ✅ Progress tracking and statistics generation
- ✅ Comprehensive reporting

## Requirements

```bash
pip install openai pandas tqdm
```

### Package Versions
- `openai`: >= 1.0.0
- `pandas`: >= 1.3.0
- `tqdm`: >= 4.60.0

## Input Data Format

CSV file with the following columns:
- `id`: Sentence identifier
- `ru`: Russian source text
- `reference`: Korean reference translation (optional)
- `translation_*`: One or more model translation columns

### Example CSV Structure

```csv
id,ru,reference,translation_google,translation_nllb,translation_mbart
1,"Текст","참조 번역","구글 번역","NLLB 번역","mBART 번역"
```

## Usage

### Basic Usage

```python
from geval_evaluation import evaluate_dataset, calculate_statistics
import pandas as pd
import openai

# Set API key
openai.api_key = "your-api-key"

# Load data
df = pd.read_csv("translations.csv")

# Identify model columns
model_columns = ['translation_google', 'translation_nllb', 'translation_mbart']

# Run evaluation
results_df = evaluate_dataset(df, model_columns, output_path="results.csv")

# Calculate statistics
stats_df = calculate_statistics(results_df)
print(stats_df)
```

### Command Line Usage

```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key"

# Run evaluation
python geval_evaluation.py
```

The script will prompt you for:
1. OpenAI API key (if not set as environment variable)
2. Path to input CSV file

## Evaluation Criteria

### Fluency (유창성)

| Score | Description |
|-------|-------------|
| 5 | Excellent: Very natural Korean, appropriate legal style, no errors |
| 4 | Good: Generally natural, minor style issues |
| 3 | Fair: Understandable but awkward expressions, not typical legal style |
| 2 | Poor: Near-ungrammatical or overly literal, barely readable |
| 1 | Very Poor: Incomprehensible or severely unnatural, unusable as legal text |

### Fidelity (충실성)

| Score | Description |
|-------|-------------|
| 5 | Excellent: Complete and accurate meaning transfer, all legal elements preserved |
| 4 | Good: Core meaning preserved, minor nuance differences |
| 3 | Fair: Main meaning transferred, some distortions present |
| 2 | Poor: Significant distortions, key legal elements mistranslated |
| 1 | Very Poor: Meaning largely lost, major legal implications distorted |

## Output Files

### 1. Results CSV (`geval_results.csv`)

```csv
id,model,fluency_score,fluency_analysis,fidelity_score,fidelity_analysis,average_score
1,translation_google,4,"Natural but...",5,"Accurate...",4.5
```

### 2. Evaluation Report (`geval_report.txt`)

```
======================================================================
G-Eval Translation Quality Evaluation Report
======================================================================

Average Scores by Model
----------------------------------------------------------------------

translation_mbart
----------------------------------------------------------------------
Fluency:  4.23 (±0.65)
Fidelity: 4.45 (±0.58)
Average:  4.34 (±0.52)
```

## API Usage and Costs

### Rate Limiting
- Default delay: 0.5 seconds between API calls
- Automatic retry with exponential backoff
- Configurable via `rate_limit_delay` parameter

### Cost Estimation

For 350 samples × 5 models × 2 criteria = 3,500 evaluations:
- Model: GPT-4o
- Estimated input tokens per call: ~800
- Estimated output tokens per call: ~150
- Approximate cost: $10-15 USD

### Cost Optimization Tips
1. Start with a small sample to test
2. Use environment variable for API key
3. Implement checkpointing for large datasets
4. Consider batching if API supports it

## Example: Full Evaluation Pipeline

```python
import pandas as pd
import openai
from geval_evaluation import (
    evaluate_dataset, 
    calculate_statistics, 
    generate_report
)

# Setup
openai.api_key = "your-api-key"

# Load data
df = pd.read_csv("sample_350_with_translations.csv")

# Define models to evaluate
models = [
    'translation_google',
    'translation_nllb_original',
    'translation_nllb_finetuned',
    'translation_mbart_original',
    'translation_mbart_finetuned'
]

# Run evaluation
results = evaluate_dataset(
    df, 
    models, 
    output_path="geval_results.csv",
    rate_limit_delay=0.5
)

# Calculate statistics
stats = calculate_statistics(results)

# Generate report
report = generate_report(results, stats)
with open("geval_report.txt", "w") as f:
    f.write(report)

print("✓ Evaluation complete!")
print(f"Average scores:\n{stats}")
```

## Limitations

1. **Model Bias**: GPT-4o's evaluation may reflect its own biases
2. **Consistency**: Minor variations possible across evaluations
3. **Language Limitations**: Best for Korean legal text evaluation
4. **Cost**: Large-scale evaluations require significant API credits
5. **Rate Limits**: OpenAI API has rate limits (adjust delays accordingly)

## Best Practices

1. **Validate on small sample first** to check prompt effectiveness
2. **Compare with human evaluation** for reliability assessment
3. **Use consistent temperature (0)** for deterministic results
4. **Monitor API costs** during evaluation
5. **Save intermediate results** for large datasets

## Troubleshooting

### API Connection Errors
```python
# Check API key
import openai
openai.api_key = "your-key"
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "test"}]
)
```

### Rate Limit Errors
- Increase `rate_limit_delay` parameter
- Reduce batch size
- Check OpenAI account limits

### Invalid JSON Responses
- Script includes retry logic with exponential backoff
- Falls back to score=0 if parsing fails after retries

## Related Research

This evaluation tool was developed for doctoral dissertation research on:
- Domain-specific machine translation for Russian-Korean legal texts
- Comparative evaluation of fine-tuned vs. general-purpose models
- Automated vs. manual evaluation correlation studies

## License

MIT License

## Acknowledgments

- G-Eval methodology by Liu et al.
- OpenAI GPT-4o API
- SacreBLEU for complementary metrics
