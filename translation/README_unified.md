# Russian-Korean Legal Domain Machine Translation

Translation scripts for Russian-Korean legal domain machine translation research. This repository contains code for translating Russian legal texts to Korean using multiple models.

## Overview

This repository is part of a doctoral dissertation research on Russian-Korean legal domain machine translation with domain-specific corpus fine-tuning.

**Research Question**: Does fine-tuning with domain-specific legal corpus improve translation quality for low-resource language pairs?

## Models

We compare 4 translation approaches:

1. **Google Translate** - Commercial baseline
2. **NLLB-200 Original** - Pretrained multilingual model (600M parameters)
3. **NLLB-200 Fine-tuned** - Fine-tuned with Russian-Korean legal corpus
4. **mBART-50 Fine-tuned** - Fine-tuned with Russian-Korean legal corpus

## Repository Structure

```
translation/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── translate_google.py              # Google Translate baseline
├── translate_nllb_original.py       # NLLB-200 original model
├── translate_mbart_original.py      # mBART-50 original model
├── translate_nllb_finetuned.py      # NLLB-200 fine-tuned model
└── translate_mbart_finetuned.py     # mBART-50 fine-tuned model
```

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

All scripts follow the same interface:

```bash
python translate_[MODEL].py \
    --input path/to/input.csv \
    --output path/to/output.csv
```

### Input Format

Input CSV file must contain a column named `ru` with Russian source texts:

```csv
ru,ko
"Статья 1. Основные понятия","제1조. 기본 개념"
"Гражданский кодекс Российской Федерации","러시아연방 민법전"
...
```

### Output Format

Each script adds a new column with translations:

- `translate_google.py` → adds `google_translate` column
- `translate_nllb_original.py` → adds `nllb_original` column
- `translate_mbart_original.py` → adds `mbart_original` column
- `translate_nllb_finetuned.py` → adds `nllb_finetuned` column
- `translate_mbart_finetuned.py` → adds `mbart_finetuned` column

## Individual Script Details

### 1. Google Translate (Baseline)

```bash
python translate_google.py \
    --input test_data.csv \
    --output results_google.csv
```

- **API**: Google Translate API
- **Performance**: Fast (~1-2 minutes for 1000 sentences)
- **Requirements**: Internet connection

### 2. NLLB-200 Original Model

```bash
python translate_nllb_original.py \
    --input test_data.csv \
    --output results_nllb_original.csv
```

- **Model**: facebook/nllb-200-distilled-600M
- **Parameters**: 600M
- **Languages**: 200 languages including Russian (rus_Cyrl) and Korean (kor_Hang)
- **Performance**: ~10-15 minutes for 1000 sentences (GPU)

**Translation Parameters**:
- Source Language: rus_Cyrl
- Target Language: kor_Hang
- Max Length: 256 tokens
- Beam Size: 5
- Early Stopping: True

### 3. mBART-50 Original Model

```bash
python translate_mbart_original.py \
    --input test_data.csv \
    --output results_mbart_original.csv
```

- **Model**: facebook/mbart-large-50-many-to-many-mmt
- **Parameters**: 610M
- **Languages**: 50 languages including Russian (ru_RU) and Korean (ko_KR)
- **Performance**: ~10-15 minutes for 1000 sentences (GPU)

**Translation Parameters**:
- Source Language: ru_RU
- Target Language: ko_KR
- Max Length: 256 tokens
- Beam Size: 5
- Early Stopping: True

### 4. NLLB-200 Fine-tuned Model

```bash
python translate_nllb_finetuned.py \
    --input test_data.csv \
    --output results_nllb_finetuned.csv \
    --model path/to/finetuned/model
```

- **Base Model**: facebook/nllb-200-distilled-600M
- **Fine-tuning**: Russian-Korean legal parallel corpus
- **Corpus Size**: [Your corpus size]
- **Training**: [Brief training info]

### 5. mBART-50 Fine-tuned Model

```bash
python translate_mbart_finetuned.py \
    --input test_data.csv \
    --output results_mbart_finetuned.csv \
    --model path/to/finetuned/model
```

- **Base Model**: facebook/mbart-large-50-many-to-many-mmt
- **Fine-tuning**: Russian-Korean legal parallel corpus
- **Corpus Size**: [Your corpus size]
- **Training**: [Brief training info]

## Performance Benchmarks

Hardware: Google Colab (Tesla T4 GPU, 16GB RAM)

| Model | Time (1000 sentences) | GPU Memory |
|-------|----------------------|------------|
| Google Translate | ~2 minutes | N/A |
| NLLB-200 Original | ~10-15 minutes | ~4GB |
| mBART-50 Original | ~10-15 minutes | ~5GB |
| NLLB-200 Fine-tuned | ~10-15 minutes | ~4GB |
| mBART-50 Fine-tuned | ~10-15 minutes | ~5GB |

## Example Workflow

Translate test set with all models:

```bash
# 1. Google Translate baseline
python translate_google.py --input test.csv --output step1.csv

# 2. NLLB original
python translate_nllb_original.py --input step1.csv --output step2.csv

# 3. mBART original  
python translate_mbart_original.py --input step2.csv --output step3.csv

# 4. NLLB fine-tuned
python translate_nllb_finetuned.py --input step3.csv --output step4.csv --model ./models/nllb_legal

# 5. mBART fine-tuned
python translate_mbart_finetuned.py --input step4.csv --output final.csv --model ./models/mbart_legal
```

Final output will have all translation columns:
```csv
ru,ko,google_translate,nllb_original,mbart_original,nllb_finetuned,mbart_finetuned
```

## Fine-tuned Models

Fine-tuned models are available at:
- NLLB-200 Fine-tuned: [HuggingFace link or contact information]
- mBART-50 Fine-tuned: [HuggingFace link or contact information]

For access to the fine-tuned models or the legal parallel corpus, please contact [your email].

## Evaluation

For evaluation scripts (BLEU, chrF, G-EVAL, manual evaluation), see the `evaluation/` directory in the main repository.

## Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory issues:
- Reduce batch size (process one sentence at a time)
- Use CPU instead: The scripts automatically detect available devices
- Use smaller models

### Slow Performance on CPU

- GPU is highly recommended
- Expected CPU time: 1-2 hours for 1000 sentences
- Consider using Google Colab free GPU

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{yourname2025,
  title={Russian-Korean Legal Domain Machine Translation with Domain-Specific Corpus},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

## License

MIT License - See LICENSE file for details

## Contact

**Author**: [Your Name]  
**Email**: [Your Email]  
**Institution**: [Your University]  
**GitHub**: [Your GitHub]

## Acknowledgments

This research was supported by [funding source if applicable].

Models used in this research:
- NLLB-200: Meta AI (https://github.com/facebookresearch/fairseq/tree/nllb)
- mBART-50: Meta AI (https://github.com/facebookresearch/fairseq/tree/main/examples/mbart)
- Google Translate: Google LLC

## Related Publications

[List any related papers or preprints]
