# NLLB-200 Base Model Translation

This script translates Russian legal texts to Korean using the pretrained NLLB-200 model (facebook/nllb-200-distilled-600M) without fine-tuning.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python translate_nllb_original.py \
    --input test_data.csv \
    --output results_nllb_original.csv
```

### Input Format

The input CSV file must contain a column named `ru` with Russian source texts:

```csv
ru,ko
"Статья 1. Основные понятия","제1조. 기본 개념"
"Гражданский кодекс","민법전"
...
```

### Output Format

The script adds a new column `nllb_original` with Korean translations:

```csv
ru,ko,nllb_original
"Статья 1. Основные понятия","제1조. 기본 개념","제1조. 기본 개념"
...
```

## Model Information

- **Model**: facebook/nllb-200-distilled-600M
- **Parameters**: 600M
- **Languages**: 200 languages including Russian (rus_Cyrl) and Korean (kor_Hang)
- **Architecture**: Transformer encoder-decoder

## Translation Parameters

- **Source Language**: rus_Cyrl (Russian with Cyrillic script)
- **Target Language**: kor_Hang (Korean with Hangul script)
- **Max Length**: 256 tokens
- **Beam Size**: 5
- **Early Stopping**: True

## Performance

- **GPU (Tesla T4)**: ~10-15 minutes for 1,000 sentences
- **CPU**: ~1-2 hours for 1,000 sentences

## Citation

If you use this code, please cite:

```bibtex
@phdthesis{yourname2025,
  title={Russian-Korean Legal Domain Machine Translation with Domain-Specific Corpus},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

## License

MIT License

## Related Work

This script is part of a doctoral dissertation research on Russian-Korean legal domain machine translation.

**Related scripts:**
- `translate_nllb_original.py` - NLLB-200 original pretrained model (this script)
- `translate_nllb_finetuned.py` - NLLB-200 fine-tuned model
- `translate_mbart_finetuned.py` - mBART-50 fine-tuned model  
- `translate_google.py` - Google Translate baseline

## Contact

[Your Name]  
[Your Email]  
[Your Institution]
