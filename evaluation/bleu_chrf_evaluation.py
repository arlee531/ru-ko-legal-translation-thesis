"""
BLEU & ChrF Score Calculation for Translation Models
Description: Calculates BLEU and chrF scores for Russian-Korean legal translation models
"""

import pandas as pd
import sacrebleu
from typing import List, Dict, Tuple
import json
from datetime import datetime


def calculate_scores(
    references: List[str], 
    hypotheses: List[str], 
    model_name: str
) -> Dict:
    """
    Calculate BLEU and ChrF scores for translation pairs
    
    Args:
        references: List of reference translations
        hypotheses: List of hypothesis (model) translations
        model_name: Name of the translation model
        
    Returns:
        Dictionary containing evaluation results
    """
    # Remove empty strings and convert to string
    valid_pairs = []
    for ref, hyp in zip(references, hypotheses):
        if pd.notna(ref) and pd.notna(hyp) and str(ref).strip() and str(hyp).strip():
            valid_pairs.append((str(ref).strip(), str(hyp).strip()))
    
    if not valid_pairs:
        print(f"⚠ {model_name}: No valid translation pairs found!")
        return None
    
    refs = [pair[0] for pair in valid_pairs]
    hyps = [pair[1] for pair in valid_pairs]
    
    print(f"\n{model_name}:")
    print(f"  Valid pairs: {len(valid_pairs)} / {len(references)}")
    
    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    
    # Calculate ChrF score
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    
    results = {
        'model': model_name,
        'total_samples': len(references),
        'valid_samples': len(valid_pairs),
        'bleu': round(bleu.score, 2),
        'chrf': round(chrf.score, 2)
    }
    
    print(f"  BLEU: {results['bleu']}")
    print(f"  ChrF: {results['chrf']}")
    
    return results


def load_translation_files(file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load translation CSV files
    
    Args:
        file_paths: Dictionary mapping model names to file paths
        
    Returns:
        Dictionary of DataFrames
    """
    dataframes = {}
    
    for model_key, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        
        print(f"\nLoading {model_key}:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Find translation column
        translation_col = None
        for col in df.columns:
            if 'translation' in col.lower() or col not in ['id', 'ru', 'ko']:
                translation_col = col
                break
        
        if translation_col:
            df = df.rename(columns={translation_col: 'translation'})
            print(f"  Translation column: '{translation_col}'")
        else:
            print(f"  ⚠ Warning: Translation column not found!")
        
        # Check required columns
        required_cols = ['id', 'ru', 'ko', 'translation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ✗ Missing columns: {missing_cols}")
        else:
            print(f"  ✓ All required columns present")
            dataframes[model_key] = df
    
    return dataframes


def save_results(
    results_df: pd.DataFrame, 
    output_dir: str = './'
) -> Tuple[str, str, str]:
    """
    Save evaluation results in multiple formats
    
    Args:
        results_df: DataFrame containing evaluation results
        output_dir: Output directory path
        
    Returns:
        Tuple of (csv_path, json_path, report_path)
    """
    # Save CSV
    csv_path = f"{output_dir}evaluation_results.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ CSV saved: {csv_path}")
    
    # Save JSON
    json_path = f"{output_dir}evaluation_results.json"
    results_df.to_json(json_path, orient='records', indent=2, force_ascii=False)
    print(f"✓ JSON saved: {json_path}")
    
    # Generate text report
    report = f"""
{'='*70}
Translation Model Evaluation Results
{'='*70}

Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Test Samples: {results_df.iloc[0]['total_samples']:,}

{'='*70}
BLEU & ChrF Scores
{'='*70}

"""
    
    for _, row in results_df.iterrows():
        report += f"""
{row['model']}
{'-'*70}
Valid Samples:  {row['valid_samples']:,} / {row['total_samples']:,}
BLEU Score:     {row['bleu']:.2f}
ChrF Score:     {row['chrf']:.2f}

"""
    
    report += f"""
{'='*70}
Ranking by BLEU Score
{'='*70}
"""
    
    for rank, (_, row) in enumerate(results_df.iterrows(), 1):
        report += f"{rank}. {row['model']}: {row['bleu']:.2f}\n"
    
    report += f"""
{'='*70}
Notes:
- BLEU: Bilingual Evaluation Understudy (0-100, higher is better)
- ChrF: Character n-gram F-score (0-100, higher is better)
- Calculated using sacrebleu library
{'='*70}
"""
    
    report_path = f"{output_dir}evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Report saved: {report_path}")
    print(report)
    
    return csv_path, json_path, report_path


def main():
    """Main execution function"""
    
    print("="*70)
    print("BLEU & ChrF Score Calculation")
    print("="*70)
    
    # Define file paths (modify these paths according to your setup)
    file_paths = {
        'Google Translate': 'google_translate.csv',
        'NLLB Original': 'nllb_original.csv',
        'NLLB Fine-tuned': 'nllb_finetuned.csv',
        'mBART Original': 'mbart_original.csv',
        'mBART Fine-tuned': 'mbart_finetuned.csv'
    }
    
    # Load translation files
    print(f"\n{'='*70}")
    print("Loading Files")
    print(f"{'='*70}")
    dataframes = load_translation_files(file_paths)
    
    # Calculate scores for all models
    print(f"\n{'='*70}")
    print("Calculating Scores")
    print(f"{'='*70}")
    
    all_results = []
    for model_name, df in dataframes.items():
        references = df['ko'].tolist()
        hypotheses = df['translation'].tolist()
        
        results = calculate_scores(references, hypotheses, model_name)
        if results:
            all_results.append(results)
    
    # Create results DataFrame
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('bleu', ascending=False).reset_index(drop=True)
    
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")
    
    csv_path, json_path, report_path = save_results(results_df)
    
    # Final message
    print(f"\n{'='*70}")
    print("SUCCESS! 🎉")
    print(f"{'='*70}")
    print(f"""
Results saved:
1. {csv_path}
2. {json_path}
3. {report_path}

You can now:
- Compare BLEU/ChrF scores across models
- Proceed with further evaluation
- Use results for publication
""")


if __name__ == "__main__":
    main()
