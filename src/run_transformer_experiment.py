"""
Experiment runner for Transformer Cross-Encoder Stance Classification on args.me.
Compares:
1. Traditional Best Baseline (Logistic Regression TF-IDF: 54.9%)
2. Zero-Shot NLI Cross-Encoder (cross-encoder/nli-deberta-v3-small)
3. Fine-Tuned DeBERTa-v3-small Cross-Encoder
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_DIR
from src.transformer_stance import (
    TransformerStanceCrossEncoder,
    ZeroShotNLIStanceClassifier,
    get_compute_device,
)


def run_transformer_benchmark():
    parquet_path = DATA_DIR / "args_me_benchmark_5k.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {parquet_path}")

    print(f"Loading benchmark dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df_clean = df[df["stance"].isin(["PRO", "CON"])].copy().reset_index(drop=True)

    print(f"Total dataset size: {len(df_clean)} records across {df_clean['conclusion'].nunique()} topics.")

    # We use a stratified split (80% train, 20% test) matching the traditional benchmark
    train_val_df, test_df = train_test_split(
        df_clean, test_size=0.20, random_state=42, stratify=df_clean["stance"]
    )
    # Further split 10% of train for validation / early stopping
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.10, random_state=42, stratify=train_val_df["stance"]
    )

    print(f"Train set: {len(train_df)} | Validation set: {len(val_df)} | Test set: {len(test_df)}")
    device = get_compute_device()
    print(f"Active Hardware Accelerator: {device}")

    results = []

    # 1. Zero-Shot Pre-Trained NLI Baseline
    print("\n" + "=" * 80)
    print("PHASE 1: ZERO-SHOT PRE-TRAINED NLI CROSS-ENCODER EVALUATION")
    print("=" * 80)
    try:
        zero_shot_clf = ZeroShotNLIStanceClassifier(
            model_name="cross-encoder/nli-deberta-v3-small",
            device=device,
            batch_size=16,
        )
        zs_metrics = zero_shot_clf.evaluate_df(
            test_df, benchmark_name="Zero-Shot NLI Cross-Encoder (Pre-Trained)"
        )
        print(f"Zero-Shot Accuracy: {zs_metrics.accuracy:.4f} | Macro F1: {zs_metrics.macro_f1:.4f} | ROC-AUC: {zs_metrics.roc_auc}")
        results.append(zs_metrics)
    except Exception as e:
        print(f"Zero-shot NLI evaluation encountered notice: {e}")

    # 2. Fine-Tuned RoBERTa-base Cross-Encoder
    print("\n" + "=" * 80)
    print("PHASE 2: FINE-TUNING ROBERTA-BASE CROSS-ENCODER (2 Epochs, lr=2e-5)")
    print("=" * 80)
    cross_encoder = TransformerStanceCrossEncoder(
        model_name="roberta-base",
        max_length=160,
        batch_size=16,
        learning_rate=2e-5,
        device=device,
    )

    # Train cross-encoder
    history = cross_encoder.fit(
        train_df=train_df,
        val_df=val_df,
        epochs=2,
        warmup_ratio=0.1,
    )

    # Evaluate on held-out test set
    ft_metrics = cross_encoder.evaluate_df(
        test_df, benchmark_name="Fine-Tuned DeBERTa-v3-small Cross-Encoder"
    )
    print(f"\nFine-Tuned Accuracy: {ft_metrics.accuracy:.4f} | Macro F1: {ft_metrics.macro_f1:.4f} | ROC-AUC: {ft_metrics.roc_auc}")
    results.append(ft_metrics)

    # Summary Comparison
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON TABLE (Test Set N = 1,000)")
    print("=" * 80)
    print(f"{'Model':<45} | {'Accuracy':<8} | {'Macro F1':<8} | {'CON F1':<8} | {'PRO F1':<8} | {'ROC-AUC':<8}")
    print("-" * 95)
    print(f"{'Majority Baseline (Zero-R)':<45} | 0.5020   | 0.3342   | 0.6684   | 0.0000   | 0.5000")
    print(f"{'VADER Sentiment Polarity':<45} | 0.5040   | 0.4988   | 0.4477   | 0.5499   | N/A")
    print(f"{'Logistic Regression (Premise TF-IDF)':<45} | 0.5490   | 0.5490   | 0.5449   | 0.5530   | 0.5580")
    for r in results:
        print(f"{r.model_name:<45} | {r.accuracy:<8.4f} | {r.macro_f1:<8.4f} | {r.con_f1:<8.4f} | {r.pro_f1:<8.4f} | {r.roc_auc if r.roc_auc is not None else 'N/A':<8}")

    return results


if __name__ == "__main__":
    run_transformer_benchmark()
