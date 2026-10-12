"""
CLI Runner to evaluate Advanced Multi-Question DSPy & Tev1 Pipeline against baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_DIR
from src.dspy_stance import Tev1MultiQuestionPipeline


def run_advanced_dspy_benchmark(sample_size: int = 200):
    parquet_path = DATA_DIR / "args_me_benchmark_5k.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {parquet_path}")

    print(f"Loading benchmark dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df_clean = df[df["stance"].isin(["PRO", "CON"])].copy().reset_index(drop=True)

    # Use exact same test split
    _, test_df = train_test_split(
        df_clean, test_size=0.20, random_state=42, stratify=df_clean["stance"]
    )

    print(f"Held-out test set size: {len(test_df)}")
    print(f"Evaluating Advanced Multi-Question DSPy Pipeline on N = {sample_size} arguments...")

    pipeline = Tev1MultiQuestionPipeline(model_name="tev1:4b")
    metrics, results_df = pipeline.evaluate_df(test_df, sample_size=sample_size, random_state=42)

    print("\n" + "=" * 80)
    print("ADVANCED MULTI-QUESTION DSPY / TEV1 PIPELINE RESULTS")
    print("=" * 80)
    print(f"Model:           {metrics.model_name}")
    print(f"Accuracy:        {metrics.accuracy:.4f} ({metrics.accuracy * 100:.2f}%)")
    print(f"Macro F1:        {metrics.macro_f1:.4f}")
    print(f"Macro Precision: {metrics.macro_precision:.4f}")
    print(f"Macro Recall:    {metrics.macro_recall:.4f}")
    print(f"CON F1:          {metrics.con_f1:.4f} | PRO F1: {metrics.pro_f1:.4f}")
    print(f"ROC-AUC:         {metrics.roc_auc:.4f}")
    print(f"Confusion Matrix (CON, PRO): {metrics.confusion_mat}")
    print(f"Average Latency: {results_df['latency_ms'].mean():.1f} ms")

    # Diagnostic statistics
    chatter_count = (results_df["is_substantive"] == "chatter").sum()
    quotes_count = results_df["quotes_opponent"].sum()
    print(f"\nDialectic Diagnostic Insights:")
    print(f"  - Detected Non-Substantive Chatter/Forfeit: {chatter_count}/{len(results_df)} ({chatter_count / len(results_df) * 100:.1f}%)")
    print(f"  - Detected Adversary Quoting in Premises:    {quotes_count}/{len(results_df)} ({quotes_count / len(results_df) * 100:.1f}%)")

    # Paradigm comparison
    print("\n" + "=" * 80)
    print("COMPLETE ARCHITECTURAL PARADIGM COMPARISON")
    print("=" * 80)
    print(f"{'Paradigm / Model':<48} | {'Accuracy':<8} | {'Macro F1':<8} | {'ROC-AUC':<8}")
    print("-" * 80)
    print(f"{'Majority Baseline (Zero-R)':<48} | 0.5020   | 0.3342   | 0.5000")
    print(f"{'VADER Sentiment Polarity':<48} | 0.5040   | 0.4988   | N/A")
    print(f"{'Linear SVM (Contextual TF-IDF)':<48} | 0.5070   | 0.5067   | 0.5244")
    print(f"{'Logistic Regression (Premise TF-IDF)':<48} | 0.5490   | 0.5490   | 0.5580")
    print(f"{'Zero-Shot NLI DeBERTa-v3 Small':<48} | 0.5230   | 0.4936   | 0.5541")
    print(f"{'Fine-Tuned RoBERTa-base Cross-Encoder':<48} | 0.5890   | 0.5868   | 0.6265")
    print(f"{'Standard Tev1:4B Single-Question':<48} | 0.7500   | 0.7500   | 0.8214")
    print(f"{metrics.model_name:<48} | {metrics.accuracy:<8.4f} | {metrics.macro_f1:<8.4f} | {metrics.roc_auc:<8.4f}")

    return metrics, results_df


if __name__ == "__main__":
    run_advanced_dspy_benchmark(sample_size=200)
