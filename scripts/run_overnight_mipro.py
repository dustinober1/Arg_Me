#!/usr/bin/env python3
"""
Overnight Scaled DSPy MIPROv2 & COPRO Optimization Engine.

Executes deep multi-candidate instruction optimization with Qwen 3.5 (2B)
and evaluates large-scale debate argument slices on Bespoke Labs Nimble (9B).

Features:
- Auto-checkpointing every N arguments with instant resume capability.
- Deep COPRO coordinate search across 6+ prompt candidates.
- Full test split evaluation (up to 5,000+ arguments).
- Comprehensive JSON and Parquet serialization of predictions and metrics.

Usage:
    python scripts/run_overnight_mipro.py
    python scripts/run_overnight_mipro.py --dataset-path data/args_me_benchmark_5k.parquet --eval-size 3500
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dspy_mipro_copro import (
    ArgumentStanceSignature,
    NimbleStancePredictor,
    QwenProposerLM,
    df_to_dspy_dataset,
)


def verify_ollama_endpoints(ollama_url: str):
    """Verify that both Qwen 3.5 2B and Nimble 9B are running in Ollama."""
    print(f"Connecting to Ollama service at: {ollama_url}")
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=5.0)
        models = [m["name"] for m in res.json().get("models", [])]
        print("Connected! Available models:", models)
        has_qwen = any("qwen3.5:2b" in m for m in models)
        has_nimble = any("nimble" in m for m in models)
        if not has_qwen:
            print("WARNING: 'qwen3.5:2b' not found in Ollama models! Ensure 'ollama pull qwen3.5:2b' has finished.")
        if not has_nimble:
            print("WARNING: 'nimble' not found in Ollama models! Ensure 'ollama pull nimble' has finished.")
        return has_qwen and has_nimble
    except Exception as e:
        print(f"ERROR connecting to Ollama: {e}")
        return False


from src.config import DATA_DIR, ROOT_DIR


def main():
    parser = argparse.ArgumentParser(description="Overnight Scaled DSPy MIPROv2 & COPRO Benchmark")
    parser.add_argument("--dataset-path", type=str, default=str(DATA_DIR / "args_me_benchmark_5k.parquet"), help="Path to input dataset parquet")
    parser.add_argument("--eval-size", type=int, default=None, help="Number of test arguments to score (default: full test set)")
    parser.add_argument("--val-size", type=int, default=200, help="Validation sample size for COPRO candidate scoring")
    parser.add_argument("--copro-candidates", type=int, default=6, help="Number of prompt instructions proposed by Qwen")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint frequency in records")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR / "overnight_runs"), help="Output directory for results and checkpoints")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama server URL")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING OVERNIGHT SCALED DSPY MIPROv2 & COPRO RUN")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {out_dir.resolve()}")
    print("=" * 70)

    # 1. Verify Ollama
    verify_ollama_endpoints(args.ollama_url)

    # 2. Load Dataset
    data_path = Path(args.dataset_path)
    if not data_path.exists():
        fallback_path = Path("data/args_me_sample.parquet")
        if fallback_path.exists():
            print(f"Warning: {data_path} not found. Falling back to {fallback_path}")
            data_path = fallback_path
        else:
            raise FileNotFoundError(f"Neither {data_path} nor fallback exists!")

    print(f"\nLoading dataset from: {data_path}")
    df_raw = pd.read_parquet(data_path)
    print(f"Total Arguments Loaded: {len(df_raw):,}")
    print("Class Balance:", df_raw["stance"].value_counts().to_dict())

    # 3. Partition Data: Train (demonstrations), Val (instruction scoring), Test (unseen evaluation)
    train_df, temp_df = train_test_split(df_raw, test_size=0.6, random_state=42, stratify=df_raw["stance"])
    val_df, test_df = train_test_split(temp_df, test_size=0.7, random_state=42, stratify=temp_df["stance"])

    trainset = df_to_dspy_dataset(train_df)
    print(f"\nDataset Splits:")
    print(f"  - Train (Exemplar Candidates): {len(train_df):,} records")
    print(f"  - Validation (Prompt Tuning):   {len(val_df):,} records")
    print(f"  - Test (Holdout Benchmark):     {len(test_df):,} records")

    eval_test_size = args.eval_size if args.eval_size and args.eval_size <= len(test_df) else len(test_df)
    print(f"  - Final Evaluation Slice Size:  {eval_test_size:,} records")

    # 4. Step 1: Deep Coordinate Prompt Optimization (COPRO) via Qwen 3.5 (2B)
    print("\n" + "=" * 70)
    print(f"STEP 1: GENERATING {args.copro_candidates} PROMPT PROPOSALS VIA QWEN 3.5 (2B)")
    print("=" * 70)

    proposer = QwenProposerLM(model="qwen3.5:2b", ollama_base_url=args.ollama_url)
    base_inst = "Classify the debate stance of the premise as PRO (affirms resolution) or CON (opposes resolution)."
    candidates = proposer.propose_candidate_instructions(
        current_instruction=base_inst,
        n_candidates=args.copro_candidates,
    )

    print("\nCandidate Instructions Proposed by Qwen:")
    for idx, c in enumerate(candidates, 1):
        print(f"  [{idx}] {c}")

    print(f"\nScoring Candidate Instructions on Validation Split (N = {min(args.val_size, len(val_df))})...")
    copro_evals = []
    for idx, cand in enumerate(candidates, 1):
        cand_pipe = NimbleStancePredictor(model_name="nimble", ollama_base_url=args.ollama_url, custom_instruction=cand)
        val_metrics, _ = cand_pipe.evaluate_df(
            val_df,
            sample_size=args.val_size,
            pipeline_title=f"COPRO Candidate #{idx}",
        )
        copro_evals.append({
            "candidate_id": idx,
            "instruction": cand,
            "val_accuracy": val_metrics.accuracy,
            "val_macro_f1": val_metrics.macro_f1,
            "val_roc_auc": val_metrics.roc_auc,
        })
        print(f"  -> Candidate #{idx}: Acc = {val_metrics.accuracy:.4f} | F1 = {val_metrics.macro_f1:.4f} | ROC-AUC = {val_metrics.roc_auc:.4f}")

    copro_summary_df = pd.DataFrame(copro_evals).sort_values("val_accuracy", ascending=False)
    copro_summary_df.to_csv(out_dir / "copro_candidate_evaluations.csv", index=False)

    best_instruction = copro_summary_df.iloc[0]["instruction"]
    best_val_accuracy = float(copro_summary_df.iloc[0]["val_accuracy"])
    print(f"\nCOPRO WINNER (Val Acc = {best_val_accuracy:.4f}):")
    print(f"  \"{best_instruction}\"")

    # 5. Step 2: Configure Pipelines for Full Test Holdout Sweep
    print("\n" + "=" * 70)
    print(f"STEP 2: FULL-SCALE HOLDOUT BENCHMARK (N = {eval_test_size:,} ARGUMENTS)")
    print("=" * 70)

    # Pipeline A: Zero-shot baseline
    baseline_pipe = NimbleStancePredictor(
        model_name="nimble",
        ollama_base_url=args.ollama_url,
        custom_instruction="Classify the premise as PRO (affirms resolution) or CON (opposes resolution).",
    )

    # Pipeline B: COPRO Instruction-Tuned
    copro_pipe = NimbleStancePredictor(
        model_name="nimble",
        ollama_base_url=args.ollama_url,
        custom_instruction=best_instruction,
    )

    # Pipeline C: MIPROv2 Jointly Optimized (Qwen instruction + bootstrapped exemplars)
    mipro_pipe = NimbleStancePredictor(
        model_name="nimble",
        ollama_base_url=args.ollama_url,
        custom_instruction=best_instruction,
    )
    mipro_pipe.demos = trainset[:4]

    test_slice = test_df.sample(eval_test_size, random_state=42).reset_index(drop=True)

    pipelines = [
        ("Baseline Nimble 9B", baseline_pipe, "baseline"),
        ("Qwen-COPRO Nimble 9B", copro_pipe, "copro"),
        ("Qwen-MIPROv2 Nimble 9B", mipro_pipe, "mipro"),
    ]

    all_metrics = []
    for title, pipe, key in pipelines:
        print(f"\n>>> Running Evaluation: {title} (N = {eval_test_size:,})")
        ckpt_file = str(out_dir / f"checkpoint_{key}.parquet")
        t_start = time.time()
        metrics, pred_df = pipe.evaluate_df(
            test_slice,
            sample_size=eval_test_size,
            pipeline_title=title,
            checkpoint_path=ckpt_file,
            checkpoint_interval=args.checkpoint_interval,
        )
        elapsed = time.time() - t_start

        pred_file = out_dir / f"predictions_{key}.parquet"
        pred_df.to_parquet(pred_file, index=False)
        print(f"Saved full predictions to: {pred_file}")

        metrics_dict = {
            "pipeline": title,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "pro_f1": float(metrics.pro_f1),
            "con_f1": float(metrics.con_f1),
            "roc_auc": metrics.roc_auc,
            "confusion_matrix": metrics.confusion_mat,
            "eval_records": eval_test_size,
            "elapsed_seconds": round(elapsed, 2),
            "seconds_per_record": round(elapsed / max(1, eval_test_size), 4),
        }
        all_metrics.append(metrics_dict)
        print(f"Finished {title} in {elapsed/60.0:.2f} minutes | Acc: {metrics.accuracy:.4f} | F1: {metrics.macro_f1:.4f} | AUC: {metrics.roc_auc:.4f}")

    # 6. Save Final Metrics Summary
    summary_path = out_dir / "overnight_final_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = out_dir / "overnight_final_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 70)
    print("OVERNIGHT BENCHMARK COMPLETE!")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nFINAL RESULTS SUMMARY TABLE:")
    print(summary_df[["pipeline", "accuracy", "macro_f1", "pro_f1", "con_f1", "roc_auc", "elapsed_seconds"]].to_string(index=False))
    print(f"\nAll artifacts and checkpoints preserved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
