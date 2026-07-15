"""
Train all models sequentially: City Type, Health Impact (ML + ANN), Air Quality (ML + LSTM).
Run this script once after placing datasets in data/ folder.

Usage:
    python train_all_models.py
"""

import sys
import os
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training_run.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ("City Type Classification",       "src.city_type_model",      "train_city_type_model"),
    ("Health Impact Classification",   "src.health_impact_model",  "train_health_impact_model"),
    ("Health Impact ANN (Deep Learning)", "src.health_impact_dl_model", "train_health_impact_ann"),
    ("Air Quality Regression",         "src.air_quality_model",    "train_air_quality_model"),
]


def main():
    print("=" * 65)
    print("  ENVIRONMENTAL IMPACT ASSESSMENT — MODEL TRAINING PIPELINE")
    print("=" * 65)
    print()

    from src.data_loader import validate_datasets
    status = validate_datasets()
    all_ok = True
    for name, info in status.items():
        if info["available"]:
            print(f"  [OK] Dataset '{name}': {info['shape']}")
        else:
            print(f"  [MISSING] Dataset '{name}': {info['error']}")
            all_ok = False

    if not all_ok:
        print("\n[ERROR] One or more datasets are missing. Place them in data/ and retry.")
        sys.exit(1)

    print()

    results = {}
    total_start = time.time()

    for i, (label, module_path, func_name) in enumerate(STEPS, 1):
        print(f"\n--- Step {i}/{len(STEPS)}: {label} ---")
        step_start = time.time()
        try:
            import importlib
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            func()
            elapsed = time.time() - step_start
            results[label] = {"status": "OK", "time": elapsed}
            print(f"  Completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - step_start
            results[label] = {"status": "FAILED", "error": str(e), "time": elapsed}
            logger.exception(f"Training failed for {label}")
            print(f"  [FAILED] {e}")

    total_time = time.time() - total_start

    print("\n" + "=" * 65)
    print("  TRAINING SUMMARY")
    print("=" * 65)
    for label, r in results.items():
        status_sym = "[OK]" if r["status"] == "OK" else "[FAILED]"
        print(f"  {status_sym} {label:<40} {r['time']:.1f}s")
    print(f"\n  Total time: {total_time:.1f}s")

    failed = [k for k, v in results.items() if v["status"] == "FAILED"]
    if failed:
        print(f"\n  [WARNING] {len(failed)} model(s) failed to train.")
        sys.exit(1)
    else:
        print("\n  All models trained successfully!")

    # ── Step 5: Health Impact Model Comparison & Report Generation ───────────
    print("\n--- Step 5/6: Health Impact Model Comparison & Report Generation ---")
    try:
        from src.model_comparison import generate_comparison
        cmp_start = time.time()
        comparison_result = generate_comparison()
        cmp_elapsed = time.time() - cmp_start
        best = comparison_result.get("best_model", "?")
        best_type = comparison_result.get("best_model_type", "?")
        print(f"  Completed in {cmp_elapsed:.1f}s")
        print(f"  Best model: {best} ({best_type})")
        print("  Reports saved to: reports/")
    except Exception as e:
        logger.exception("Health Impact comparison step failed (non-fatal)")
        print(f"  [WARNING] Comparison step failed: {e}")
        print("  You can run it manually: python src/model_comparison.py")

    # ── Step 6: Air Quality LSTM Training + Comparison ────────────────────────
    print("\n--- Step 6/6: Air Quality LSTM + Comparison ---")
    try:
        from src.air_quality_lstm_model import train_air_quality_lstm
        lstm_start = time.time()
        lstm_result = train_air_quality_lstm()
        lstm_elapsed = time.time() - lstm_start
        m = lstm_result.get("metrics", {})
        print(f"  LSTM trained in {lstm_elapsed:.1f}s")
        print(f"  LSTM RMSE={m.get('rmse', '?'):.4f} | R2={m.get('r2', '?'):.4f}")

        from src.air_quality_comparison import generate_comparison as aq_compare
        aq_cmp_result = aq_compare()
        aq_best = aq_cmp_result.get("best_model", "?")
        aq_best_type = aq_cmp_result.get("best_model_type", "?")
        print(f"  AQ best model: {aq_best} ({aq_best_type})")
        print("  AQ reports saved to: reports/")
    except Exception as e:
        logger.exception("Air Quality LSTM step failed (non-fatal)")
        print(f"  [WARNING] LSTM step failed: {e}")
        print("  You can run manually: python src/air_quality_lstm_model.py")

    print("\n  Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
