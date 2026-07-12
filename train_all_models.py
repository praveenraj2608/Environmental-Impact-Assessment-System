"""
Train all 3 models sequentially.
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
    ("City Type Classification", "src.city_type_model", "train_city_type_model"),
    ("Health Impact Classification", "src.health_impact_model", "train_health_impact_model"),
    ("Air Quality Regression", "src.air_quality_model", "train_air_quality_model"),
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
        print(f"\n--- Step {i}/3: {label} ---")
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
        print(f"  {status_sym} {label:<35} {r['time']:.1f}s")
    print(f"\n  Total time: {total_time:.1f}s")

    failed = [k for k, v in results.items() if v["status"] == "FAILED"]
    if failed:
        print(f"\n  [WARNING] {len(failed)} model(s) failed to train.")
        sys.exit(1)
    else:
        print("\n  All models trained successfully!")
        print("  Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
