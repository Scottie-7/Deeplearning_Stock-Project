import asyncio
import os
import sys
from preprocessing import run_ps5_pipeline


def print_ps5_verification_report():
    print("=" * 78)
    print("SONAR (Stock Project) - Stage 5 Refined Preprocessing & Model Comparison")
    print("=" * 78)
    res = run_ps5_pipeline()
    print(f"[1] Atomic L1 Parquet Saved : {res['l1_path']}")
    print(f"[2] Atomic L2 47-Col Saved  : {res['l2_path']}")
    print(f"[3] DeepLOB Spatial Tensor  : {res['x_deeplob_shape']} (Batch, Ch=1, T=100, W=40)")
    print(f"[4] Aux Flow Sequence Tensor: {res['x_aux_shape']}    (Batch, T=100, F=16)")
    print(f"[5] Tabular Feature Matrix  : {res['x_tab_shape']}       (Batch, 56 Features)")
    print(f"[6] Triple-Barrier Labels   : {res['class_counts']} (0=Down, 1=Neutral, 2=Up)")
    print("-" * 78)
    print(f"{'Model':<28} | {'Macro-F1':<9} | {'F1 Down':<9} | {'F1 Neut':<9} | {'F1 Up':<9}")
    print("-" * 78)
    for name, m in res["models"].items():
        print(f"{name:<28} | {m['macro_f1']:<9.4f} | {m['f1_down']:<9.4f} | {m['f1_neutral']:<9.4f} | {m['f1_up']:<9.4f}")
    print("-" * 78)
    print("[7] Random Forest Confusion Matrix (Rows=True [0,1,2], Cols=Pred [0,1,2]):")
    for row in res["models"]["Random Forest"]["cm"]:
        print(f"    {row}")
    print("[8] Late-Fusion MLP Confusion Matrix (Rows=True [0,1,2], Cols=Pred [0,1,2]):")
    for row in res["models"]["Late-Fusion MLP (Neural)"]["cm"]:
        print(f"    {row}")
    print("=" * 78)


if __name__ == "__main__":
    print_ps5_verification_report()

    if "--live" in sys.argv:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if os.getenv("TASTY_REFRESH_TOKEN"):
            from providers.tastytrade import TastytradeAdapter
            ticker = input("Enter ticker for live Tastytrade L1 stream (default IWM): ").strip() or "IWM"
            adapter = TastytradeAdapter()
            asyncio.run(adapter.stream_data(ticker))
        else:
            print("Set TASTY_REFRESH_TOKEN in src/.env to run --live.")
    else:
        print("Tip: Pass `--live` with a configured `src/.env` file to stream live Tastytrade quotes.")
