"""Read copied episode metrics only; never train, evaluate or rewrite artifacts."""
from pathlib import Path
import csv
from statistics import mean

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "原样成果包结构/08_惯量域随机化与泛化边界研究/03_测试数据与配对结果/episode_metrics.csv"

def main():
    with DATA.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for axis in ("single", "three"):
        for controller in ("PID_N", "PID_R", "SAC_N", "SAC_R"):
            group = [r for r in rows if r["axis"] == axis and r["controller"] == controller
                     and r["experiment"] == "scalar" and r["domain"] == "out_of_domain"]
            good = [r for r in group if r["success"].lower() == "true"]
            print(f"{axis:6s} {controller:5s}: {len(good)}/{len(group)} = "
                  f"{len(good)/len(group):.1%}; successful settling="
                  f"{mean(float(r['settling_time_s']) for r in good):.4f}s; "
                  f"all-episode RMS={mean(float(r['rms_error_deg']) for r in group):.4f}deg; "
                  f"all-episode cost={mean(float(r['control_cost']) for r in group):.6f}")
    print("SAC pools three training seeds over reused initial states; these are not independent initial states.")

if __name__ == "__main__":
    main()
