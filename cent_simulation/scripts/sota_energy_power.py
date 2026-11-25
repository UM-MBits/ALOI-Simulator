#!/usr/bin/env python3
"""Energy per token, power and tokens/J for the SOTA results (single and double buffering).

Single buffer: run_sim's per-component energy is rebuilt from the Ramulator log
(system_energy_breakdown._kv_head_energy) and must reproduce the CSV total.

Double buffering (analytical, like the latency estimate):
- per-operation (dynamic) energy is unchanged: the same commands still run;
- time-dependent energy -- all ``*_STT`` leakage and DRAM ``ACT_STBY`` /
  ``PRE_STBY`` -- is charged over PIM time in run_sim, so it is scaled by
  new PIM cycles / old PIM cycles;
- the second 2 KiB GB per channel adds one more ``GB_STT`` over the new PIM time.

Energy per token does not depend on the PCIe generation (CXL energy is per
bit); power = energy per token x throughput does.

Usage (from cent_simulation/):
  python scripts/sota_energy_power.py [ROOT ...]     # default: every output/sota_results/* root
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

CENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT))
sys.path.insert(0, str(CENT / "scripts"))
sys.path.insert(0, str(CENT / "scripts/utility"))
import pandas as pd  # noqa: E402
import run_sim as simulator  # noqa: E402
import system_energy_breakdown as seb  # noqa: E402

LAYERS = 80
TIME_DEPENDENT = ("ACT_STBY", "PRE_STBY")


def is_time_dependent(component):
    return component.endswith("_STT") or component in TIME_DEPENDENT


def variant_energy(variant_dir):
    """Per-component energy of one traced batch group (all TP ranks), and the CSV row."""
    csv_path = next(Path(variant_dir).glob("*/simulation_results.csv"))
    source = pd.read_csv(csv_path).iloc[0]
    log = next(csv_path.parent.joinpath("ramulator").rglob("*.txt.log"))
    simulator.set_channel_count(int(source["Channels per device"]), int(source["Parallel SRAM banks"]))
    stat = simulator.command_processor(str(log))
    energy = seb._kv_head_energy(
        source, stat, log, model=str(source["Model"]), tp=int(source["Tensor parallelism"]),
        batch_size=int(source["Batch size"]), seqlen=int(source["Sequence length"]),
        dram_impl="LPDDR4X", source_pp=int(source["Pipeline parallelism"]))
    per_token = sum(energy.values()) * LAYERS / int(source["Batch size"])
    if not math.isclose(per_token, float(source["Token energy (mJ)"]), rel_tol=1e-9):
        raise ValueError(f"Energy rebuild {per_token} != CSV {source['Token energy (mJ)']} for {variant_dir}")
    return energy, source


def double_buffer_energy(energy, pim_ratio):
    scaled = {k: (v * pim_ratio if is_time_dependent(k) else v) for k, v in energy.items()}
    scaled["GB_STT (second GB)"] = energy.get("GB_STT", 0.0) * pim_ratio
    return scaled


def summarize_root(root):
    root = Path(root)
    variants = {}
    for breakdown in sorted(root.glob("full*/breakdown.json")):
        block = json.loads(breakdown.read_text())["case"]["flash_attention_block_size"]
        variants[f"full_flash{block}"] = breakdown.parent
    rows = []
    for label, variant_dir in variants.items():
        energy, source = variant_energy(variant_dir)
        batch, devices = int(source["Batch size"]), int(source["Device number"])
        for gen in (5, 7):
            db_dir = root / f"double buffering pcie{gen}"
            analysis = json.loads((db_dir / label / "analysis.json").read_text())
            totals = analysis["models"]["dependency_aware"]["totals"]
            ratio = totals["new_pim_cycles_analytical"] / totals["old_pim_cycles_measured"]
            throughput = {r["scenario"]: r for r in csv.DictReader(open(db_dir / "throughput/comparison.csv"))
                          if r["case"] == label}
            for scenario, components in (("single_buffer", energy),
                                         ("dependency_aware", double_buffer_energy(energy, ratio))):
                mj = sum(components.values()) * LAYERS / batch
                t = throughput[scenario]
                system_tps = float(t["system_tokens_per_s"])
                rows.append({
                    "case": root.name, "flash_block": label.removeprefix("full_flash"),
                    "devices": devices, "tp": int(source["Tensor parallelism"]),
                    "pp": int(source["Pipeline parallelism"]), "pcie": gen, "scenario": scenario,
                    "block_us": float(t["block_us"]), "system_tokens_per_s": system_tps,
                    "tokens_per_s_per_device": float(t["tokens_per_s_per_device"]),
                    "energy_mj_per_token": mj, "tokens_per_joule": 1000.0 / mj,
                    "system_power_w": mj * system_tps / 1000.0,
                    "power_per_device_w": mj * system_tps / 1000.0 / devices,
                    "time_dependent_energy_share_pct":
                        100 * sum(v for k, v in components.items() if is_time_dependent(k) or k.startswith("GB_STT"))
                        / sum(components.values()),
                })
    if rows:
        with (root / "energy_power.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path)
    args = parser.parse_args(argv)
    roots = args.roots or sorted(p for p in (CENT / "output/sota_results").iterdir() if p.is_dir())
    print("case                                          block PCIe scenario          mJ/token  tokens/J  "
          "tok/s/dev  W/device  system W")
    for root in roots:
        for r in summarize_root(root):
            print(f"{r['case']:<45} {r['flash_block']:>5} {r['pcie']:>4} {r['scenario']:<17} "
                  f"{r['energy_mj_per_token']:>8.2f} {r['tokens_per_joule']:>9.3f} {r['tokens_per_s_per_device']:>10.2f} "
                  f"{r['power_per_device_w']:>9.2f} {r['system_power_w']:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
