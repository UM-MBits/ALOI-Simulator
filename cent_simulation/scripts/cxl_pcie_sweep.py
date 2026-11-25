#!/usr/bin/env python3
"""Exposed compute/communication per PCIe generation with an all-to-all CXL model.

Rows per PCIe generation:
  Single Layer         1 layer, no PP handoff
  Two Pipeline Stages  1 layer on each of 2 stages, 1 handoff
  <model>              --layers layers over --pp-stages stages, (pp-1) handoffs

Communication uses cxl_latency (one-shot all-to-all TP all-reduce, no multicast
discount, FP8 wire, 180 ns fixed per transfer).  Compute is
not overlapped with communication.

Reproduce the reference spreadsheet (128 layers, 8 stages, raw bytes, GiB/s,
140 ns TP / 70 ns PP fixed overhead):
  python scripts/cxl_pcie_sweep.py --layers 128 --pp-stages 8 --no-flit-packing --gib --tp-fixed-ns 140 --pp-fixed-ns 70
"""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxl_latency import BYTES_PER_GB, PCIE_GB_PER_LANE, PP_FIXED_NS, TP_FIXED_NS, pipeline_comm_us  # noqa: E402

FIGURE_COMPUTE_PER_LAYER_US = 267.937


def sweep(compute_per_layer_us, layers, pp_stages, hidden, batch, tp, lanes, gens,
          bytes_per_value, flit_packing, model_label, bytes_per_bandwidth_unit=BYTES_PER_GB,
          tp_fixed_ns=TP_FIXED_NS, pp_fixed_ns=PP_FIXED_NS):
    rows = []
    values = hidden * batch
    for gen in gens:
        bw = lanes * PCIE_GB_PER_LANE[gen]
        for trace, n_layers, n_stages in (("Single Layer", 1, 1), ("Two Pipeline Stages", 2, 2),
                                          (model_label, layers, pp_stages)):
            tp_us, pp_us = pipeline_comm_us(n_layers, n_stages, values, bw, tp=tp,
                                            bytes_per_value=bytes_per_value, flit_packing=flit_packing,
                                            bytes_per_bandwidth_unit=bytes_per_bandwidth_unit,
                                            tp_fixed_ns=tp_fixed_ns, pp_fixed_ns=pp_fixed_ns)
            compute = n_layers * compute_per_layer_us
            rows.append({"PCIe Version": gen, "CXL Bandwidth Per Device (GB/s)": bw, "Trace": trace,
                         "Layers": n_layers, "PP Stages": n_stages,
                         "Exposed Compute (us)": round(compute, 3),
                         "Exposed Communication (us)": round(tp_us + pp_us, 3),
                         "TP All-Reduce (us)": round(tp_us, 3), "PP Handoff (us)": round(pp_us, 3),
                         "Comm Share (%)": round(100 * (tp_us + pp_us) / (compute + tp_us + pp_us), 3)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compute-per-layer-us", type=float, default=FIGURE_COMPUTE_PER_LAYER_US)
    p.add_argument("--layers", type=int, default=80, help="Llama3.1-70B has 80 decoder layers")
    p.add_argument("--pp-stages", type=int, default=8)
    p.add_argument("--model-label", default="Llama3.1 70B")
    p.add_argument("--hidden", type=int, default=8192)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--lanes-per-device", type=int, default=4)
    p.add_argument("--pcie-gens", type=int, nargs="+", choices=sorted(PCIE_GB_PER_LANE), default=[5, 6, 7])
    p.add_argument("--bytes-per-value", type=int, default=1, help="1 = FP8 on the wire, 2 = BF16")
    p.add_argument("--no-flit-packing", action="store_true", help="drop the 256/192 flit overhead")
    p.add_argument("--gib", action="store_true", help="treat link bandwidth as GiB/s instead of GB/s")
    p.add_argument("--tp-fixed-ns", type=float, default=TP_FIXED_NS)
    p.add_argument("--pp-fixed-ns", type=float, default=PP_FIXED_NS)
    p.add_argument("--csv", type=Path)
    args = p.parse_args(argv)
    if not 1 <= args.pp_stages <= args.layers:
        p.error("--pp-stages must be between 1 and --layers")
    rows = sweep(args.compute_per_layer_us, args.layers, args.pp_stages, args.hidden, args.batch, args.tp,
                 args.lanes_per_device, args.pcie_gens, args.bytes_per_value, not args.no_flit_packing,
                 args.model_label, 2**30 if args.gib else BYTES_PER_GB, args.tp_fixed_ns, args.pp_fixed_ns)
    cols = list(rows[0])
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
