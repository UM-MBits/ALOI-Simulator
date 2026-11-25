#!/usr/bin/env python3
"""Reproduce or retain the FP8 TP2/batch4 LPDDR4 nCCD2 SOTA results (full@2048 and controls)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

CENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT))
from cycle_breakdown import analyze_case, read_case, render_summary, write_kernel_table, write_report  # noqa: E402
from cxl_latency import DEFAULT_PCIE_GEN, PCIE_GB_PER_LANE  # noqa: E402

VARIANTS = {
    "baseline": [],
    "parallel": ["--channel-concurrency"],
    "flash": ["--flash-attention"],
    # FlashAttention block: --flash-block, or the case default (default_flash_block).
    "full": ["--channel-concurrency", "--flash-attention", "--pipelined-softmax"],
    # Fixed-block controls.
    "full_flash2048": ["--channel-concurrency", "--flash-attention",
                       "--flash-attention-block-size", "2048", "--pipelined-softmax"],
    "full_flash4096": ["--channel-concurrency", "--flash-attention",
                       "--flash-attention-block-size", "4096", "--pipelined-softmax"],
    "full_flash1024": ["--channel-concurrency", "--flash-attention",
                       "--flash-attention-block-size", "1024", "--pipelined-softmax"],
}
# SOTA cases: 8K TP1xPP8, 32K TP2xPP8 (this default), 64K TP4xPP4; all batch 4.
DEFAULT_CASE = {"tp": 2, "pp": 8, "context_window": 32768, "batch": 4}
DECODE_TOKENS = 3584  # active length = window - DECODE_TOKENS / 2 (long-context midpoint)
LLAMA2_70B_KV_HEADS = 8
DEVICE_BANKS = 32 * 8
QK_LANES = 32  # FP8 elements per 256-bit word


def default_flash_block(tp):
    """Smallest block (>= 2048) in which one QK wave fills all 32 lanes of every bank.

    A local KV head spans DEVICE_BANKS / local_kv_heads banks, each scoring 32
    tokens per wave: TP1 -> 1024 (so 2048), TP2 -> 2048, TP4 -> 4096.
    """
    local_kv_heads = LLAMA2_70B_KV_HEADS // tp
    return max(2048, DEVICE_BANKS // local_kv_heads * QK_LANES)


def default_root_name(tp, pp, context_window, batch, flash_block):
    return (f"LPDDR4_nCCD2_TP{tp}_PP{pp}_{context_window // 1024}K_batch{batch}"
            f"_flash{flash_block}_pnm_residual")


LPDDR4_ORGANIZATION = {"channels": 32, "channel_width_bits": 16, "dq_bits": 16,
                      "prefetch": 16, "transaction_bytes": 32}


def publish_interface_correction(root, reports):
    """Keep the superseded interface comparison separate from current results."""
    reference_path = root / "interface_correction_reference.json"
    if not reference_path.exists():
        return []
    reference = json.loads(reference_path.read_text())
    lines = ["", "LPDDR4 interface correction: previous width32/prefetch8/nBL4 -> x16/prefetch16/nBL8.",
             "Before/after use identical host traces, mapping, precision, and optimization settings.",
             "PNM/CXL and Softmax overlap remain analytical; internal MAC nCCD stays 2.",
             "", "variant       old PIM cycles  new PIM cycles  PIM change    old Block(us)  new Block(us)  Block change"]
    totals, kernels, kernel_lines = [], [], []
    for variant, report in reports.items():
        if variant not in reference["reports"]:
            continue
        if report["case"].get("dram_organization") != LPDDR4_ORGANIZATION or report["case"].get("nBL") != 8:
            raise ValueError(f"Cannot label {variant} as corrected x16 LPDDR4")
        old = reference["reports"][variant]
        if old["run_identity"]["artifacts"]["trace"]["sha256"] != report["run_identity"]["artifacts"]["trace"]["sha256"]:
            raise ValueError(f"Interface comparison changed {variant} host trace")
        before, after = old["full_run_totals"], report["full_run_totals"]
        item = {"variant": variant, "old_run_id": old["run_id"], "new_run_id": report["run_id"]}
        for metric, old_value, new_value in (
            ("pim_cycles", before["pim_native_cycles"], after["pim_native_cycles"]),
            ("block_us", before["block_latency_ns"] / 1000, after["block_latency_ns"] / 1000),
            ("throughput_tokens_s", float(old["run_identity"]["source_row"]["Throughput (tokens/s)"]),
             float(report["run_identity"]["source_row"]["Throughput (tokens/s)"])),
            ("token_energy_mj", float(old["run_identity"]["source_row"]["Token energy (mJ)"]),
             float(report["run_identity"]["source_row"]["Token energy (mJ)"])),
        ):
            item.update({f"old_{metric}": old_value, f"new_{metric}": new_value,
                         f"{metric}_change_percent": (new_value / old_value - 1) * 100})
        totals.append(item)
        lines.append(f"{variant:<12} {item['old_pim_cycles']:>14,} {item['new_pim_cycles']:>14,} "
                     f"{item['pim_cycles_change_percent']:>10.2f}% {item['old_block_us']:>16.6f} "
                     f"{item['new_block_us']:>14.6f} {item['block_us_change_percent']:>12.2f}%")
        previous = {r["region_id"]: r for r in old["regions"]}
        kernel_lines.extend(["", f"{variant}: per-kernel interface correction (PIM cycles are issued-gap estimates)",
                      "kernel                             old PIM      new PIM      delta PIM    old total(us)  new total(us)"])
        for region in report["regions"]:
            prior = previous[region["region_id"]]
            entry = {"variant": variant, "region_id": region["region_id"], "kernel": region["display_name"]}
            for label, source in (("old", prior), ("new", region)):
                entry[f"{label}_pim_cycles"] = source["backends"]["pim"]["reference_equivalent_cycles"]
                entry[f"{label}_total_us"] = source["legacy_ledger_duration_ns"] / 1000
                commands = {c["command"]: c for c in (source.get("pim_issued_gap_details") or {}).get("commands", [])}
                for command in ("MAC8", "WRGB"):
                    entry[f"{label}_{command}_cycles"] = commands.get(command, {}).get("gap_cycles", 0)
                    entry[f"{label}_{command}_count"] = commands.get(command, {}).get("issued_event_count", 0)
            if entry["old_pim_cycles"] is not None and entry["new_pim_cycles"] is not None:
                entry["delta_pim_cycles"] = entry["new_pim_cycles"] - entry["old_pim_cycles"]
                kernel_lines.append(f"{entry['kernel']:<32} {entry['old_pim_cycles']:>11,.0f} {entry['new_pim_cycles']:>12,.0f} "
                             f"{entry['delta_pim_cycles']:>14,.0f} {entry['old_total_us']:>15.6f} {entry['new_total_us']:>14.6f}")
            else:
                entry["delta_pim_cycles"] = None
                kernel_lines.append(f"{entry['kernel']}: independent PIM timing unmodeled/unavailable.")
            kernels.append(entry)
    for filename, rows in (("interface_correction_comparison.csv", totals), ("kernel_interface_correction.csv", kernels)):
        if rows:
            with (root / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if all(variant in reports for variant in ("baseline", "full")):
        baseline = reports["baseline"]["full_run_totals"]["block_latency_ns"]
        full = reports["full"]["full_run_totals"]["block_latency_ns"]
        lines.extend(["", f"Corrected full vs baseline: block latency reduction={(1 - full / baseline) * 100:.2f}%; "
                      f"throughput ratio={baseline / full:.4f}x.",
                      "The slower external bus increases GB/register I/O. Baseline score transfer remains limited by serialized host admission.",
                      "MAC/WRGB work counts are unchanged; control and refresh timing can shift."])
    return lines + kernel_lines


def publish_comparison(root, comparison, commands):
    """Publish an index plus complete kernel details at stable result paths."""
    root = Path(root)
    reports = {}
    for item in comparison:
        report_path = Path(item["summary"]).with_name("breakdown.json").resolve()
        report = json.loads(report_path.read_text())
        reports[item["variant"]] = report
        folder = root / item["variant"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.txt").write_text(render_summary(report))
        for filename in ("breakdown.json", "breakdown.csv", "run_manifest.json"):
            link = folder / filename
            target = report_path.parent / filename
            if link.exists() or link.is_symlink():
                if not link.is_symlink():
                    raise ValueError(f"Refusing to replace a regular report file: {link}")
                link.unlink()
            link.symlink_to(target.relative_to(folder))
        item["summary"] = str((folder / "summary.txt").resolve())
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    if all(variant in reports for variant in ("baseline", "full")):
        write_kernel_table({"baseline": reports["baseline"], "optimized": reports["full"]},
                           root / "kernel_breakdown.csv")
    case = next(iter(reports.values()))["case"]
    lines = [f"Llama2-70B FP8 SA4x32; LPDDR4 nCCD2; TP{case['tp']}; PP{case['pp']}; batch{case['batch_size']}; "
             f"window{case['context_window']}; active{case['sequence_length']}",
             "Scope: one representative rank, one Transformer block, one batch group.",
             "Baseline: FlashAttention OFF; pipelined Softmax OFF; channel concurrency OFF.",
             "EWMUL_PNM remains ON in the baseline because the systolic implementation forces it on; this is not an all-four-off result.",
             f"Full: EWMUL_PNM ON; FlashAttention ON (block{reports['full']['case']['flash_attention_block_size'] if 'full' in reports else '?'}); "
             "pipelined Softmax ON; channel concurrency ON.",
             "PIM totals are simulated; kernel PIM attribution uses issued gaps. PNM/CXL and overlap use legacy analytical formulas.",
             "", "variant       PIM cycles        PIM(us)       PNM(us)       CXL(us)     Block(us)   ScoreTransfer(us)"]
    for item in comparison:
        lines.append(f"{item['variant']:<12} {int(item['pim_cycles']):>11,} "
                     + " ".join(f"{float(item[key]):>13.6f}" for key in ("pim_us", "pnm_us", "cxl_us", "block_us"))
                     + f" {float(item['score_transfer_cycles']) * 0.000625:>19.6f}")
    organization = next(iter(reports.values()))["case"].get("dram_organization")
    if organization:
        lines.insert(1, f"Resolved interface: {organization['channels']} channels × x{organization['channel_width_bits']}, "
                     f"prefetch{organization['prefetch']}, {organization['transaction_bytes']} B/burst, "
                     f"nBL={next(iter(reports.values()))['case']['nBL']} clocks.")
    lines.extend(publish_interface_correction(root, reports))
    for item in comparison:
        lines.extend(["", "=" * 78, f"{item['variant']} — complete kernel cycle details",
                      f"Stable summary: {item['summary']}", "=" * 78,
                      render_summary(reports[item["variant"]]).rstrip()])
    (root / "summary.txt").write_text("\n".join(lines) + "\n")


def retain_pair(root, trace_root, *, clean_legacy_v00=False):
    """Validate both retained runs before deleting redundant experiment outputs."""
    root, trace_root = Path(root).resolve(), Path(trace_root).resolve()
    with (root / "comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected, retained, deletions = [], {}, []
    raw_keys = ("source_csv", "trace", "systolic_metadata", "pim_log", "resolved_timing",
                "backend_config", "issued_command_trace", "source_provenance")
    for variant in ("baseline", "full"):
        matches = [row for row in rows if row["variant"] == variant]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one indexed {variant} result")
        row = matches[0]
        report_path = Path(row["summary"]).with_name("breakdown.json").resolve()
        report = json.loads(report_path.read_text())
        artifacts = report["run_identity"]["artifacts"]

        def artifact_path(key):
            path = Path(artifacts[key]["path"])
            return (path if path.is_absolute() else CENT.parent / path).resolve()

        run_folder = artifact_path("source_csv").parent
        trace_folder = trace_root / variant / run_folder.name
        if (run_folder.parent != root / variant
                or report_path.parent.parent != run_folder / "breakdown"
                or report_path.parent.name != report["run_id"]):
            raise ValueError(f"Retained {variant} artifacts are outside the expected experiment layout")
        for key in raw_keys:
            path = artifact_path(key)
            path.relative_to(trace_folder if key in ("trace", "systolic_metadata") else run_folder)
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifacts[key]["sha256"]:
                raise ValueError(f"Retained {variant} input changed: {key}")
        for filename in ("breakdown.csv", "summary.txt", "run_manifest.json"):
            if not (report_path.parent / filename).is_file():
                raise ValueError(f"Retained {variant} report is incomplete: {filename}")
        case = report["case"]
        if not case.get("ewmul_pnm") or any(case[k] != (variant == "full") for k in
                                           ("flash_attention", "pipelined_softmax", "channel_concurrency")):
            raise ValueError(f"Unexpected optimization settings in {variant}")
        details = [r["pim_issued_gap_details"] for r in report["regions"] if r.get("pim_issued_gap_details")]
        if (sum(d["total_gap_cycles"] for d in details) != report["full_run_totals"]["pim_native_cycles"]
                or any(abs(value) > 1e-6 for value in report["reconciliation_residuals_ns"].values())):
            raise ValueError(f"Retained {variant} cycle details do not reconcile")
        retained[variant] = {"run": run_folder, "report": report_path.parent, "trace": trace_folder}
        selected.append(row)
    aliases = {"summary.txt", "breakdown.json", "breakdown.csv", "run_manifest.json"}
    for variant in VARIANTS:
        for base in (root, trace_root):
            folder = base / variant
            if not folder.exists():
                continue
            if variant not in retained:
                deletions.append(folder)
                continue
            keep = retained[variant]["run" if base == root else "trace"]
            deletions.extend(p for p in folder.iterdir()
                             if p != keep and not (base == root and p.name in aliases))
        if variant in retained:
            report_dir = retained[variant]["report"]
            deletions.extend(p for p in report_dir.parent.iterdir() if p != report_dir)
    if clean_legacy_v00:
        legacy = CENT / "output/legacy_v0.0/LPDDR4_nCCD2_TP2_32K_batch4"
        if legacy.exists():
            for path in legacy.rglob("breakdown.json"):
                old = json.loads(path.read_text())
                if (old["case"]["tp"], old["case"]["batch_size"], old["full_run_totals"]["pim_native_cycles"]) != (2, 4, 653953):
                    raise ValueError(f"Unexpected legacy result; leaving cleanup untouched: {path}")
            deletions.append(legacy)
    commands = json.loads((root / "commands.json").read_text())
    if any(variant not in commands for variant in retained):
        raise ValueError("Missing reproduction command for a retained result")
    removed_bytes = sum(p.stat().st_size for target in deletions
                        for p in (target.rglob("*") if target.is_dir() else [target])
                        if p.is_file() and not p.is_symlink())
    publish_comparison(root, selected, {variant: commands[variant] for variant in retained})
    for path in deletions:
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    (root / "cleanup_manifest.json").write_text(json.dumps({
        "retained": {variant: {key: str(value) for key, value in paths.items()}
                     for variant, paths in retained.items()},
        "deleted": [str(path) for path in deletions], "removed_bytes": removed_bytes,
    }, indent=2) + "\n")
    return removed_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=DEFAULT_CASE["tp"])
    parser.add_argument("--pp", type=int, default=DEFAULT_CASE["pp"], help="devices = TP x PP")
    parser.add_argument("--context-window", type=int, default=DEFAULT_CASE["context_window"])
    parser.add_argument("--seqlen", type=int,
                        help=f"active decode length (default: window - {DECODE_TOKENS // 2})")
    parser.add_argument("--batch", type=int, default=DEFAULT_CASE["batch"])
    parser.add_argument("--flash-block", type=int,
                        help="FlashAttention block of the full variant (default: default_flash_block(TP))")
    parser.add_argument("--pcie-lanes-per-device", type=int, default=4,
                        help="PCIe lanes per device (total = lanes x TP x PP)")
    parser.add_argument("--output-root", type=Path, help="default: output/sota_results/<case name>")
    parser.add_argument("--trace-root", type=Path, help="default: trace/sota_results/<case name>")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=["baseline", "full"])
    parser.add_argument("--pcie-gen", type=int, choices=sorted(PCIE_GB_PER_LANE), default=DEFAULT_PCIE_GEN,
                        help="PCIe generation for CXL links (per-lane GB/s: %s)" % PCIE_GB_PER_LANE)
    parser.add_argument("--retain-only", action="store_true",
                        help="Keep the indexed baseline/full pair and remove redundant runs; no simulation")
    parser.add_argument("--clean-legacy-v00", action="store_true",
                        help="Also remove the duplicate legacy_v0.0 TP2/batch4 reports")
    args = parser.parse_args()
    if args.seqlen is None:
        args.seqlen = args.context_window - DECODE_TOKENS // 2
    if min(args.tp, args.pp, args.batch, args.seqlen) < 1 or args.seqlen > args.context_window:
        parser.error("TP, PP, batch and seqlen must be positive and seqlen <= context window")
    if LLAMA2_70B_KV_HEADS % args.tp:
        parser.error("TP must divide the 8 KV heads")
    args.flash_block = args.flash_block or default_flash_block(args.tp)
    case_name = default_root_name(args.tp, args.pp, args.context_window, args.batch, args.flash_block)
    args.output_root = args.output_root or CENT / "output/sota_results" / case_name
    args.trace_root = args.trace_root or CENT / "trace/sota_results" / case_name
    devices = args.tp * args.pp
    if args.pcie_lanes_per_device < 1:
        parser.error("--pcie-lanes-per-device must be positive")
    pcie_lanes = args.pcie_lanes_per_device * devices
    root = args.output_root.resolve()
    if args.retain_only:
        removed = retain_pair(root, args.trace_root, clean_legacy_v00=args.clean_legacy_v00)
        print(f"Retained baseline/full; removed {removed / 1024**3:.3f} GiB. Summary: {root / 'summary.txt'}")
        return
    if args.clean_legacy_v00:
        parser.error("--clean-legacy-v00 requires --retain-only")
    root.mkdir(parents=True, exist_ok=True)
    # One PCIe generation per output root: variant aliases and comparison.csv
    # are keyed by variant only.  Roots predating this marker are PCIe 6.
    marker = root / "PCIE_GENERATION"
    if marker.exists():
        root_gen = int(marker.read_text().strip())
    elif (root / "comparison.csv").exists():
        root_gen = DEFAULT_PCIE_GEN
    else:
        root_gen = args.pcie_gen
    if root_gen != args.pcie_gen:
        parser.error(f"{root} holds PCIe {root_gen} results; use a separate --output-root for PCIe {args.pcie_gen}")
    marker.write_text(f"{args.pcie_gen}\n")
    commands, comparison = {}, []
    tracked_sources = ["cent_simulation/run_sim.py", "cent_simulation/cxl_latency.py",
                       "cent_simulation/TransformerBlock.py",
                       "cent_simulation/aim_sim.py", "cent_simulation/tp_mapping.py",
                       "aim_simulator/src/frontend/impl/memory_trace/aim_trace.cpp",
                       "aim_simulator/src/memory_system/impl/aim_DRAM_system.cpp",
                       "aim_simulator/src/dram/impl/LPDDR4.cpp",
                       "aim_simulator/src/dram_controller/impl/plugin/dram_timing_exporter.cpp",
                       "aim_simulator/src/addr_mapper/impl/linear_mappers.cpp",
                       "aim_simulator/test/generated/example_LPDDR4X_nCCD2.yaml",
                       "aim_simulator/build/ramulator2", "aim_simulator/libramulator.so"]
    manifest = {"sha256": {name: hashlib.sha256((CENT.parent / name).read_bytes()).hexdigest()
                            for name in tracked_sources}, "python": sys.version}
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (root / "BUILD_MANIFEST.json").write_text(manifest_text)
    provenance = root / "RUN_PROVENANCE.md"
    provenance.write_text(
        f"# FP8 LPDDR4 nCCD2 TP{args.tp} PP{args.pp} batch{args.batch} optimization comparison\n\n"
        f"Window={args.context_window}; active decode length={args.seqlen}; devices={devices}; "
        f"PCIe lanes={args.pcie_lanes_per_device}/device ({pcie_lanes} total); array=4x32; 32 channels; EWMUL_PNM=true.\n"
        "Corrected LPDDR4 interface: x16/channel; prefetch16; 32 B/burst; nBL8; 625 ps/clock; internal MAC nCCD2.\n"
        "Supersedes width32/prefetch8/nBL4 results. QK query reloads and all host trace work are unchanged.\n"
        "FP8 geometry and legacy analytical formulas restored from ALOI 637537c.\n"
        "PNM and streamed Softmax are analytical models; FlashAttention is a timing-trace abstraction.\n"
        f"CXL: one-shot all-to-all collectives (cxl_latency.py), PCIe {args.pcie_gen}, GB/s, FP8 wire, 192/256 flits.\n"
        "Concurrent mode accepts at most one request/channel/cycle; DRAM timing and SYNC/EOC remain active.\n"
        "WR_GB retains multicast semantics. Native full-run PIM cycles are simulated; kernel attribution uses issued gaps.\n"
        "Reproduce: python scripts/run_sota_results.py\n"
        f"BUILD_MANIFEST.json SHA256: {hashlib.sha256(manifest_text.encode()).hexdigest()}\n"
    )
    # Default PCIe generation keeps the historical option list (and fingerprint).
    pcie_options = [] if args.pcie_gen == DEFAULT_PCIE_GEN else ["--pcie-gen", str(args.pcie_gen)]
    case = {"tp": args.tp, "pp": args.pp, "context_window": args.context_window, "batch": args.batch}
    case_key = {"case": {**case, "seqlen": args.seqlen, "pcie_lanes_per_device": args.pcie_lanes_per_device}}
    for variant in args.variants:
        block_options = ["--flash-attention-block-size", str(args.flash_block)] if variant == "full" else []
        options = VARIANTS[variant] + block_options + pcie_options
        input_fingerprint = hashlib.sha256(json.dumps(
            {"build": manifest, "options": options, **case_key}, sort_keys=True).encode()).hexdigest()[:16]
        out = root / variant / input_fingerprint
        trace_root = args.trace_root.resolve() / variant / input_fingerprint
        out.mkdir(parents=True, exist_ok=True)
        run_provenance = out / "RUN_PROVENANCE.md"
        run_provenance.write_text(provenance.read_text())
        (out / "BUILD_MANIFEST.json").write_text(manifest_text)
        source = out / "simulation_results.csv"
        command = [sys.executable, "run_sim.py", "--model", "Llama2-70B",
                   "--precision", "fp8", "--num_channels", "32", "--num_banks", "8",
                   "--num_devices", str(devices), "--PCIE_lanes", str(pcie_lanes), "--model_parallel",
                   "--kv-head-tp", "--tp-values", str(args.tp), "--decode-only", "--systolic-pim",
                   "--systolic-dim", "4", "--batch-size", str(args.batch), "--EWMUL_PNM",
                   "--seqlen", str(args.seqlen), "--max-seq-len", str(args.context_window),
                   "--experiment", "LPDDR4X_nCCD2", "--dram-power-impl", "LPDDR4X",
                   "--ramulator-config", str(CENT.parent / "aim_simulator/test/generated/example_LPDDR4X_nCCD2.yaml"),
                   "--trace-root", str(trace_root), "--log-root", str(out / "ramulator"),
                   "--simulation_result_path", str(source), "--record-command-trace",
                   "--generate_trace", "--simulate_trace", "--update_csv", *options]
        commands[variant] = command
        print(f"Running {variant}", flush=True)
        with (out / "driver.log").open("w") as stream:
            subprocess.run(command, cwd=CENT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        trace_name = f"trace_{args.tp}_FC_devices_seqlen_{args.seqlen}.txt"
        trace = next(trace_root.rglob(trace_name))
        log = next((out / "ramulator").rglob(trace_name + ".log"))
        stem = str(log).removesuffix(".log")
        row = read_case(source, model="Llama2-70B", tp=args.tp, batch=args.batch,
                        context_window=args.context_window, sequence_length=args.seqlen)
        report = analyze_case(row, source_csv=source, trace=trace, log=log,
                              timing=stem + ".timing.yaml", metadata_path=str(trace) + ".systolic.json",
                              config=stem + ".ramulator.yaml", provenance=run_provenance)
        if (report["case"]["dram_organization"] != LPDDR4_ORGANIZATION
                or report["case"]["nBL"] != 8 or report["case"]["nCCD"] != 2):
            raise ValueError("Rebuild AiM with corrected x16/prefetch16/nBL8 LPDDR4 before publishing")
        report_dir = write_report(report, out / "breakdown")
        totals = report["full_run_totals"]
        comparison.append({"variant": variant, "pim_cycles": totals["pim_native_cycles"],
                           "pim_us": totals["pim_latency_ns"] / 1000,
                           "pnm_us": totals["pnm_exposed_latency_ns"] / 1000,
                           "cxl_us": totals["cxl_latency_ns"] / 1000,
                           "block_us": totals["block_latency_ns"] / 1000,
                           "score_transfer_cycles": next(r["backends"]["pim"]["reference_equivalent_cycles"]
                                                         for r in report["regions"] if r["region_id"] == "aux.score_transfer"),
                           "summary": str(report_dir / "summary.txt")})
        print(f"{variant}: {comparison[-1]['block_us']:.6f} us; {report_dir}", flush=True)
    if (root / "comparison.csv").exists():
        with (root / "comparison.csv").open(newline="") as stream:
            prior = list(csv.DictReader(stream))
        comparison = [r for r in prior if r["variant"] not in args.variants] + comparison
    previous_commands = json.loads((root / "commands.json").read_text()) if (root / "commands.json").exists() else {}
    previous_commands.update(commands)
    publish_comparison(root, comparison, {item["variant"]: previous_commands[item["variant"]] for item in comparison})
    print(f"Comparison: {root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
