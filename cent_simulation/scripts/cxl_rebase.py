"""Rebase the CXL time of derived double-buffer reports to another PCIe generation.

The source report's CXL is first reproduced with ``cxl_latency.block_cxl_costs_us``
at the generation recorded by run_sim (rows without the column are PCIe 6).
Both the single- and double-buffer columns are then recomputed at the
requested generation; PIM and PNM are untouched.
"""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cxl_latency  # noqa: E402
from cxl_latency import DEFAULT_PCIE_GEN, block_cxl_costs_us, pcie_lane_bandwidth  # noqa: E402

LEGACY_TOTAL_PCIE_LANES = 144  # rows without "PCIe lanes per device" split 144 lanes


def source_lanes_per_device(source_row):
    """PCIe lanes per device recorded by run_sim (older rows: 144 // devices)."""
    value = source_row.get("PCIe lanes per device")
    if value in (None, ""):
        return LEGACY_TOTAL_PCIE_LANES // int(source_row["Device number"])
    return int(float(value))


def source_pcie_generation(source_row):
    """PCIe generation recorded by run_sim; rows predating the column are PCIe 6."""
    value = source_row.get("PCIe generation")
    return DEFAULT_PCIE_GEN if value in (None, "") else int(float(value))


def source_precision(source_row):
    """Simulation precision recorded by run_sim (rows without it are BF16)."""
    return str(source_row.get("Precision") or "bf16").lower()


def cxl_costs_us(hidden_dim, batch, tp, pp, lanes, pcie_gen=DEFAULT_PCIE_GEN, *, precision):
    return block_cxl_costs_us(hidden_dim * batch, lanes, tp, pp, pcie_gen, precision=precision)


def apply_pcie_generation(result, source_row, shape, pcie_gen=None):
    """Rebase both single/double-buffer comparisons; preserve source provenance."""
    if "cxl_bandwidth_model" in result:
        raise ValueError("CXL rebase already applied")
    case = result["source"]["case"]
    devices = int(source_row["Device number"])
    lanes = source_lanes_per_device(source_row)
    source_gen = source_pcie_generation(source_row)
    target_gen = source_gen if pcie_gen is None else int(pcie_gen)
    precision = source_precision(source_row)
    old_costs = cxl_costs_us(shape["dim"], case["batch_size"], case["tp"], case["pp"], lanes, source_gen,
                             precision=precision)
    new_costs = cxl_costs_us(shape["dim"], case["batch_size"], case["tp"], case["pp"], lanes, target_gen,
                             precision=precision)
    changed = target_gen != source_gen
    old_us, new_us = sum(old_costs.values()), sum(new_costs.values())
    period_us = result["source"]["reference_clock_period_ps"] / 1e6
    delta_us = new_us - old_us
    for model in result["models"].values():
        totals = model["totals"]
        if not math.isclose(totals["cxl_us_frozen"], old_us, abs_tol=1e-9):
            raise ValueError("CXL model does not reproduce the source report's CXL total")
        for row in model["kernels"]:
            old_row_us = row["cxl_equivalent_cycles"] * period_us
            if not math.isclose(old_row_us, old_costs.get(row["region_id"], 0.0), abs_tol=1e-9):
                raise ValueError(f"CXL allocation differs from the source: {row['region_id']}")
            if row["region_id"] not in new_costs or not changed:
                continue
            delta = new_costs[row["region_id"]] - old_row_us
            row["cxl_equivalent_cycles"] = new_costs[row["region_id"]] / period_us
            for key in ("old_total_us", "new_total_us_estimate"):
                row[key] += delta
            for key in ("old_block_contribution_cycles", "new_block_contribution_cycles_estimate"):
                row[key] += delta / period_us
        if changed:
            totals["cxl_us_frozen"] = new_us
            for key in ("old_block_us", "new_block_us_analytical"):
                totals[key] += delta_us
            totals["old_block_equivalent_cycles"] = totals["old_block_us"] / period_us
            totals["new_block_equivalent_cycles_analytical"] = totals["new_block_us_analytical"] / period_us
            totals["block_reduction_percent"] = 100 * totals["saved_us"] / totals["old_block_us"]
            totals["block_speedup"] = totals["old_block_us"] / totals["new_block_us_analytical"]
            totals["per_output_token_amortized_us"] = totals["new_block_us_analytical"] / case["batch_size"]
        for row_key, total_key in (("old_total_us", "old_block_us"),
                                   ("new_total_us_estimate", "new_block_us_analytical")):
            if not math.isclose(sum(r[row_key] for r in model["kernels"]), totals[total_key], abs_tol=1e-9):
                raise ValueError(f"Adjusted CXL rows do not reconcile: {total_key}")
    effective = lanes * pcie_lane_bandwidth(target_gen)
    result["cxl_bandwidth_model"] = {
        "cxl_model": "one_shot_all_to_all", "unit": "GB/s",
        "total_pcie_lanes": lanes * devices, "devices": devices, "lanes_per_device": lanes,
        "source_pcie_generation": source_gen, "pcie_generation": target_gen,
        "raw_bandwidth_per_lane": pcie_lane_bandwidth(target_gen),
        "multicast_bandwidth_factor": 1.0, "effective_bandwidth_per_device": effective,
        "fixed_transfer_overhead_ns": {"tp_transfer": cxl_latency.TP_FIXED_NS,
                                       "pp_handoff": cxl_latency.PP_FIXED_NS},
        "wire_precision": precision, "wire_bytes_per_value": cxl_latency.wire_bytes_per_value(precision),
        "payload_bytes_per_flit": cxl_latency.FLIT_PAYLOAD_BYTES, "wire_bytes_per_flit": cxl_latency.FLIT_BYTES,
        "original_source_cxl_us": old_us, "selected_cxl_us": new_us,
        "block_delta_us_from_original_source": delta_us,
        "original_region_cxl_us": old_costs, "selected_region_cxl_us": new_costs,
        "pp_handoff_policy": "one_transfer_per_block_when_pp_gt_1",
        "scope": "Derived reports only; original simulation sources remain immutable.",
    }
    result["validation"]["cxl_model_reproduces_source"] = True
    result["validation"]["cxl_rebase_applied_to_both_buffer_baselines"] = True
    note = (f"CXL: PCIe {target_gen} (source PCIe {source_gen}), one-shot all-to-all TP all-reduce, "
            f"{lanes} lanes/device x {pcie_lane_bandwidth(target_gen)} = {effective:g} GB/s, {precision.upper()} wire payload, "
            f"192/256 flit packing, {cxl_latency.TP_FIXED_NS} ns per TP transfer, "
            f"{cxl_latency.PP_FIXED_NS} ns per PP handoff; source CXL={old_us:.9f} us, selected CXL={new_us:.9f} us. "
            "Both single/double-buffer columns use the selected generation.")
    result["assumptions"] = [
        ("PNM and Softmax exposed time stay as in the source. CXL stays fixed across the "
         "single/double-buffer comparison at the selected PCIe generation.")
        if a.startswith("PNM, CXL and legacy Softmax") else a
        for a in result["assumptions"]
    ] + [note]
    return result
