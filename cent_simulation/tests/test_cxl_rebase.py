"""Checks for PCIe-generation rebasing of CXL in derived double-buffer reports."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cxl_rebase import apply_pcie_generation, cxl_costs_us, source_lanes_per_device  # noqa: E402


def fixture(pcie_gen=6, precision="fp8"):
    costs = cxl_costs_us(8192, 4, 2, 16, 4, pcie_gen, precision=precision)
    cxl = sum(costs.values())
    kernels = [{"region_id": key, "cxl_equivalent_cycles": us / .000625,
                "old_total_us": us + 10, "new_total_us_estimate": us + 5,
                "old_block_contribution_cycles": (us + 10) / .000625,
                "new_block_contribution_cycles_estimate": (us + 5) / .000625,
                "old_pim_cycles": 16000, "new_pim_cycles_estimate": 8000}
               for key, us in costs.items()]
    return {"source": {"case": {"batch_size": 4, "tp": 2, "pp": 16},
                       "reference_clock_period_ps": 625},
            "models": {"dependency_aware": {
                "kernels": kernels, "waves": [{"hidden_wrgb_cycles": 24000}],
                "totals": {"cxl_us_frozen": cxl, "old_block_us": cxl + 30,
                           "new_block_us_analytical": cxl + 15, "saved_us": 15,
                           "new_pim_cycles_analytical": 24000}}},
            "validation": {}, "assumptions": []}


class CXLRebaseTests(unittest.TestCase):
    def test_block_costs_one_shot_all_to_all(self):
        costs = cxl_costs_us(8192, 4, 2, 16, 4, 7, precision="fp8")
        wire = 171 * 256 / 64e9 * 1e6  # FP8, flit-packed, 4 lanes x 16 GB/s, no discount
        self.assertAlmostEqual(costs["attention.o_projection"], 0.180 + wire)
        self.assertAlmostEqual(costs["ffn.down_projection"], 0.180 + wire)
        self.assertAlmostEqual(costs["aux.pp_handoff"], 0.180 + wire)

    def test_same_generation_preserves_all_model_values(self):
        result = fixture()
        before = copy.deepcopy(result["models"])
        apply_pcie_generation(result, {"Device number": "32", "Precision": "FP8"}, {"dim": 8192})
        self.assertEqual(result["models"], before)
        self.assertEqual(result["cxl_bandwidth_model"]["pcie_generation"], 6)
        with self.assertRaises(ValueError):
            apply_pcie_generation(result, {"Device number": "32", "Precision": "FP8"}, {"dim": 8192})

    def test_target_generation_rebases_both_columns_and_keeps_pim(self):
        result = fixture()
        before = copy.deepcopy(result)
        apply_pcie_generation(result, {"Device number": "32", "Precision": "FP8"}, {"dim": 8192}, pcie_gen=7)
        pcie7 = sum(cxl_costs_us(8192, 4, 2, 16, 4, 7, precision="fp8").values())
        new = result["models"]["dependency_aware"]
        old = before["models"]["dependency_aware"]
        self.assertEqual(result["source"], before["source"])
        self.assertEqual(new["waves"], old["waves"])
        for r, previous in zip(new["kernels"], old["kernels"]):
            self.assertEqual(r["old_pim_cycles"], previous["old_pim_cycles"])
            self.assertEqual(r["new_pim_cycles_estimate"], previous["new_pim_cycles_estimate"])
        self.assertAlmostEqual(new["totals"]["cxl_us_frozen"], pcie7)
        self.assertAlmostEqual(new["totals"]["old_block_us"], pcie7 + 30)
        self.assertAlmostEqual(new["totals"]["new_block_us_analytical"], pcie7 + 15)
        self.assertEqual(new["totals"]["saved_us"], 15)
        model = result["cxl_bandwidth_model"]
        self.assertEqual((model["source_pcie_generation"], model["pcie_generation"]), (6, 7))
        self.assertEqual(model["effective_bandwidth_per_device"], 64)
        self.assertEqual(model["multicast_bandwidth_factor"], 1.0)
        self.assertEqual(model["wire_bytes_per_value"], 1)

    def test_source_generation_is_read_from_csv_row(self):
        result = fixture(pcie_gen=5)
        with self.assertRaises(ValueError):  # a PCIe 6 assumption cannot reproduce a PCIe 5 source
            apply_pcie_generation(copy.deepcopy(result), {"Device number": "32", "Precision": "FP8"}, {"dim": 8192})
        apply_pcie_generation(result, {"Device number": "32", "PCIe generation": "5", "Precision": "FP8"}, {"dim": 8192})
        self.assertEqual(result["cxl_bandwidth_model"]["pcie_generation"], 5)

    def test_wire_precision_comes_from_csv_row(self):
        result = fixture(precision="bf16")
        with self.assertRaises(ValueError):  # FP8 costs cannot reproduce a BF16 source
            apply_pcie_generation(copy.deepcopy(result), {"Device number": "32", "Precision": "FP8"}, {"dim": 8192})
        apply_pcie_generation(result, {"Device number": "32", "Precision": "BF16"}, {"dim": 8192}, pcie_gen=7)
        model = result["cxl_bandwidth_model"]
        self.assertEqual((model["wire_precision"], model["wire_bytes_per_value"]), ("bf16", 2))

    def test_lanes_per_device_come_from_csv_row(self):
        self.assertEqual(source_lanes_per_device({"Device number": "8", "PCIe lanes per device": "4"}), 4)
        self.assertEqual(source_lanes_per_device({"Device number": "8"}), 18)  # legacy 144 split
        self.assertEqual(source_lanes_per_device({"Device number": "32", "Precision": "FP8"}), 4)


if __name__ == "__main__":
    unittest.main()
