import math
import unittest

import cxl_latency
from scripts import cxl_pcie_sweep

# Reference spreadsheet: (gen, single layer, two stages, 128 layers / 8 stages); raw FP8 bytes, GiB/s.
FIGURE = [(5, 4.095, 10.167, 537.963), (6, 2.187, 5.398, 287.146), (7, 1.234, 3.014, 161.738)]


def comm(rows, gen, trace):
    return next(r["Exposed Communication (us)"] for r in rows
                if r["PCIe Version"] == gen and r["Trace"] == trace)


class CxlModelTest(unittest.TestCase):
    def test_reproduces_reference_spreadsheet_without_flit_packing(self):
        rows = cxl_pcie_sweep.sweep(267.937, 128, 8, 8192, 4, 2, 4, [5, 6, 7], 1, False, "Llama 70B", 2**30,
                                    tp_fixed_ns=140, pp_fixed_ns=70)
        for gen, single, two, full in FIGURE:
            self.assertAlmostEqual(comm(rows, gen, "Single Layer"), single, places=3)
            self.assertAlmostEqual(comm(rows, gen, "Two Pipeline Stages"), two, places=3)
            self.assertAlmostEqual(comm(rows, gen, "Llama 70B"), full, places=3)

    def test_flit_packing_overhead(self):
        self.assertEqual(cxl_latency.wire_bytes(32768, bytes_per_value=1, flit_packing=False), 32768)
        self.assertEqual(cxl_latency.wire_bytes(32768, bytes_per_value=1), 171 * 256)
        self.assertEqual(cxl_latency.wire_bytes(32768, bytes_per_value=2), 342 * 256)

    def test_default_block_costs_gb_s_fp8(self):
        costs = cxl_latency.block_cxl_costs_us(32768, 4, 2, 16, pcie_gen=6, precision="fp8")
        wire = 171 * 256 / 32e9 * 1e6
        self.assertEqual(cxl_latency.TP_FIXED_NS, 180)
        self.assertEqual(cxl_latency.PP_FIXED_NS, 180)
        self.assertAlmostEqual(costs["attention.o_projection"], 0.180 + wire)
        self.assertAlmostEqual(costs["ffn.down_projection"], 0.180 + wire)
        self.assertAlmostEqual(costs["aux.pp_handoff"], 0.180 + wire)
        self.assertEqual(cxl_latency.block_cxl_costs_us(32768, 4, 2, 1, precision="fp8")["aux.pp_handoff"], 0.0)

    def test_run_sim_interface_matches_block_costs(self):
        costs = cxl_latency.block_cxl_costs_us(32768, 4, 2, 16, pcie_gen=7, precision="fp8")
        self.assertAlmostEqual(cxl_latency.kv_head_tp_latency_ms(32768, 4, 2, 32, 7, precision="fp8") * 1000,
                               costs["attention.o_projection"] + costs["ffn.down_projection"])
        self.assertAlmostEqual(cxl_latency.pp_handoff_latency_ms(32768, 4, 7, precision="fp8") * 1000, costs["aux.pp_handoff"])
        self.assertEqual(cxl_latency.kv_head_tp_latency_ms(4096, 18, 1, 16, precision="fp8"), 0.0)

    def test_generation_scales_only_the_wire_term(self):
        wire = {gen: cxl_latency.pp_handoff_latency_ms(32768, 4, gen, precision="fp8") * 1000 - 0.180 for gen in (5, 6, 7)}
        self.assertAlmostEqual(wire[5] / wire[6], 2.0)
        self.assertAlmostEqual(wire[6] / wire[7], 2.0)
        with self.assertRaises(ValueError):
            cxl_latency.pp_handoff_latency_ms(32768, 4, pcie_gen=8, precision="fp8")

    def test_pipeline_counts(self):
        tp_us, pp_us = cxl_latency.pipeline_comm_us(80, 5, 32768, 32, bytes_per_value=1)
        self.assertTrue(math.isclose(tp_us, 160 * cxl_latency.allreduce_us(32768, 32, bytes_per_value=1)))
        self.assertTrue(math.isclose(pp_us, 4 * cxl_latency.transfer_us(32768, 32, bytes_per_value=1)))
        self.assertEqual(cxl_latency.pipeline_comm_us(80, 1, 32768, 32, bytes_per_value=1)[1], 0.0)
        self.assertEqual(cxl_latency.allreduce_us(32768, 32, tp=1, bytes_per_value=1), 0.0)
        with self.assertRaises(ValueError):
            cxl_latency.pipeline_comm_us(80, 81, 32768, 32, bytes_per_value=1)

    def test_tp_scales_ingress(self):
        one = cxl_latency.allreduce_us(32768, 32, tp=2, fixed_ns=0, bytes_per_value=1)
        self.assertTrue(math.isclose(cxl_latency.allreduce_us(32768, 32, tp=4, fixed_ns=0, bytes_per_value=1), 3 * one))

    def test_cent_tp_mapping_uses_allgathers(self):
        bw = cxl_latency.device_bandwidth(4, 6)
        expected = 5 * cxl_latency.allgather_us(8192, bw, 4, bytes_per_value=2) + cxl_latency.allgather_us(28672, bw, 4, bytes_per_value=2)
        self.assertAlmostEqual(cxl_latency.cent_tp_latency_ms([8192, 28672], 4, 4, 32, precision="bf16") * 1000, expected)
        self.assertEqual(cxl_latency.allgather_us(8192, bw, 1, bytes_per_value=2), 0.0)

    def test_wire_precision_follows_simulation_precision(self):
        self.assertEqual(cxl_latency.wire_bytes_per_value("fp8"), 1)
        self.assertEqual(cxl_latency.wire_bytes_per_value("BF16"), 2)
        self.assertEqual(cxl_latency.pp_handoff_pcie_bits(32768, precision="fp8"), 32768 * 8)
        self.assertEqual(cxl_latency.pp_handoff_pcie_bits(32768, precision="bf16"), 32768 * 16)
        self.assertEqual(cxl_latency.kv_head_tp_pcie_bits(8192, 2, precision="bf16")[0],
                         2 * cxl_latency.kv_head_tp_pcie_bits(8192, 2, precision="fp8")[0])
        fp8 = cxl_latency.block_cxl_costs_us(32768, 4, 2, 16, precision="fp8")
        bf16 = cxl_latency.block_cxl_costs_us(32768, 4, 2, 16, precision="bf16")
        for key in fp8:  # the wire term doubles (171 -> 342 flits); the 180 ns stays
            self.assertAlmostEqual(bf16[key] - 0.180, 2 * (fp8[key] - 0.180))
        with self.assertRaises(TypeError):  # precision must be explicit
            cxl_latency.block_cxl_costs_us(32768, 4, 2, 16)
        with self.assertRaises(ValueError):
            cxl_latency.wire_bytes_per_value("int4")

    def test_run_sim_generation_defaults_for_older_namespaces(self):
        from types import SimpleNamespace
        import run_sim
        self.assertEqual(run_sim.cxl_precision(SimpleNamespace()), "bf16")
        self.assertEqual(run_sim.cxl_precision(SimpleNamespace(precision="fp8")), "fp8")
        self.assertEqual(run_sim.pcie_generation(SimpleNamespace()), 6)
        self.assertEqual(run_sim.pcie_generation(SimpleNamespace(pcie_gen=7)), 7)


if __name__ == "__main__":
    unittest.main()
