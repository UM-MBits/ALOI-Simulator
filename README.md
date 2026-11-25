# Artifact of the CENT Paper, ASPLOS 2025

This repository provides the following artifact required for the evaluation of **CENT**, "PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference" paper published in ASPLOS 2025.

<p align="center">
<img src="CENT.png" alt="CENT" width="80%">
</p>

## Current implementation and SOTA results

The performance results use the **legacy trace-driven simulation framework**:
Python generates AiM traces, the C++ AiM/Ramulator backend models PIM timing,
and Python adds analytical PNM/CXL latency and energy. The planned ALOI
compiler/command DAG and new functional + cycle-level simulation engine are not
the implementation behind these results.

The current SOTA configuration is **Llama2-70B, FP8, SA4×32, batch4**, evaluated at
8K (TP1×PP8), 16K (TP1×PP8), 32K (TP2×PP8) and 64K (TP4×PP4)
windows, LPDDR4 **32 channels × x16,
prefetch16, nBL=8, nCCD=2, 625 ps/clock**, with all four optimizations on
(EWMUL_PNM, FlashAttention, pipelined Softmax, channel concurrency).
See [SOTA_RESULTS.md](SOTA_RESULTS.md) for the method, CXL model, reproduction
commands and per-kernel results. Results live (gitignored) under
`cent_simulation/output/sota_results/` (one folder per case).

| Context | Parallelism | FlashAttention block | PCIe | Single-buffer block (µs) | Double-buffer block (µs) | tokens/s/device (single → double) | Double-buffer mJ/token | Double-buffer tokens/J | Double-buffer W/device |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | TP1 × PP8 | 2048 | 5.0 | 403.475 | 332.039 | 123.92 → 150.58 | 165.27 | 6.051 | 24.89 |
| 8K | TP1 × PP8 | 2048 | 7.0 | 401.423 | 329.987 | 124.56 → 151.52 | 165.27 | 6.051 | 25.04 |
| 16K | TP1 × PP8 | 2048 | 5.0 | 469.835 | 388.159 | 106.42 → 128.81 | 194.63 | 5.138 | 25.07 |
| 16K | TP1 × PP8 | 2048 | 7.0 | 467.783 | 386.107 | 106.89 → 129.50 | 194.63 | 5.138 | 25.20 |
| 32K | TP2 × PP8 | 2048 | 5.0 | 321.046 | 267.929 | 77.87 → 93.31 | 258.75 | 3.865 | 24.14 |
| 32K | TP2 × PP8 | 2048 | 7.0 | 314.890 | 261.773 | 79.39 → 95.50 | 258.75 | 3.865 | 24.71 |
| 64K | TP4 × PP4 | 4096 | 5.0 | 240.693 | 204.476 | 51.93 → 61.13 | 381.19 | 2.623 | 23.30 |
| 64K | TP4 × PP4 | 4096 | 7.0 | 226.329 | 190.112 | 55.23 → 65.75 | 381.19 | 2.623 | 25.06 |

Single-buffer PIM is simulated; double buffering is an analytical estimate on
that run. PNM uses 32 lanes (FP8) × 32 accelerators per device; every device
has 4 PCIe lanes. The FlashAttention block is chosen so one QK wave fills all
32 lanes (TP1/TP2: 2048, TP4: 4096); a 64K block-2048 control is kept.

These are per-block, per-batch-group results on one representative TP rank.

**CXL** ([`cent_simulation/cxl_latency.py`](cent_simulation/cxl_latency.py)) is
one model everywhere: one-shot all-to-all TP all-reduce, unicast (no multicast
discount), values on the wire at the simulation precision (FP8 1 B, BF16 2 B)
with 192 B payload per 256 B flit, GB/s per lane per
direction set by `--pcie-gen {5,6,7}` (4/8/16), and a 180 ns fixed overhead per
transfer (4 CXL ports, 2 PCIe PHY traversals, 1 switch).

Double buffering here is an analytical estimate on the simulated single-buffer
run. The `double-buffering` branch additionally implements the opt-in `AiMAsync`
two-GB execution profile, selected by `--pim-double-buffering`; see its
[double-buffer comparison](https://github.com/Linus-Wangjz/ALOI/blob/double-buffering/cent_simulation/results/double_buffering_fp8_32k_batch4/summary.txt).
New performance runs should explicitly enable the four default optimizations
documented below.

## Dependencies

AiM Simulator is tested and verified with `g++-11/12/13` and `clang++-15`.
Python infrastructure requires `pandas`, `matplotlib`, `torch`, and `scipy` packages.

## Build

Clone the repository recursively:

```bash
git clone --recursive https://github.com/Linus-Wangjz/ALOI.git
cd ALOI
```

For the double-buffer execution profile, switch both the parent checkout and
its pinned simulator before building:

```bash
git switch double-buffering
git submodule update --init --recursive
```

Install the Python packages locally or create a conda environment:

```bash
conda create -n cent python=3.10 -y
conda activate cent
pip install -r requirements.txt
```

Build the AiM simulator:

```bash
# use g++-11/12/13. e.g. (export CXX=/usr/bin/g++-12)
cd aim_simulator
mkdir build
cd build
cmake ..
make -j4
cd ../../
```

## Artifact Scripts

Remove old results:

```bash
bash remove_old_results.sh
```

### Run Simulation

#### Default optimizations — required for agents and new runs

**后续 agent 跑新的性能实验时，默认必须同时开启以下四项优化。**
For new systolic KV-head TP performance runs, explicitly enable all four flags
unless the user requests a historical baseline or an ablation experiment:

| Optimization | Required flag | Behavior |
| --- | --- | --- |
| PNM element-wise multiplication | `--EWMUL_PNM` | RMSNorm, RoPE, and FFN element-wise multiplies use PNM; systolic PIM also enables this implicitly, but write the flag explicitly. |
| FlashAttention | `--flash-attention --flash-attention-block-size 2048` | Block QK/SV execution avoids the score DRAM workspace write/read. The default block is 2048 = 64 banks per KV head × 32 FP8 lanes; 1024 fills only half the QK lanes and nearly doubles QK waves at TP2. |
| Pipelined Softmax | `--pipelined-softmax` | Apply the legacy QK/Softmax overlap model while retaining full Softmax work and dynamic energy. |
| Channel concurrency | `--channel-concurrency` | Up to one accepted request per channel per cycle, across 32 channels, including expanded WR_GB bursts. |

This is the default experiment policy. CLI flags remain opt-in for historical
reproducibility, so a command that omits them does not automatically enable all
four. WR_GB retains channel-mask multicast; DRAM timing, backpressure, and
SYNC/EOC remain active. Use the supported systolic KV-head TP path for this
configuration; identify unsupported options explicitly when using other paths.

For the current FP8, LPDDR4 nCCD2, TP2, batch4, 32K case, build AiM and run the
fully optimized variant:

```bash
cd cent_simulation
python3 scripts/run_sota_results.py --variants full
cd ..
```

For a broader optimized campaign, pass the four flags explicitly:

```bash
cd cent_simulation
python3 scripts/run_cent_memory_cases.py \
  --kv-head-tp-systolic --precision fp8 \
  --models Llama2-70B --context-windows 32768 --cases LPDDR4X_nCCD2 \
  --batch-sizes 4 \
  --EWMUL_PNM --flash-attention --flash-attention-block-size 2048 \
  --pipelined-softmax --channel-concurrency \
  --record-command-trace --trace-workers 8 --run-workers 8
cd ..
```

The broader command sweeps TP=1/2/4/8. The 32K window is 32,768 tokens, with a
30,976-token midpoint decode sample. Select models, contexts, memory cases,
and batches explicitly for the requested experiment. Generated CSVs are under
`cent_simulation/output`; functional traces are under `cent_simulation/trace`.

Check that the result CSV/summary records `EWMUL PNM effective`, `Flash attention`,
`Pipelined softmax`, and `Channel concurrency` as true. Concurrent YAML must have
frontend `issue_width=32` and memory `host_issue_width=32`; the simulator statistic
`max_channel_requests_per_cycle` must be 1. Label any disabled optimization in
the run name and report, and keep its traces/logs separate from optimized runs.

#### Historical artifact reproduction

The original command below runs decode-only trace generation, Ramulator
simulation, and Cellar-based energy modeling for GDDR6 and LPDDR4X nCCD=2/6.
Use it when reproducing the historical configuration; it omits the four
optimization flags:

```bash
cd cent_simulation
python3 scripts/run_cent_memory_cases.py --trace-workers 8 --run-workers 8
cd ..
```

Its default matrix is Llama2-7B/Llama2-70B at 4K, 32K, and 128K context. Use
`--models`, `--contexts`, and `--cases` to select a subset. The generated CSVs
are under `cent_simulation/output`; functional traces are under
`cent_simulation/trace`.

### Legacy v0.0 cycle breakdown

`cent_simulation/scripts/analyze_cycle_breakdown.py` reads an existing
Llama2-70B systolic KV-head TP run and writes `breakdown.json`,
`breakdown.csv`, `summary.txt`, and `run_manifest.json` to a content-addressed
directory. `--plot` also writes a stacked latency figure. It does not generate
traces or invoke Ramulator.

For the archived FP8/SA4x32 LPDDR4 nCCD2, TP2, batch4 case:

```bash
breakdown_case_trace=cent_simulation/trace/kv_head_tp_systolic_fp8_32K/long_context_midpoint/LPDDR4X/systolic_pim_4_batch_size_4_ewmul_pnm_1_fp8/model_parallel_kv_head_main/Llama2-70B/trace_2_FC_devices_seqlen_30976.txt
breakdown_case_log=cent_simulation/output/kv_head_tp_systolic_fp8_32K/LPDDR4X/ramulator_long_context_midpoint_nCCD2/systolic_pim_4_batch_size_4_ewmul_pnm_1_fp8/model_parallel_kv_head_main/Llama2-70B/trace_2_FC_devices_seqlen_30976.txt.log
python cent_simulation/scripts/analyze_cycle_breakdown.py \
  --source-csv cent_simulation/output/kv_head_tp_systolic_fp8_32K/LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD2.csv \
  --trace "$breakdown_case_trace" --log "$breakdown_case_log" \
  --provenance cent_simulation/output/kv_head_tp_systolic_fp8_32K/RUN_PROVENANCE.md \
  --tp 2 --batch-size 4 --context-window 32768 --sequence-length 30976 \
  --expected-dram LPDDR4 --expected-nccd 2 \
  --output-dir cent_simulation/output/legacy_v0.0/LPDDR4_nCCD2_TP2_32K_batch4 \
  --plot
```

The 32K campaign has a 32,768-token window and a 30,976-token active decode
sample. The resolved timing implementation is LPDDR4; the archived power
label is LPDDR4X. The measured full-trace PIM cycle count is preserved.
Per-kernel PIM values are estimates: `auto` uses existing issued-command
traces aligned to validated sidecar work budgets, otherwise a single
channel's request-service counters apportioned by traced work. Startup,
background, tail, and unidentified time remain explicit. There are no
kernel completion timestamps in v0.0. PNM/CXL retain the legacy formulas;
equivalent cycles use the recorded PIM clock. Residual adds use the PNM
VEC_ADD formula when the source row records them; older rows mark them
unmodeled. FlashAttention without supported ownership metadata
retains its full total/formulas with PIM region attribution unavailable.

With issued-command traces, `summary.txt` expands every PIM region into
MAC, GB/register I/O, ACT/PRE, and control-command gap contributions. It also
lists raw issued-command counts and MAC predecessor/gap histograms; these
contributions reconcile to the existing PIM totals and are not additional
latency charges. The same details are saved in `breakdown.json`.

The tool checks trace/log counts, metadata, resolved timing, source CSV
latency, and additive totals. Missing sidecars leave PIM regions unavailable.
Use `--estimator service-work` to compare the conservative fallback and
`--estimator issued-gap` to require existing TraceRecorder evidence.

Run the regression suite from `cent_simulation`:

```bash
cd cent_simulation
python -m unittest discover -s tests -v
```

When explicitly requested to compare a baseline and ablations for FP8 TP2/batch4,
build AiM and run the extra variants into a separate output root:

```bash
cd cent_simulation
python scripts/run_sota_results.py --variants baseline parallel flash full full_flash1024 \
  --output-root output/<ablation_root> --trace-root trace/<ablation_root>
```

The baseline disables FlashAttention, pipelined Softmax, and channel concurrency;
**EWMUL_PNM stays enabled because systolic PIM forces it on.** It must not be
described as an all-four-off run. Trace/log directories include a build/input
fingerprint (including `cxl_latency.py` and the PCIe generation), so a changed
backend, generator or CXL model does not reuse incompatible simulation logs.
Each run saves its source and binary SHA256 records in `BUILD_MANIFEST.json`.
One output root holds one PCIe generation.

LPDDR4 results use **32 channels × x16, prefetch16, 32-byte transactions, and
nBL=8 clocks** at 625 ps/clock; internal PIM MAC spacing stays nCCD=2.
FP8 uses a 4x32 array and an 8192-token Softmax startup window. PNM vector
ops are one 256-bit word wide: 32 lanes for FP8, 16 for BF16 (`--pnm-lanes`).
`--pnm-accelerators` sets the PNM accelerators per device (default: one per
channel); they share `--parallel-sram` shared-buffer ports, so each read takes
accelerators / ports cycles.

Concurrent mode sets frontend `issue_width=32` and memory `host_issue_width=32`.
Every channel accepts at most one request per cycle, including expanded WR_GB
bursts. WR_GB retains its existing channel-mask multicast; DRAM timing,
backpressure, and SYNC/EOC remain active. Width 1 preserves legacy admission.
`--record-command-trace` records issued commands with the legacy energy model.
FlashAttention remains the existing analytical PNM/online-softmax abstraction;
the run validates timing and command structure, not numerical correctness.

Generate all figures using the following script or generate them one by one using scripts in sections below:

```bash
bash generate_figures.sh
```

After completing this step, CSV and PDF files for Figures 11–14 will be generated in the `figures` and `figure_source_data` directories. The figures in the published paper were created using `cent_simulation/Figures.xlsx` Excel file.
To reproduce the figures in the same format as those in the paper, copy the content of the generated CSV files into the `Data` sheet of the Excel file. The figures will then be generated in the `Figures` sheet.

### Figure 12

The CXL controller costs are broken down into die, packaging and Non Recurring Engineering (NRE) components. The die cost is derived from the wafer cost, considering the CXL controller die area and yield rate. The cost of 2D packaging is assumed to be 29% of chip cost (die and package). The NRE cost is influenced by chip production volumes.

```bash
python figure_scripts/figure_12.py
```

### Figure 13

CENT speedup over GPU baselines. (a) Batch = 1 Latency comparison. (b) Throughput comparison under the highest batch size that CENT and GPU and achieve. (c) TCO normalized throughput comprison.

```bash
python figure_scripts/figure_13a.py
python figure_scripts/figure_13b.py
python figure_scripts/figure_13c.py
```

### Figure 14

Analysis on Llama2-70B. (a) CENT achieves higher decoding throughputs with long context windows and 3.5K decoding sizes. (b) QoS analysis: CENT shows less query latency when achieving the similar to GPUs. (c) CENT latency breakdown with different parallelism strategies. (d) Prefill (In) and decoding (Out) latency comparison with different In/Out sizes, at maximum supported batches for both GPU and CENT.

```bash
python figure_scripts/figure_14a.py
python figure_scripts/figure_14b.py
python figure_scripts/figure_14c.py
python figure_scripts/figure_14d.py
```

### Figure 15

Power and energy analysis. (a) Power consumption, (b) GPU SM frequency and board power, and (c) energy efficiency of CENT and GPU for different stages of Llama2 models using the maximum batch size, 512 prefill tokens and 3584 decoding tokens.

```bash
python figure_scripts/figure_15a.py
python figure_scripts/figure_15c.py
```

## Citation

If you use *CENT*, please cite this paper:

> Yufeng Gu, Alireza Khadem, Sumanth Umesh, Ning Liang, Xavier Servot, Onur Mutlu, Ravi Iyer, and Reetuparna Das.
> *PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference*,
> In 2025 International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)

```
@inproceedings{cent,
  title={PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference},
  author={Gu, Yufeng and Khadem, Alireza and Umesh, Sumanth, and Liang, Ning and Servot, Xavier and Mutlu, Onur and Iyer, Ravi and and Das, Reetuparna},
  booktitle={2025 International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)}, 
  year={2025}
}
```

## Issues and bug reporting

We appreciate any feedback and suggestions from the community.
Feel free to raise an issue or submit a pull request on Github.
For assistance in using CENT, please contact: Yufeng Gu (yufenggu@umich.edu) and Alireza Khadem (arkhadem@umich.edu)

## Licensing

This repository is available under a [MIT license](/LICENSE).

## Acknowledgement

This work was supported in part by the NSF under the CAREER-1652294 and NSF-1908601 awards and by Intel gift.
