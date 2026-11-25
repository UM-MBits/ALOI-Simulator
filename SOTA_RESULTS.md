# SOTA 结果：8K / 16K / 32K / 64K 单缓冲与 double buffering（PCIe 5.0 / 7.0）

本文说明当前 SOTA 配置（Llama2-70B，四项优化全开，FlashAttention / online softmax block 8K/16K/32K 为 2048、64K 为 4096）的实现方式、CXL 和 PNM 模型、复现命令，以及性能、能耗和功耗结果。每个 context 保留两组结果：

- **单缓冲**：Ramulator AiM 仿真，PIM 周期来自仿真。
- **double buffering**：在单缓冲仿真结果上做的**解析估算**，没有重跑 Ramulator。

| Context | 并行 | device 数 | 结果目录（`cent_simulation/output/sota_results/` 下，被 gitignore） |
|---|---|---:|---|
| 8K | TP1 × PP8 | 8 | `LPDDR4_nCCD2_TP1_PP8_8K_batch4_flash2048_pnm_residual/` |
| 16K | TP1 × PP8 | 8 | `LPDDR4_nCCD2_TP1_PP8_16K_batch4_flash2048_pnm_residual/` |
| 32K | TP2 × PP8 | 16 | `LPDDR4_nCCD2_TP2_PP8_32K_batch4_flash2048_pnm_residual/` |
| 64K | TP4 × PP4 | 16 | `LPDDR4_nCCD2_TP4_PP4_64K_batch4_flash4096_pnm_residual/`（`full` = block 4096；`full_flash2048` 是 block 2048 的对照） |

32K 原来用 TP2 × PP16（32 个 device）。改成 PP8 后 block 时间、每 device 吞吐和能耗都不变（两者每个 block 都有一次 PP handoff），只有 device 数和系统吞吐减半；PP16 的结果已删除。16K 也比较过 TP1 × PP16 和 TP1 × PP8，结论相同（每 device 吞吐、block 时间和能耗完全一样），选了 PP8。

两组都在 PCIe 5.0 和 PCIe 7.0 下给出。仿真本身在 PCIe 6 下运行；PCIe 只影响 CXL，PCIe 5/7 的结果由同一次仿真换算得到（见 4.3）。

---

## 1. 配置与范围

| 项 | 值 |
|---|---|
| 模型 | Llama2-70B（80 层），decode-only |
| 精度 / 阵列 | FP8，systolic array 4×32（一个 256-bit word 放 32 个 FP8 元素） |
| 并行 | 见上表；KV-head TP |
| Batch | 每个 microbatch 4 个请求 |
| 序列 | 实际 decode 长度 = context window − 3584 / 2：8K → 6,400；16K → 14,592；32K → 30,976；64K → 63,744 |
| DRAM | LPDDR4，每 device 32 channel × x16，每 channel 8 bank，prefetch16，32 B/burst，nBL=8，内部 MAC nCCD=2，tCK=625 ps；功耗标签用 LPDDR4X |
| PNM | 每 device 32 个 32-lane 向量加速器（每 channel 一个），2 GHz；共享 4 MiB Shared Buffer，32 个 SRAM 端口（见 3.1） |
| CXL | 每个 device 固定 4 条 PCIe lane（`--pcie-lanes-per-device`，总数 = 4 × device 数）；one-shot all-to-all（见第 4 节） |

所有 latency 的口径都是**一个代表 TP rank、一个 Transformer block、一个 batch group**，没有乘 80 层，也没有乘 TP。周期数统一按 625 ps 的 DRAM 时钟换算。

---

## 2. 优化配置

结果由 [`run_sota_results.py`](cent_simulation/scripts/run_sota_results.py) 的 `full` variant 生成：

| 优化 | flag | 实现方式 |
|---|---|---|
| EWMUL_PNM | `--EWMUL_PNM` | RMSNorm、RoPE 和 FFN 里的逐元素乘法放到 PNM；systolic 路径会强制打开 |
| FlashAttention | `--flash-attention`（block 按 TP 选，见下表） | 对每个请求把序列切成 block，每个 block 先做 QK 再做 SV。score 只放在 Shared Buffer，不生成 `W_MEM`/`R_MEM`，Score Transfer 为 0 |
| Pipelined Softmax | `--pipelined-softmax` | 解析模型：`exposure = min(8192, seqlen) / seqlen`，`8192 = 32 lane × 32 channel × 8 bank`。8K 下 exposure = 100%（序列比启动窗口短），32K 下 26.4%，64K 下 12.9%。能耗仍按完整 Softmax 计算 |
| Channel concurrency | `--channel-concurrency` | AiM host/frontend 的 `issue_width` 设为 32：每个 channel 每周期最多接收一个请求；`WR_GB` 仍是 multicast；DRAM timing、backpressure 和 SYNC/EOC 都保留 |

**block 大小和 lane 利用率**：一个 local KV head 分到 `256 bank / local KV head 数` 个 bank，每个 bank 一次 QK 并行处理 32 个 token。

| 配置 | local KV head | 每个 KV head 的 bank | 一次 QK 覆盖的 token | block 2048 时 lane 利用率 | **默认 block** |
|---|---:|---:|---:|---:|---:|
| TP1 | 8 | 32 | 1,024 | 100%（每 bank 64 token，2 个 wave） | **2048** |
| TP2 | 4 | 64 | 2,048 | 100%（每 bank 32 token） | **2048** |
| TP4 | 2 | 128 | 4,096 | 50%（每 bank 16 token） | **4096** |

runner 的默认 block = max(2048, 一次 QK 覆盖的 token)，保证每个 bank 的 32 条 lane 都用上（`run_sota_results.py::default_flash_block`，可用 `--flash-block` 覆盖）。

64K 用 block 2048（对照 `full_flash2048`）时只用一半 lane，QK 的 wave 数翻倍，QK 是 183,896 周期；block 4096 下降到 90,348 周期，PIM 总周期少 22.8%（见 7.1）。SB 容量够：4096 × 16 head × 2 B = 128 KB。

---

## 3. 单缓冲结果是怎么算出来的

```
run_sim.py --generate_trace   → 生成 host trace + .systolic.json sidecar
           --simulate_trace   → Ramulator AiM（单缓冲）仿真，记录 32 个 channel 的命令（*.cmd.chN）
           --update_csv       → PIM 时间取仿真结果，PNM/CXL 用解析公式，写入 simulation_results.csv
cycle_breakdown.analyze_case  → 把时间拆到各个 kernel，输出 breakdown.json / csv / summary.txt
```

- **PIM（仿真）**：每个 GEMV 的 trace 是 `WR_BIAS` → `WR_GB` → `MAC_ABK`（fill、reduction、drain）→ `RD_MAC`。block 的 PIM 总周期直接取 Ramulator log。AiM 的 `ISR_EWADD` 不做任何事，residual add 在 PNM 计时。
- **PNM（解析，`run_sim.py::calculate_acc_latency`）**：RMSNorm ×2、RoPE、Softmax（按 exposure 计入）、FFN activation、Q/KV/SV reduction、FlashAttention merge、residual add ×2。参数见 3.1。
- **kernel 拆分（[`cycle_breakdown.py`](cent_simulation/cycle_breakdown.py)，`legacy_tp_issued_gap_v1`）**：按 sidecar 的 MAC 工作量给 terminal 命令打标签，在参考 channel ch15 上把相邻命令的间隔记到后一条命令所属 kernel。拆出来的数**包含等待时间**，是估算归因。拆完会对账：各 kernel 之和等于 PIM 总周期；PIM + PNM + CXL 等于 block latency。
- **summary 里的 kernel 明细**：每个 kernel 一段，写出占 block 的比例、Total / PIM / PNM / CXL 等效周期，以及 PIM 按命令分组的周期（MAC8、WR_GB、WR_MAC、RD_MAC、ACT / PRE、TMOD / CAS 等，分组和 kernel CSV 相同，加起来等于该 kernel 的 PIM）。单缓冲的在 `full/summary.txt`；double buffering 目录的 `full_flash2048/summary.txt` 里同时有该 PCIe 代数下单缓冲和 `dependency_aware` 两份。

### 3.1 PNM 加速器

| 参数 | flag | 默认值 |
|---|---|---|
| 每次向量操作处理的元素数（lane） | `--pnm-lanes` | 256 bit / 元素位宽：FP8 为 32，BF16 为 16 |
| 每 device 加速器数量 | `--pnm-accelerators` | 每 channel 一个（32） |
| Shared Buffer 端口数 | `--parallel-sram` | 32 |

- 每次向量操作的周期：读 SRAM 需要 `加速器数 / 端口数` 个周期。默认 EXP = 1 次读 + 11 + 1 次写 = 13；VEC_ADD / VEC_MUL = 2 次读 + 1 + 1 次写 = 4。
- 所有 PNM 工作量按 `元素数 / lane / 加速器数` 均摊。Softmax 每个 lane 组做 EXP + ADD + 2 × MUL = 25 个周期，每个 head 再加 RISC-V 的 18.25 个周期（8 个核）。
- 32K 下，一个 2048-token block、一个请求有 65,536 个 score；每个加速器分到 2,048 个 score（256 token × 8 query head），32 lane 下是 64 轮 × 25 个周期 = 0.8 µs，比 PIM 算这个 block 的 QK（约 0.9 µs）快。
- RoPE 主要是 RISC-V 的逐元素时间（每个元素 3/8 个周期），不随 lane 和加速器数量变化。
- PNM 能耗模型按每 16 个元素一次操作计，lane 宽度只改变延迟，不改变能耗。

---

## 4. CXL 模型（[`cxl_latency.py`](cent_simulation/cxl_latency.py)）

整个仿真器只有这一套 CXL 模型；原来的 gather + broadcast 模型（50% multicast 折扣、BF16）已经删除。

### 4.1 参数

| 项 | 值 |
|---|---|
| 每条 lane 带宽（单方向，GB/s = 10⁹ B/s） | PCIe 5：4；PCIe 6：8；PCIe 7：16（`--pcie-gen`，默认 6） |
| 每 device 带宽 | lane 数 × 每 lane 带宽；点对点，没有 multicast 折扣 |
| 线上精度 | 跟仿真精度（`--precision`）：FP8 每个值 1 B，BF16 2 B。本文结果都是 FP8 |
| flit 打包 | 256 B flit 装 192 B 有效载荷（3 × 64 B），即 1.33× 开销 |
| 固定开销 | 每次传输（TP 和 PP）180 ns = 4 个 CXL port × 25 + 2 段 PCIe × 30 + 1 个 switch × 20 |

### 4.2 每个 block 的通信

- **TP all-reduce**（O 之后和 Down 之后各一次，TP1 没有）：one-shot all-to-all。每个 rank 同时把完整的 partial 向量发给其余 TP−1 个 rank（全双工），在本地相加。时间由入口决定：
  `180 ns + (TP−1) × wire_bytes / 带宽`
- **PP handoff**（每个 block 一次）：`180 ns + wire_bytes / 带宽`
- 一个向量 = 8192 × batch 4 = 32,768 个值；FP8 下是 171 个 flit = 43,776 B（BF16 是 342 个 flit）。
- 能耗按有效载荷计：每个 rank 每 block 发 2 × (TP−1) × 32,768 个值；PP handoff 32,768 个值；每个值的 bit 数跟仿真精度；4.4 pJ/bit，不随 PCIe 代数变化。
- 原 CENT 的非 KV-head TP 映射（`simulation.sh`）也用这套模型：每对 gather + broadcast 改成一次 one-shot all-to-all all-gather。

### 4.3 PCIe 代数的换算

PCIe 只改变 CXL，不改变 PIM 和 PNM。仿真在 PCIe 6 下跑一次，[`cxl_rebase.py`](cent_simulation/scripts/cxl_rebase.py) 先用同一套公式复现 source 的 CXL（对不上就报错），再按目标代数重算单缓冲和双缓冲两列的 CXL。也可以直接用 `run_sota_results.py --pcie-gen 5|7` 重新仿真，结果相同，但每个 output root 只能放一种代数。

---

## 5. Double buffering（解析估算）

方法 `analytical_wrgb_issue_gap_credit_v2`，见 [`analyze_legacy_double_buffering.py`](cent_simulation/scripts/analyze_legacy_double_buffering.py)。适用于 FP8 SA4×32、FlashAttention 和 channel concurrency 打开、SV 为 paired-KV-head 布局的 KV-head TP（TP1/2/4）。

- **硬件假设**：每个 channel 有两块独立的 2 KiB GB，在一块上做 MAC 的同时往另一块写下一份输入。
- **load wave**：一条 host `WR_GB` 展开出来的全部 burst，加上随后使用这块 GB 的 MAC。
- **能藏住多少写入**：`L_i = bursts_i × 8`，`C_(i-1) = MAC8_count_(i-1) × 2`；输入已可用时 `hidden_i = min(L_i, C_(i-1))`，第一轮或输入未就绪时为 0。
- **输入什么时候可用（`dependency_aware`，主结果）**：同一累加 phase 内的下一个 tile/wave、Q → KV、SV → 下一个 QK 可以提前写；MEM、SYNC、EWADD、AF 边界，QK → SV（要等整块 QK 和 Softmax），以及其他依赖 PNM 或 collective 的切换不行。
- **账本扣减**：对能藏住的 burst，每个扣 `min(被覆盖的名义时间, 该 WRGB 原 issue gap)`；CASWRGB 和所有非 WRGB 的 PIM、PNM、CXL 不动。新 PIM 周期 = 原 PIM 周期 − 扣掉的 WRGB。
- **`ideal_ready_inputs`**：去掉输入可用性限制，只作为乐观上界。
- 后续两步：[`separate_legacy_cxl.py`](cent_simulation/scripts/separate_legacy_cxl.py) 把 O/Down 里的 TP CXL 拆成单独的行（总量不变）；[`postprocess_legacy_double_buffering.py`](cent_simulation/scripts/postprocess_legacy_double_buffering.py) 换算吞吐：`token_ms = block_us × 80 / 1000`，`系统 tok/s = 1000 × PP × batch / token_ms`，每 device 再除以 TP × PP。

---

## 6. 怎么复现

在仓库根目录、装好依赖的 conda 环境中执行：

```bash
# 0) 编译 AiM（-j4 到 -j8；-j32 曾因 exit 137 失败）
cd aim_simulator/build && cmake .. && make -j6 && cd ../..

# 1) 单缓冲仿真（PCIe 6），输出到 output/sota_results/<case>（默认路径）
cd cent_simulation
python scripts/run_sota_results.py --variants full --tp 1 --pp 8 --context-window 8192
python scripts/run_sota_results.py --variants full --tp 1 --pp 8 --context-window 16384
python scripts/run_sota_results.py --variants full                                  # 默认：TP2 PP8 32K
python scripts/run_sota_results.py --variants full full_flash2048 --tp 4 --pp 4 --context-window 65536
cd ..

# 2) Double buffering + 换算到 PCIe 5 / 7（依赖第 1 步的报告、host trace、sidecar 和 ch15 命令记录）
for c in TP1_PP8_8K_batch4_flash2048 TP1_PP8_16K_batch4_flash2048 TP2_PP8_32K_batch4_flash2048 TP4_PP4_64K_batch4_flash4096; do
  R=cent_simulation/output/sota_results/LPDDR4_nCCD2_${c}_pnm_residual
  SUP=""; [ -e "$R/full_flash2048/breakdown.json" ] && SUP="--supplement-source $R/full_flash2048/breakdown.json"
  for g in 5 7; do
    OUT="$R/double buffering pcie$g"
    python cent_simulation/scripts/analyze_legacy_double_buffering.py --source "$R/full/breakdown.json" $SUP \
      --pcie-gen $g --plot --output-dir "$OUT"
    python cent_simulation/scripts/separate_legacy_cxl.py --source-root "$OUT"
    python cent_simulation/scripts/postprocess_legacy_double_buffering.py --source-root "$OUT"
  done
done

# 3) 能耗和功耗（写入每个结果目录的 energy_power.csv）
cd cent_simulation && python scripts/sota_energy_power.py && cd ..

# 4) 测试
cd cent_simulation && python -m unittest discover -s tests
```

- runner 的 `--tp`、`--pp`、`--context-window`、`--seqlen`（默认 window − 1792）、`--batch`、`--pcie-lanes-per-device`（默认 4）决定 case；device 数 = TP × PP。
- runner 对源码（含 `cxl_latency.py`）、二进制和 case 参数计算指纹，结果放在以内容哈希命名的子目录里。
- 测试要在 `cent_simulation/` 目录下跑，当前 160 个全部通过。
- 只看 CXL 随 PCIe 代数、PP 级数的变化：`python cent_simulation/scripts/cxl_pcie_sweep.py --pp-stages 8`。

**主要输出文件**（每个结果目录下）：

| 文件 | 内容 |
|---|---|
| `comparison.csv`、`summary.txt`、`full/summary.txt`、`full/breakdown.json` | 单缓冲仿真（PCIe 6）的总计、kernel 明细和 kernel 卡片 |
| `full/<hash>/simulation_results.csv` | run_sim 输出，含能耗和 `PCIe generation`、`PCIe lanes per device`、`PNM lanes`、`PNM accelerators` 列 |
| `double buffering pcie{5,7}/full_flash{2048,4096}/summary.txt` | 该代数下单缓冲 vs 双缓冲，以及两份 kernel 卡片（64K 有两个 block 大小） |
| `double buffering pcie{5,7}/full_flash2048/*_kernels.csv`、`*_waves.csv` | 每个 kernel、每个 wave 的明细 |
| `double buffering pcie{5,7}/cxl_separated/`、`throughput/` | CXL 拆分视图、吞吐 |

---

## 7. 结果（每个 block、每个 batch group）

### 7.1 每 device 吞吐

| Context | 并行 | block | PCIe | 单缓冲 tok/s/device | **double buffering tok/s/device** | 提升 | 系统 tok/s（单 → 双） |
|---|---|---:|---|---:|---:|---:|---:|
| 8K | TP1 × PP8 | 2048 | 5 | 123.92 | **150.58** | +21.51% | 991.4 → 1204.7 |
| 8K | TP1 × PP8 | 2048 | 7 | 124.56 | **151.52** | +21.65% | 996.5 → 1212.2 |
| 16K | TP1 × PP8 | 2048 | 5 | 106.42 | **128.81** | +21.04% | 851.4 → 1030.5 |
| 16K | TP1 × PP8 | 2048 | 7 | 106.89 | **129.50** | +21.15% | 855.1 → 1036.0 |
| 32K | TP2 × PP8 | 2048 | 5 | 77.87 | **93.31** | +19.82% | 1245.9 → 1492.9 |
| 32K | TP2 × PP8 | 2048 | 7 | 79.39 | **95.50** | +20.29% | 1270.3 → 1528.0 |
| 64K | TP4 × PP4 | 4096 | 5 | 51.93 | **61.13** | +17.71% | 830.9 → 978.1 |
| 64K | TP4 × PP4 | 4096 | 7 | 55.23 | **65.75** | +19.05% | 883.7 → 1052.0 |
| 64K（对照） | TP4 × PP4 | 2048 | 5 | 41.25 | 48.68 | +18.01% | 659.9 → 778.8 |
| 64K（对照） | TP4 × PP4 | 2048 | 7 | 43.30 | 51.56 | +19.08% | 692.8 → 825.0 |

64K 下 block 4096 比 2048：double buffering 的每 device 吞吐 PCIe 5 下 +25.6%，PCIe 7 下 +27.5%；能耗从 490.71 降到 385.14 mJ/token。

### 7.2 Block 时间

| Context | block | PCIe | 场景 | PIM 周期 | PNM (µs) | CXL (µs) | Block (µs) | Block 周期 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| 8K | 2048 | 5 | 单缓冲 | 591,452 | 30.901 | 2.916 | 403.475 | 645,560 |
| 8K | 2048 | 5 | **double buffering** | **477,154** | 30.901 | 2.916 | **332.039** | **531,262** |
| 8K | 2048 | 7 | 单缓冲 | 591,452 | 30.901 | 0.864 | 401.423 | 642,277 |
| 8K | 2048 | 7 | **double buffering** | **477,154** | 30.901 | 0.864 | **329.987** | **527,979** |
| 16K | 2048 | 5 | 单缓冲 | 690,304 | 35.479 | 2.916 | 469.835 | 751,736 |
| 16K | 2048 | 5 | **double buffering** | **559,622** | 35.479 | 2.916 | **388.159** | **621,054** |
| 16K | 2048 | 7 | 单缓冲 | 690,304 | 35.479 | 0.864 | 467.783 | 748,453 |
| 16K | 2048 | 7 | **double buffering** | **559,622** | 35.479 | 0.864 | **386.107** | **617,771** |
| 32K | 2048 | 5 | 单缓冲 | 470,825 | 18.033 | 8.748 | 321.046 | 513,674 |
| 32K | 2048 | 5 | **double buffering** | **385,838** | 18.033 | 8.748 | **267.929** | **428,687** |
| 32K | 2048 | 7 | 单缓冲 | 470,825 | 18.033 | 2.592 | 314.890 | 503,824 |
| 32K | 2048 | 7 | **double buffering** | **385,838** | 18.033 | 2.592 | **261.773** | **418,837** |
| 64K | 4096 | 5 | 单缓冲 | 338,275 | 9.579 | 19.692 | 240.693 | 385,108 |
| 64K | 4096 | 5 | **double buffering** | **280,328** | 9.579 | 19.692 | **204.476** | **327,161** |
| 64K | 4096 | 7 | 单缓冲 | 338,275 | 9.579 | 5.328 | 226.329 | 362,126 |
| 64K | 4096 | 7 | **double buffering** | **280,328** | 9.579 | 5.328 | **190.112** | **304,179** |
| 64K（对照） | 2048 | 5 | 单缓冲 | 438,052 | 9.581 | 19.692 | 303.055 | 484,889 |
| 64K（对照） | 2048 | 5 | double buffering | 364,042 | 9.581 | 19.692 | 256.799 | 410,879 |
| 64K（对照） | 2048 | 7 | 单缓冲 | 438,052 | 9.581 | 5.328 | 288.691 | 461,906 |
| 64K（对照） | 2048 | 7 | double buffering | 364,042 | 9.581 | 5.328 | 242.435 | 387,896 |

- PIM 和 PNM 在所有 PCIe 代数下相同。`ideal_ready_inputs` 上界在各目录的 `comparison.csv` 里。
- 能耗、功耗和 tokens/J 见 7.5。

PNM 各项（µs，4 个请求；64K 两种 block 只有 merge 不同）：

| 项 | 8K | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| Softmax（计入部分） | 22.336（完整 22.336，exposure 100%） | 26.911（完整 47.936，56.1%） | 13.109（完整 49.568，26.4%） | 6.475（完整 50.384，12.9%） |
| RoPE Q + K | 6.984 | 6.984 | 3.492 | 1.746 |
| SV reduction | 0.448 | 0.448 | 0.480 | 0.496 |
| RMSNorm ×2 | 0.508 | 0.508 | 0.508 | 0.508 |
| FFN activation | 0.448 | 0.448 | 0.224 | 0.112 |
| 其余（residual、Q/KV reduction、merge） | 0.178 | 0.180 | 0.220 | 0.244 |
| **合计** | **30.901** | **35.479** | **18.033** | **9.581** |

### 7.3 CXL 明细（µs，每 device 4 条 lane）

| Context | PCIe | 每 device 带宽 | O 之后 all-reduce | Down 之后 all-reduce | PP handoff | 合计 |
|---|---|---:|---:|---:|---:|---:|
| 8K（TP1） | 5 | 16 GB/s | — | — | 2.916 | 2.916 |
| 8K（TP1） | 7 | 64 GB/s | — | — | 0.864 | 0.864 |
| 16K（TP1） | 5 | 16 GB/s | — | — | 2.916 | 2.916 |
| 16K（TP1） | 7 | 64 GB/s | — | — | 0.864 | 0.864 |
| 32K（TP2） | 5 | 16 GB/s | 2.916 | 2.916 | 2.916 | 8.748 |
| 32K（TP2） | 7 | 64 GB/s | 0.864 | 0.864 | 0.864 | 2.592 |
| 64K（TP4） | 5 | 16 GB/s | 8.388 | 8.388 | 2.916 | 19.692 |
| 64K（TP4） | 7 | 64 GB/s | 2.232 | 2.232 | 0.864 | 5.328 |

TP4 每次 all-reduce 要收 3 个 peer 的向量，所以 64K 的 CXL 是 32K 的两倍多。

### 7.4 各 kernel 的 PIM 周期（单缓冲 → double buffering，和 PCIe 代数无关）

| kernel | 8K 单 | 8K 双 | 16K 单 | 16K 双 | 32K 单 | 32K 双 | 64K 单 | 64K 双 | 64K@2048 对照 单 | 64K@2048 对照 双 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q | 39,402 | 31,834 | 39,402 | 31,834 | 19,640 | 16,112 | 9,764 | 8,249 | 9,764 | 8,249 |
| K/V | 9,775 | 7,748 | 9,775 | 7,748 | 4,750 | 3,733 | 2,411 | 1,899 | 2,411 | 1,899 |
| QK | 40,225 | 33,348 | 86,505 | 71,429 | 91,929 | 75,925 | 90,348 | 74,092 | 183,896 | 151,584 |
| SV | 42,509 | 36,109 | 95,451 | 80,859 | 103,419 | 87,973 | 109,399 | 93,463 | 115,255 | 99,319 |
| O | 39,032 | 31,464 | 39,173 | 31,605 | 19,555 | 16,027 | 9,455 | 7,947 | 9,459 | 7,951 |
| Up/Gate | 276,929 | 220,839 | 276,317 | 220,234 | 158,391 | 126,555 | 79,332 | 63,677 | 79,157 | 63,495 |
| Down | 138,450 | 110,682 | 138,007 | 110,239 | 68,847 | 55,219 | 34,725 | 28,160 | 34,725 | 28,160 |
| KV Update | 1,377 | 1,377 | 1,377 | 1,377 | 1,403 | 1,403 | 772 | 772 | 772 | 772 |
| FFN Activation | 309 | 309 | 309 | 309 | 211 | 211 | 162 | 162 | 162 | 162 |
| 未归属 / 背景 / 尾部 | 3,444 | 3,444 | 3,988 | 3,988 | 2,680 | 2,680 | 1,907 | 1,907 | 2,451 | 2,451 |
| **合计** | **591,452** | **477,154** | **690,304** | **559,622** | **470,825** | **385,838** | **338,275** | **280,328** | **438,052** | **364,042** |
| 藏掉的 WRGB | | 114,298 |  | 130,682 | | 84,987 | | 57,947 | | 74,010 |

8K、16K（TP1）的时间主要在 FFN（Up/Gate + Down 占 60–70%）；64K（TP4）主要在 attention（QK + SV 占 68%）。

### 7.5 能耗、功耗和 tokens/J

由 [`sota_energy_power.py`](cent_simulation/scripts/sota_energy_power.py) 生成，写在每个结果目录的 `energy_power.csv`。

| Context | block | PCIe | 场景 | mJ/token | tokens/J | tok/s/device | W/device | 系统 W |
|---|---:|---|---|---:|---:|---:|---:|---:|
| 8K | 2048 | 5 | 单缓冲 | 167.23 | 5.980 | 123.92 | 20.72 | 165.8 |
| 8K | 2048 | 5 | **double buffering** | **165.27** | **6.051** | 150.58 | 24.89 | 199.1 |
| 8K | 2048 | 7 | 单缓冲 | 167.23 | 5.980 | 124.56 | 20.83 | 166.6 |
| 8K | 2048 | 7 | **double buffering** | **165.27** | **6.051** | 151.52 | 25.04 | 200.3 |
| 16K | 2048 | 5 | 单缓冲 | 196.86 | 5.080 | 106.42 | 20.95 | 167.6 |
| 16K | 2048 | 5 | **double buffering** | **194.63** | **5.138** | 128.81 | 25.07 | 200.6 |
| 16K | 2048 | 7 | 单缓冲 | 196.86 | 5.080 | 106.89 | 21.04 | 168.3 |
| 16K | 2048 | 7 | **double buffering** | **194.63** | **5.138** | 129.50 | 25.20 | 201.6 |
| 32K | 2048 | 5 | 单缓冲 | 261.64 | 3.822 | 77.87 | 20.37 | 326.0 |
| 32K | 2048 | 5 | **double buffering** | **258.75** | **3.865** | 93.31 | 24.14 | 386.3 |
| 32K | 2048 | 7 | 单缓冲 | 261.64 | 3.822 | 79.39 | 20.77 | 332.4 |
| 32K | 2048 | 7 | **double buffering** | **258.75** | **3.865** | 95.50 | 24.71 | 395.4 |
| 64K | 4096 | 5 | 单缓冲 | 385.14 | 2.596 | 51.93 | 20.00 | 320.0 |
| 64K | 4096 | 5 | **double buffering** | **381.19** | **2.623** | 61.13 | 23.30 | 372.8 |
| 64K | 4096 | 7 | 单缓冲 | 385.14 | 2.596 | 55.23 | 21.27 | 340.3 |
| 64K | 4096 | 7 | **double buffering** | **381.19** | **2.623** | 65.75 | 25.06 | 401.0 |
| 64K（对照） | 2048 | 5 | 单缓冲 | 490.71 | 2.038 | 41.25 | 20.24 | 323.8 |
| 64K（对照） | 2048 | 5 | double buffering | 485.65 | 2.059 | 48.68 | 23.64 | 378.2 |
| 64K（对照） | 2048 | 7 | 单缓冲 | 490.71 | 2.038 | 43.30 | 21.25 | 340.0 |
| 64K（对照） | 2048 | 7 | double buffering | 485.65 | 2.059 | 51.56 | 25.04 | 400.6 |

- **怎么算的**：能耗是整个系统每输出一个 token 的能量（80 层、所有 TP rank）；tokens/J = 1000 / mJ per token；功耗 = 能耗 × 系统吞吐，W/device 再除以 device 数。
- **单缓冲**：从 Ramulator log 重建 run_sim 的逐项能耗，和 run_sim 的 `Token energy (mJ)` 完全一致。
- **double buffering（解析估算）**：每个操作的动态能耗不变；随时间计的能耗（所有 `*_STT` 漏电和 DRAM 的 `ACT_STBY` / `PRE_STBY`，run_sim 里按 PIM 时间计）按新 PIM 周期 / 原 PIM 周期缩放；再加上每个 channel 第二块 GB 的漏电。
- **PCIe 代数不影响能耗**（CXL 按 bit 计能耗），只通过吞吐影响功耗。
- 随时间计的能耗只占约 6%（主要是 DRAM ACT standby），所以 double buffering 的能耗只降约 1%，tokens/J 只升约 1%；功耗则随吞吐上升约 20%，每 device 约 23–25 W。
- 64K 的 block 4096 比 2048 每 token 少 21.5% 能耗，因为 QK 的 MAC 和 GB 写入减半。

---

## 8. 已知限制

1. **Double buffering 是解析估算，不是仿真。** 没有重新计算 refresh、资源争用和能耗，结果目录里也没有 build manifest。`double-buffering` 分支上的 `AiMAsync` 仿真（`--pim-double-buffering`）是另一套实现，这里没有用。
2. **CXL 和计算完全不重叠。** 固定开销 180 ns 沿用原 CENT 的 port/PCIe/switch 估计，TP 和 PP 传输相同。
3. **每个 device 固定 4 条 PCIe lane。**
4. **PCIe 能耗不分代数**，统一 4.4 pJ/bit，按有效载荷计，不含 flit 开销。
5. **Pipelined Softmax 是比例模型。** 启动窗口 8192 是整个 device 一轮的产出量，不随 PNM 速度变化；8K 的序列比它短，所以 Softmax 完全不隐藏。online softmax 的修正项（`O` 的 rescale、每个 block 的 max/sum）没有建模。
6. **block 按 lane 利用率选**（见第 2 节）。更大的 block 会让 online softmax 的 merge 次数变少，但 merge 的修正项本来就没有建模，所以这里只反映 PIM 一侧的收益。
7. **没有检查容量。** 每个 device 是否放得下对应的权重和 KV cache（例如 8 device 放下整个 70B 模型）不在这份结果里。
8. **能耗模型里静态功耗只在 PIM 时间内计。** PNM 和 CXL 的时间不计漏电和 DRAM standby，所以功耗偏低；double buffering 的能耗是解析估算。
9. **kernel 拆分是估算。** 包含等待时间，只基于 ch15 一个 channel；O、Up/Gate、Down 在不同 run 之间几百周期的差异来自归因波动。
10. **结果没有进 git。** `output/sota_results/*` 和 `trace/sota_results/*` 都被 gitignore，只在本地。baseline、block 1024 等对照组没有保留，需要时用 `run_sota_results.py --variants baseline full_flash1024 --output-root <新目录>` 重跑。
