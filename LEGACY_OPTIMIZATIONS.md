# Legacy 优化结果：baseline、full@2048/1024 与 double buffering

本文说明 `main`（`a692619`）上这一组 Llama2-70B 实验的实现方式和复现命令：

- **baseline**：单缓冲 AiM，FlashAttention、pipelined Softmax、channel concurrency 都关闭。
- **full@2048**：四项优化全开，FlashAttention block = 2048（现在的默认值）。
- **full@1024**：同上，但 block = 1024，作为对照组。
- **double buffering**：在 full@2048 和 full@1024 上做的**解析估算**，没有重跑仿真。

结果目录（被 gitignore，只在本地）：
`cent_simulation/output/legacy_optimizations/LPDDR4_nCCD2_TP2_32K_batch4_flash2048_pnm_residual/`

block 1024 的旧归档在 `cent_simulation/output/legacy_optimizations/LPDDR4_nCCD2_TP2_32K_batch4/`，保持原样，用作对比。

---

## 1. 共同配置与范围

| 项 | 值 |
|---|---|
| 模型 | Llama2-70B，decode-only |
| 精度 / 阵列 | FP8，systolic array 4×32（一个 256-bit word 放 32 个 FP8 元素） |
| 并行 | 32 个 device = TP2 × PP16；KV-head TP |
| Batch | 每个 microbatch 4 个请求 |
| 序列 | context window 32,768；实际 decode 长度 30,976 |
| DRAM | LPDDR4，32 channel × x16，每 channel 8 bank，prefetch16，32 B/burst，nBL=8，内部 MAC nCCD=2，tCK=625 ps；功耗标签用 LPDDR4X |
| PNM | 16-lane 向量单元，2 GHz，32 个 SRAM bank，4 MiB Shared Buffer |
| CXL | 144 条 PCIe lane / 32 个 device = 每 device 4 条，8 GB/s/lane，multicast 打 50% 折扣，每跳固定 180 ns |

所有 latency 的口径都是**一个代表 TP rank、一个 Transformer block、一个 batch group**，没有乘 80 层，也没有乘 TP。

---

## 2. 三个 variant 的定义

三个 variant 都由 [`cent_simulation/scripts/run_legacy_optimization_case.py`](cent_simulation/scripts/run_legacy_optimization_case.py) 里的 `VARIANTS` 生成。公共参数完全相同，区别只在下表的 flag：

| Variant | EWMUL_PNM | FlashAttention | Pipelined Softmax | Channel concurrency | 额外 flag |
|---|---|---|---|---|---|
| `baseline` | ON | OFF | OFF | OFF | — |
| `full`（@2048） | ON | ON，block 2048（默认值） | ON | ON | `--channel-concurrency --flash-attention --pipelined-softmax` |
| `full_flash1024` | ON | ON，block 1024 | ON | ON | 同上，再加 `--flash-attention-block-size 1024` |

systolic 路径会强制打开 EWMUL_PNM，所以 baseline **并不是**四项全关。

---

## 3. 单缓冲结果是怎么算出来的

每个 variant 都依次经过四步：

```
run_sim.py --generate_trace   → 生成 host trace + .systolic.json sidecar
           --simulate_trace   → Ramulator AiM（单缓冲）仿真，记录 32 个 channel 的命令（*.cmd.chN）
           --update_csv       → PIM 时间取仿真结果，PNM/CXL 用解析公式，写入 simulation_results.csv
cycle_breakdown.analyze_case  → 把时间拆到各个 kernel，输出 breakdown.json / csv / summary.txt
```

### 3.1 PIM（仿真）

- 生成 trace：`Llama.py` 和 `TransformerBlock.py` 中 systolic KV-head TP 的 `*_only_trace` 路径。tensor 在 bank 上的布局由 [`tp_mapping.py`](cent_simulation/tp_mapping.py) 里的 `SystolicTPLayout` 决定。
- 每个 GEMV 的 trace 结构相同：先 `WR_BIAS` 初始化累加器，再 `WR_GB` 装入输入，然后 `MAC_ABK`（fill、reduction、drain 三段），最后 `RD_MAC` 读出结果。
- block 的 PIM 总周期直接取 Ramulator log 的值。
- 注意：AiM 里 `ISR_EWADD` 不做任何事（见 `aim_simulator/src/memory_system/impl/aim_DRAM_system.cpp`），所以 residual add 不在 PIM 里计时，见 3.2。

### 3.2 PNM（解析公式，见 `run_sim.py::calculate_acc_latency`）

| 项 | 公式要点 |
|---|---|
| RMSNorm ×2 | VEC_ADD/VEC_MUL 加 RISC-V 的 RMSNorm 周期 |
| RoPE Q/K | 每个元素的 RISC-V 周期，加 VEC_MUL |
| Softmax | 每 16 个 score 做 EXP(13) + VEC_ADD(4) + 2×VEC_MUL(8) 周期，再加每个 head 的 RISC-V max/sum。完整工作量 97.968 µs |
| FFN activation | `SiLU(W1)·W3`，用 VEC_MUL |
| Q/KV/SV reduction | 把多个 channel group 的 partial sum 加起来，用 VEC_ADD |
| FlashAttention merge | 每个 block 一次 VEC_ADD（`seqlen // block`） |
| **Residual add ×2**（新增） | `8192 / 16 / 32 × VEC_ADD = 64 cycles = 32 ns`，每个 batch 两次 |

residual add 只在 EWMUL_PNM 打开时计入，能耗也走 VEC_ADD 的路径。旧 CSV 没有 residual 列，postprocessor 照旧把它显示为"未建模"。

### 3.3 CXL（解析公式，见 [`cent_simulation/cxl_latency.py`](cent_simulation/cxl_latency.py)）

- **TP collective**：每个 block 有两次 reduction，分别在 O 之后和 Down 之后。每次 reduction 是一次 gather（helper → main）加一次 broadcast（main → helper），两者串行。
  - 一次传输 = 342 个 flit × 256 B / 16 GiB/s + 180 ns = 5.276 µs
  - 一次 reduction = 10.552 µs，O 和 Down 各计一份
- **PP handoff**：每个 block 一次向量传输，5.276 µs。
- `run_sim` 用的带宽单位是 **GiB/s**（除以 1024³），合计 CXL = 26.381 µs。

### 3.4 四项优化各自怎么实现

| 优化 | 实现方式 |
|---|---|
| EWMUL_PNM | RMSNorm、RoPE 和 FFN 里的逐元素乘法放到 PNM 做，按 3.2 的公式计时 |
| FlashAttention | `Vector_Matrix_Mul_flash_attention_systolic_tp_only_trace` 对每个请求把序列切成多个 block。每个 block 先对该段 K 做 QK，再对同一段 V 做 SV。score 只放在 Shared Buffer（2048 × 32 head × 2 B = 128 KB），不生成 `W_MEM`/`R_MEM`，所以 Score Transfer 为 0 |
| Pipelined Softmax | 只在解析层面处理：`exposure = startup_tokens / seqlen`，其中 `startup_tokens = 32 lane × 32 channel × 8 bank = 8192`，exposure = 26.4%。exposed 时间是 25.909 µs，能耗仍按完整 Softmax 计算 |
| Channel concurrency | AiM 的 host 和 frontend 的 `issue_width` 设为 32：每个 channel 每周期最多接收一个请求，32 个 channel 可以并行；`WR_GB` 仍然是 multicast；DRAM timing、backpressure 和 SYNC/EOC 都保留 |

**为什么 block 默认是 2048**：每个 local KV head 分到 32 × 8 / 4 = 64 个 bank，每个 bank 一次 QK 并行处理 32 个 token。因此一次 QK 正好覆盖 64 × 32 = **2048** 个 token。

- block = 1024 时，每个 bank 只用 16 条 lane。QK 要跑的轮数从 16 变成 31，QK 的 PIM 周期从约 92k 涨到 177k。
- block ≥ 2048 时，QK 恢复到 baseline 的水平。

### 3.5 把时间拆到各 kernel（[`cent_simulation/cycle_breakdown.py`](cent_simulation/cycle_breakdown.py)）

- **方法**：`legacy_tp_issued_gap_v1`。
  1. 按 sidecar 里 `command_phases` 记录的 MAC 工作量，解码 host trace，确定每条 terminal 命令属于哪个 kernel。
  2. 只看一个参考 channel（ch15），把这些标签按顺序对到实际发出的命令上。ACT、PRE、TMOD、CAS 这类前置命令归到下一条 terminal 所属的 kernel。
  3. 相邻两条命令之间的间隔，记到后一条命令名下。
- 这样拆出来的数**包含等待时间**，是估算归因，不是每个 kernel 独立执行的时间。
- 拆完会做对账：各 kernel 加起来必须等于 PIM 总周期；PIM + PNM + CXL 必须等于 CSV 里的 block latency。
- CXL 计入 O 和 Down 两行；K/V 和 Up/Gate 是融合的，没有拆开。

---

## 4. Double buffering 的实现（解析估算）

> 注意：`main` 上的 double buffering 是**在单缓冲结果上做的解析估算**，没有重跑 Ramulator。`double-buffering` 分支上的 `AiMAsync` 仿真（`--pim-double-buffering`）是另一套实现，本文的结果没有用它。

### 4.1 模型（[`cent_simulation/scripts/analyze_legacy_double_buffering.py`](cent_simulation/scripts/analyze_legacy_double_buffering.py)，方法 `analytical_wrgb_issue_gap_credit_v2`）

- **硬件假设**：每个 channel 有两块独立的 2 KiB GB。在一块 GB 上做 MAC 的同时，可以往另一块 GB 写入下一份输入。
- **load wave**：一条 host `WR_GB` 展开出来的全部 burst，加上随后使用这块 GB 的 MAC。它不是一个 query-head wave，也不是整个 kernel。
- **能藏住多少写入时间**：
  - 第 i 次写入的时间 `L_i = bursts_i × nBL(8)`；上一轮的计算时间 `C_(i-1) = MAC8_count_(i-1) × nCCD(2)`，包含 fill 和 drain。
  - 如果这一轮的输入已经可用，`hidden_i = min(L_i, C_(i-1))`；第一轮或者输入还没准备好时，`hidden_i = 0`。
- **输入什么时候算可用**（`dependency_aware`，主结果）：
  - 可以提前写入：同一个累加 phase 内、同一个 kernel 的下一个 tile 或 wave、Q → KV、SV → 下一个 QK。
  - 不能提前写入：遇到 MEM、SYNC、EWADD、AF 这类边界；QK → SV，因为要等整块 QK 和 Softmax 都完成；以及其他依赖 PNM 或 collective 的 kernel 切换。
- **从账本里扣多少（v2）**：对能藏住的那部分 burst，每个 burst 扣 `min(被覆盖的名义时间, 这个 WRGB 原来的 issue gap)`。CASWRGB 以及所有不是 WRGB 的 PIM、PNM、CXL 开销都不动。
  - 新的 PIM 周期 = 原 PIM 周期 − 扣掉的 WRGB 时间；每个 kernel 剩下的 WRGB 不能为负。
- **`ideal_ready_inputs`**：去掉所有"输入是否可用"的限制，只作为乐观上界参考，实际调度做不到。
- **CXL 单位**：用 [`legacy_cxl_units.py`](cent_simulation/scripts/legacy_cxl_units.py) 把 CXL 统一按 **GB/s**（10⁹）重新算。CXL 从 26.381 µs 变成 28.260 µs，单缓冲和双缓冲的对比列都用这个口径。

### 4.2 后续两步

- [`separate_legacy_cxl.py`](cent_simulation/scripts/separate_legacy_cxl.py)：把 O 和 Down 里的 TP CXL 拆成单独的行（`CXL TP All-Reduce (after O/Down)`），只改变展示方式，总量不变。输出到 `double buffering/cxl_separated/`。
- [`postprocess_legacy_double_buffering.py`](cent_simulation/scripts/postprocess_legacy_double_buffering.py)：按 run_sim 的 decode 吞吐公式换算成系统吞吐，输出到 `double buffering/throughput/`。
  - `token_ms = block_us × 80 / 1000`
  - `system tok/s = 1000 × PP × batch / token_ms`
  - 每个 device 的吞吐再除以 32

---

## 5. 怎么复现

以下命令都在 `cent` conda 环境中、从仓库根目录执行。

```bash
# 0) 编译 AiM。用 -j4 到 -j8；-j32 曾经因为 exit 137 失败
cd aim_simulator/build && cmake .. && make -j6 && cd ../..

# 1) 单缓冲三个 variant：生成 trace、仿真、更新 CSV、拆 kernel，并生成对比和 kernel_breakdown.csv
cd cent_simulation
python scripts/run_legacy_optimization_case.py \
  --output-root output/legacy_optimizations/LPDDR4_nCCD2_TP2_32K_batch4_flash2048_pnm_residual \
  --trace-root  trace/legacy_optimizations/LPDDR4_nCCD2_TP2_32K_batch4_flash2048_pnm_residual \
  --variants baseline full full_flash1024
cd ..

# 2) Double buffering 解析估算。依赖第 1 步的 full、full_flash1024 报告，
#    以及本地的 host trace、sidecar 和 ch15 命令记录
python cent_simulation/scripts/analyze_legacy_double_buffering.py --cxl-bandwidth-unit GB/s --plot
python cent_simulation/scripts/separate_legacy_cxl.py
python cent_simulation/scripts/postprocess_legacy_double_buffering.py
```

- **一定要指定新的 `--output-root`**。runner 会覆写输出根目录下的 `comparison.csv`、`summary.txt`、`BUILD_MANIFEST.json`；默认路径就是旧归档，直接跑会覆盖它。
- runner 会对源码和二进制计算指纹，结果放在以内容哈希命名的子目录里（例如 `full/92dc6eae47338b42/`），不会覆盖内容不同的已有结果。
- 跑完后检查 `cent_simulation/tests`：`python -m unittest discover -s cent_simulation/tests`，当前 147 个测试全部通过。

**主要输出文件**（都在结果目录下）：

| 文件 | 内容 |
|---|---|
| `comparison.csv`、`summary.txt` | 三个 variant 的总计，以及完整的 kernel 明细 |
| `kernel_breakdown.csv` | baseline 和 optimized 两组；每个 kernel 一行，列分成 PIM / PNM / CXL / Total 四组，两行表头 |
| `old_vs_new_comparison.txt` | 和 1024 旧归档逐 kernel 对比 |
| `<variant>/summary.txt`、`breakdown.json` | 每个 variant 单独的报告 |
| `double buffering/summary.txt`、`comparison.csv` | double buffering 主结果 |
| `double buffering/full_flash*/dependency_aware_kernels.csv`、`*_waves.csv` | 每个 kernel、每个 wave 的明细 |
| `double buffering/cxl_separated/`、`throughput/` | CXL 拆分视图、吞吐 |

---

## 6. 结果汇总（每个 block、每个 batch group）

### 6.1 单缓冲（仿真；CXL 用 run_sim 的 GiB/s 口径）

| | PIM 周期 | PIM (µs) | PNM (µs) | CXL (µs) | Block (µs) | 系统 tok/s | mJ/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧归档 baseline | 713,214 | 445.759 | 103.980 | 26.381 | 576.120 | 1388.6 | 299.11 |
| **baseline** | 713,214 | 445.759 | 104.236 | 26.381 | **576.376** | 1388.0 | 299.27 |
| 旧归档 full@1024 | 564,405 | 352.753 | 31.936 | 26.381 | 411.070 | 1946.1 | 311.05 |
| **full@1024** | 564,405 | 352.753 | 32.192 | 26.381 | **411.326** | 1944.9 | 311.20 |
| **full@2048** | 470,825 | 294.266 | 32.184 | 26.381 | **352.831** | 2267.4 | 261.76 |

- 新代码在相同配置下，PIM 周期和旧归档完全一致。旧归档写 README 时担心 fence 修复会改变结果，在这个 case 上并没有。
- 新 baseline 和 full@1024 比旧归档多出的 0.256 µs，正好是两次 residual add（2 × 32 ns × 4 batch）。
- full@2048 和 full@1024 比：block −14.2%，吞吐 +16.5%，能耗 −15.8%。和 baseline 比，block 时间减少了 **38.8%**。

主要 kernel 的 PIM 周期：

| kernel | baseline | full@1024 | full@2048 |
|---|---:|---:|---:|
| QK | 93,686 | 177,013 | 91,929 |
| SV | 94,126 | 111,635 | 103,419 |
| Score Transfer | 248,552 | 0 | 0 |
| Up/Gate | 157,727 | 157,871 | 158,391 |
| Down | 69,372 | 69,177 | 68,847 |

### 6.2 Double buffering（解析估算；CXL 用 GB/s 口径）

| case / 场景 | PIM 周期 | Block (µs) | 和单缓冲比 | 每 device tok/s |
|---|---:|---:|---:|---:|
| full@2048 单缓冲（GB/s） | 470,825 | 354.710 | — | 70.48 |
| **full@2048 dependency_aware** | **385,838** | **301.593** | **−14.97%** | **82.89（+17.6%）** |
| full@2048 ideal_ready_inputs | 369,203 | 291.196 | −17.91% | 85.85 |
| full@1024 单缓冲（GB/s） | 564,405 | 413.205 | — | 60.50 |
| full@1024 dependency_aware | 464,037 | 350.475 | −15.18% | 71.33 |
| full@1024 ideal_ready_inputs | 447,437 | 340.100 | −17.69% | 73.51 |

full@2048 `dependency_aware` 下各 kernel 的 PIM 周期，前后对比：

| kernel | 单缓冲 | 双缓冲 | 藏掉的 WRGB |
|---|---:|---:|---:|
| Q | 19,640 | 16,112 | 3,528 |
| K/V | 4,750 | 3,733 | 1,017 |
| QK | 91,929 | 75,925 | 16,004 |
| SV | 103,419 | 87,973 | 15,446（QK→SV 依赖保留了一部分 WRGB） |
| O | 19,555 | 16,027 | 3,528 |
| Up/Gate | 158,391 | 126,555 | 31,836 |
| Down | 68,847 | 55,219 | 13,628 |

---

## 7. 已知限制和口径差异

1. **两套结果的 CXL 单位不一样。** 单缓冲报告（`kernel_breakdown.csv` 和各 summary）用 GiB/s，CXL = 26.381 µs；double buffering 用 GB/s，CXL = 28.260 µs。所以 full@2048 的 block 时间在两处分别写成 352.831 和 354.710 µs，比较时要用同一个口径。`cxl_latency.py` 本身目前还是 GiB/s。
2. **CXL 模型偏保守。**
   - multicast 的 50% 带宽折扣也用在了 gather 上，而 gather 是点对点传输。
   - TP2 时 gather 和 broadcast 串行，没有考虑两边同时互发。
   - lane 数向下取整（4.5 → 4）。
   - CXL 和计算完全不重叠。
3. **Pipelined Softmax 是比例模型。**
   - 启动窗口 8192 是整个 device 所有 bank 一轮的产出量，拿来除以单个 head 的序列长度。
   - 没有考虑 Softmax 的处理速度（完整做一遍 98 µs）比 QK 产出 score 的速度（57 µs）慢。
   - online softmax 的修正项（`O` 的 rescale 和累加、每个 block 的 max/sum）没有建模；FlashAttention merge 的 block 数是向下取整。
4. **kernel 拆分是估算。** 拆出来的数包含等待时间，只基于 ch15 一个 channel；O、Up/Gate、Down 在不同 run 之间几百周期的差异来自归因波动。
5. **Double buffering 是解析估算，不是仿真。** 它没有重新计算 refresh、资源争用和能耗，结果目录里也没有 build manifest。
6. **结果没有进 git。** `output/legacy_optimizations/*` 和 `trace/legacy_optimizations/*` 都被 gitignore，三个 variant 的输出加起来约 400 MB，只在本地。旧归档 `LPDDR4_nCCD2_TP2_32K_batch4/` 是单独强制加进 git 的。
7. **分支已经分开。** `origin/double-buffering`（AiMAsync 仿真）缺 `main` 上的 `a692619`，所以没有 block 2048 默认值、PNM residual，也没有这些解析脚本；反过来，`main` 也没有 `double-buffering` 分支上的 5 个提交。
