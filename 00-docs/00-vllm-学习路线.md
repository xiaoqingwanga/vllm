# 熟悉 vLLM —— 四阶能力地图与 Level 3 学习路线

## 一、四阶能力地图

JD 里的"熟悉 vLLM"歧义很大，本质是两条正交能力轴：**理解轴**（能不能解释行为）和**改动轴**（能不能改变行为）。

```
① 用户级    vllm serve · 调 API                        ← 太浅，无壁垒
② 调优级    部署·调参·压测·看 metrics·debug OOM/性能     ← ★ 多数 JD 的"熟悉"落这
            + 能读 config.py / scheduler 解释行为
③ 二次开发  见下方两个分支                              ← 强候选人 vs 打勾候选人的分水岭
④ 引擎研发  写/改 CUDA·Triton kernel · 实现 PD 分离       ← 只有 kernel/引擎优化岗才要
```

**核心认知**：概念地图 ≠ 引擎动手，引擎动手 ≠ 会开发引擎。理解轴再高也跨不到改动轴——判据很硬：git 里有没有一个改过 vLLM 行为的 diff / 部署过多大规模的集群。

---

## 二、Level 3 的两个分支

③ 不是一条路，是两个分支，需求量差别很大：

```
③ 二次开发 ─┬─ ③a 改源码（scheduler / metric / 集成）   ← 偏研发岗
            └─ ③b 集群部署深度（并行 / 多机 / PD / 副本）  ← ★ 绝大多数"熟悉 vLLM"落这，且不碰源码
```

> 结论：**先走 ③b（部署/集群），它需求量最大且不用改源码；③a 作为可选加分。** ④ 不碰，属 kernel/引擎优化岗地盘。

---

## 路线 A —— ③a 二次开发（改源码）

判据：**git 里有一个改过 vLLM 行为的 commit + 量化的 before/after 数字。**

### 前置：走通"一个请求的一生"（1–2 天）
```
API 请求 → async_llm.py → EngineCore(core.py) step()
        → Scheduler.schedule()   ← 决定这一步跑哪些请求
        → Worker 前向 → update_from_output()  ← 记账/判停
        → OutputProcessor → 流式返回
```
必读：
- `vllm/v1/engine/core.py` `EngineCore.step()` —— 引擎心跳
- `vllm/v1/core/sched/scheduler.py` `Scheduler.schedule()` —— **最核心**，挑请求 / token budget / preempt
- `vllm/v1/core/sched/scheduler.py` `Scheduler.update_from_output()` —— 前向后记账判停

> 注：vLLM 迭代极快，锚点一律用**函数名**而非行号，行号很快会烂。
- 验证：能口述"一个请求为什么会从 RUNNING 被踢回 WAITING"（`_preempt_request()` + KV block 不够的分支）

### 能力 A：加一个 metric（1 天，先做，最快见效）
- 读：`vllm/v1/metrics/stats.py` + `vllm/v1/metrics/loggers.py`（Prometheus 定义）
- 模板：preemption 计数 vLLM **已经实现**（`loggers.py` 的 `counter_num_preempted_reqs`），把它当抄写模板，但自己加的指标必须换一个没有的，否则 diff 无意义。候选：请求从 WAITING 到首次被调度的等待步数分布 / 每步 KV block 分配失败次数
- 采集点在 `scheduler.py` 的 `Scheduler.make_stats()`
- 验证：`vllm serve` → 压请求 → `curl localhost:8000/metrics | grep 指标名`

### 能力 B：改一条调度逻辑（3–5 天，分水岭）
- 读透：`scheduler.py` 的 `Scheduler.schedule()` 全文 + `sched/interface.py` + `sched/request_queue.py`
- 改（选一）：调 preemption 触发条件 / 改一条 admission 逻辑 / 走 CLI 开关加调度变体
  （加 CLI 开关走 `SchedulerConfig` 加字段 → `arg_utils.py` 的 `get_kwargs` 自动生成 flag → scheduler 读它）
- 验证：**同组请求跑 `benchmarks/` 压测，记录改前 vs 改后的吞吐 / TTFT / P99**
- ⚠️ 预期管理：调度器已被高度优化，别指望拿到正向改善的数字。现实目标是**展示 trade-off**——比如"TTFT 降 X% 但吞吐掉 Y%"，并能解释为什么。面试证据的价值在于你理解权衡，不在于打败上游。

### 能力 C：集成（可选，偏平台岗）
- 把 `vllm/v1/engine/async_llm.py` 的 `AsyncLLM` 接进自写 FastAPI 路由，跑通多实例

---

## 路线 B —— ③b 集群部署深度（不改源码，需求最大）★

判据：**部署过多大规模、扛过多少并发、排过什么故障。** ① 是单卡起服务；③b 是"大模型跨多卡/多机跑起来，副本会扩，KV 不 OOM，掉卡能救"。

### 第 1 层：并行策略——单机多卡（2–3 天，必修）
- 配置：`vllm/config/parallel.py` 的 `tensor_parallel_size`(TP) / `pipeline_parallel_size`(PP) / `data_parallel_size`(DP)
- 读：`docs/serving/parallelism_scaling.md`
- 判据：能答"放不下一张卡用 TP 还是 PP？为什么 TP 通常不跨机？DP 和多副本区别？"

### 第 2 层：多机部署（2–3 天，必修）
- 读：`docs/serving/data_parallel_deployment.md` + `docs/serving/distributed_troubleshooting.md`
- 关键：Ray vs 多进程后端、`data_parallel_master_ip/port`、NCCL/IB/端口 排障
- 判据：能讲一次"两台机 TP 起不来"的排查路径

### 第 3 层：显存与 KV cache 调优 + 量化（2 天，必修，对口 OOM）
- 读：`docs/configuration/conserving_memory.md` + `docs/configuration/optimization.md`
- 抓手：`vllm/config/cache.py` 的 `gpu_memory_utilization` / `kv_cache_memory_bytes` / prefix caching / `max_num_seqs` / `max_num_batched_tokens`
- **量化（半天，部署岗高频）**：跑通一个 FP8 / AWQ / GPTQ 模型（`--quantization` + `docs/features/quantization/`）——量化直接决定"这张卡放不放得下"，和 KV 调优同等重要
- 判据：给定模型+卡+目标并发，调出一组不 OOM 又高吞吐的参数；能说清量化选型（FP8 vs AWQ 的精度/吞吐权衡）

### 第 4 层：多副本 + 负载均衡 + K8s（3–4 天，必修偏平台）
- 读：`docs/deployment/k8s.md` + `docs/deployment/docker.md` + `docs/deployment/nginx.md`
- 关键：用 `/metrics` 的 Prometheus 指标驱动容量规划和 HPA 自动扩缩
- 生态：生产多副本部署现在很多走 **vLLM production-stack / llm-d** 这类上层项目（带 KV-aware 路由），面试提到会加分
- 判据：能画出"LB → N 个 vLLM 副本 → 各自 TP=? → Prometheus → HPA"并说清每个数字怎么定

### 第 5 层：PD 分离部署（2–3 天，进阶加分）
- 读：`vllm/distributed/kv_transfer/README.md` + `docs/serving/expert_parallel_deployment.md`
- 注意：**"部署"一套 PD 分离集群是 ③b，不是 ④**；只配 KV connector 连 prefill/decode 节点，不碰 kernel

---

## 三、时间盘

| 路线 | 内容 | 时间 | 判据产出 |
|---|---|---|---|
| A-前置 | 请求生命周期 | 1–2 天 | 能口述 schedule/preempt |
| A-能力A | 加 metric | 1 天 | 第一个 diff + `/metrics` 截图 |
| A-能力B | 改调度 + 压测 | 3–5 天 | before/after 数字（核心证据） |
| **B-1** | TP/PP/DP 并行 | 2–3 天 | 会选并行策略 |
| **B-2** | 多机部署 + 排障 | 2–3 天 | 多机故障排查叙事 |
| **B-3** | KV/显存调优 + 量化 | 2 天 | 抗 OOM 调参数字 + 量化选型 |
| **B-4** | 多副本+K8s+LB | 3–4 天 | 部署架构图 |
| B-5 | PD 分离 | 2–3 天 | 跑通一套 PD 集群 |

> **建议**：优先 B 路线（1→4），按 **3 周**规划（表中天数是理想值，B-2/B-4 的多机 NCCL、K8s 的坑消耗时间高度不可控）。③b 能力必须在真机/云上多卡实例上练，光读文档到不了——需**预留云费用预算**，并把 B-2 和 B-4 安排在同一段租用窗口内以省钱。A 路线的"加 metric"当天可拿第一个 diff，作为快速建立信心的入口。
