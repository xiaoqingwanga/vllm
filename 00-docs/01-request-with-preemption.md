# 请求的一生 · 抢占(preemption)与 max_num_seqs 调参实战

> 环境:家用 WSL2 服务器 RTX 3070 8GB,Docker 跑 `vllm/vllm-openai:v0.24.0`,模型 Qwen3-4B-FP8。
> 原始启动:`--max-model-len 8192 --gpu-memory-utilization 0.8 --enforce-eager`。以下全部为真实压测数字。

## 一、这张卡的硬约束(来自启动日志)

```
Available KV cache memory: 1.88 GiB
GPU KV cache size:         13,664 tokens        ← KV 总容量,只有这么多
Maximum concurrency for 8,192 tokens/request: 1.67x   ← 满上下文时只够 ~1.67 个请求并发
```

关键认知:8GB 卡装完 4B 权重后只剩 1.88 GiB 给 KV,总共 **13,664 token**。这是后面一切现象的根源。
(教训:KV 容量直接查启动日志,别按采样反推——我曾反推成 32000,错得离谱。)

## 二、实验一:压测触发抢占

方法:记 `num_preemptions_total` 基线 → 并发压 → 每 0.5s 采样 `running/waiting/kv_usage/preempt`。

| 并发 × max_tokens | KV 峰值 | waiting | 新增抢占 | 结论 |
|---|---|---|---|---|
| 16 × 1200 | 64% | 0 | **0** | 假设被证伪,这张卡扛 16 并发毫无压力 |
| 40 × 2000 | ~100% | 峰值 28 | **28** | 复现成功 |

40 并发采样过程:

```
t=03  running=40  waiting=0   kv=0.094  preempt=448   ← 贪婪 admit:一步全塞进 RUNNING
t=13  running=40  waiting=0   kv=0.985  preempt=448   ← KV 爬到满
t=14  running=37  waiting=3   kv=0.988  preempt=451   ← 抢占瞬间:踢 3 个回 WAITING
t=40  running=12  waiting=28  kv=0.971  preempt=476   ← KV 钉在 ~98%,共新增 28 次
```

## 三、抢占机制 + 源码

`vllm/v1/core/sched/scheduler.py` `Scheduler.schedule()` 抢占分支(约 L533–582):

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:
        break                                    # 分到 block,正常调度
    if self.policy == SchedulingPolicy.PRIORITY:
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
    else:
        preempted_req = self.running.pop()       # 默认 FCFS:踢 running 队尾(最晚进来的)
    self._preempt_request(preempted_req, ...)    # 丢 KV,退回 WAITING
```

流程:**贪婪 admit → KV 填满 → `allocate_slots` 返 `None` → `running.pop()` + `_preempt_request()` → 退回 WAITING → recompute 从头重算**。
- 默认 **FCFS**,踢 running 队尾 = 最晚进来的请求先受害(公平性细节)。
- 默认 recompute 抢占:被踢请求的 KV 丢弃,重排队,之前生成的 token 白算。`num_preemptions` 高 = 性能警报。

## 四、实验二:调 max_num_seqs 想消灭抢占(结论反直觉)

假设:把 `max_num_seqs` 从默认(≥40)压到 8,让调度器一开始就不过量 admit → 消灭抢占。
方法:同一负载 40×2000,改配置重启后重跑,对比 TTFT / 吞吐 / 抢占。

| 指标 | BASELINE(默认) | AFTER(max_num_seqs=8) | 变化 |
|---|---|---|---|
| 新增抢占 | 80 | **2** | **−97%** ✓ |
| 平均 TTFT | 0.19 s | **100.71 s** | **+530×** ✗ |
| 平均推理耗时 | 112.82 s | 51.09 s | −55% ✓ |
| 解码吞吐 | 407.3 tok/s | **301.2 tok/s** | **−26%** ✗ |
| wall_time | 194.3 s | 260.9 s | +34% ✗ |
| 完成请求 | 40 | 40 | = |

**抢占确实消灭了(80→2),但这是净亏损的调参。** 三点反直觉发现:

1. **TTFT 爆炸(0.19s→100s)**:只放 8 个进 RUNNING,另外 32 个在 WAITING 干等好几轮,首 token 要等 100 秒。
2. **吞吐反降 26%**:8 并发 batch 太小,**GPU 没喂饱**。3070 在 40 并发大 batch 下即使有 recompute 损耗,总吞吐仍更高。过度压 `max_num_seqs` = GPU 空转。
3. **只有单请求推理耗时改善**:一旦 admit 就不再被抢占能跑完,但代价是另外 32 个请求排队。

## 五、结论:调参心法

- **`max_num_seqs` 是把钝刀**。压太低只是把"抢占问题"换成"GPU 空转 + 排队问题",总体更差。
- **抢占在这张卡上没那么伤吞吐**——贪婪大 batch 本身高效,抢占伤的是**尾延迟方差**。
- **真正的解是给 KV 扩容**,而非压并发:抬 `gpu_memory_utilization`(0.8→0.9)、降 `max_model_len`(8192→4096,多数请求用不满),让更多序列能**无抢占地并发** → 高吞吐 + 无抢占 + 低 TTFT 三赢。
- 面试叙事:不要说"我把 max_num_seqs 调小解决了抢占",要说"我用 before/after 数字证明了压 max_num_seqs 是负优化,真正瓶颈是 KV 容量,应从 gpu_mem_util / max_model_len 下手"。

## 六、运维踩坑(改配置→重启→验证 闭环)

1. **ENTRYPOINT 陷阱(照抄 `.Args` 会 double serve)**:`vllm/vllm-openai` 的 ENTRYPOINT 是 `[vllm serve]`。docker 的字段规则是 `.Path`=entrypoint[0](`vllm`)、`.Args`=entrypoint[1:] + 你的命令,所以 `docker inspect .Args` 显示 `[serve --model ...]` 里那个 `serve` **来自 entrypoint,不是命令部分**。复原时若把 `.Args` 原样当命令传 → entrypoint 再拼一个 → `vllm serve serve --model` → `unrecognized arguments: serve` 崩溃重启。**正确做法:命令只给 flags(`--model ...`),不带 serve。** 复原前务必先看镜像 ENTRYPOINT。
2. **镜像用 sha256 ID 固定**,别用 `:latest` 重建——`:latest` 可能被悄悄换版本。
3. **压测超时**:默认配置下 40×2000 因反复抢占+recompute,drain 要 >3min;改配置前先等 `running` 归零再测,否则数据被上一批污染。
4. **推理模型吃 token**:Qwen3 默认先输出 `<think>`,`max_tokens=80` 连正式答案都出不来(`finish_reason=length`)。生产需关 thinking 或把预算留到几百+。

## 七、附:可复用脚本(在服务器 /tmp)

- `relaunch.sh [max_num_seqs]`:停→删→按新参数重建容器,等 `/health`;**不带参数=还原原始配置**。
- `bench.sh <标签> <并发> <max_tokens>`:压测并从 `/metrics` 直方图算出抢占/TTFT/推理耗时/吞吐。

## 八、下一步(可选)

验证"给 KV 扩容"这个真正的解:`--max-model-len 4096 --gpu-memory-utilization 0.9`(不压 max_num_seqs),
跑同样 40×2000,预期抢占少、吞吐高、TTFT 低,用数字坐实第五节的结论。
