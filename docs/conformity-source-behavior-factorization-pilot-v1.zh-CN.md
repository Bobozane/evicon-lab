# 来源行为因素分解小规模试跑门（v1）

这个模块是离线因素分解协议之后的下一道门。它只负责一次受控的小规模真实模型试跑，不是旧的 12-case 行为资格实验，也不是网络级联主实验。

## 固定范围

- 2 个合成场景：错误观点形成表面多数、少数 Agent 持有有证据支持的纠正；
- 2 个固定随机种子；
- 8 个条件：初始判断、自我反思、无来源文本重复、同源隐藏、同源显示、独立来源隐藏、独立来源显示、外部证据；
- 共 32 个逻辑请求，每个请求最多一次传输调用；
- `max_tokens=256`、`temperature=0.0`、`max_retries=0`、超时 5 秒、`reasoning_effort=none`；
- JSON Schema 与离线协议完全相同。

所有材料都是本模块使用的最小公开合成内容。运行器不会读取旧 Pilot、WVS、历史结果或 evaluator-private truth。

## 默认行为

默认命令只做离线预检：校验配置、协议哈希、因素矩阵和额度，不读取 `EVICON_LLM_API_KEY`，不构造 Provider，也不建立输出目录。

```bash
.venv/bin/python -m evicon.conformity_source_behavior_factorization_runner_v1
```

只带 `--allow-network` 仍然不会发请求。必须同时给出精确的额度确认；输出目录必须不存在，运行器不会覆盖旧文件。

## 未来的一次性命令

只有研究者单独授权本次试跑后，才运行：

```bash
.venv/bin/python -m evicon.conformity_source_behavior_factorization_runner_v1 \
  --allow-network \
  --confirm-run \
  --confirm-request-cap 32 \
  --confirm-completion-reservation-cap 8192
```

本命令已经在一次单独授权后尝试执行过一次。运行在第 2 个请求停止：第 1 个案例通过解析，第 2 个案例（按固定顺序为 `self_reflection`）返回 2xx/`stop`，但未通过严格解析契约。实际发送了 2 次请求，完成 1 个案例，未生成完成收据，也未继续发送剩余请求。该尝试不能用于行为、因果或论文分析。

## 输出与安全边界

成功时只写入 `outputs/conformity-source-behavior-factorization-pilot-v1/` 下的：

- `safe_case_audits.jsonl`：案例 ID、条件、解析后的判断/行动类别、使用内容数量、可见来源根数量、finish reason、HTTP 分类、token usage、延迟、共享初始判断哈希；
- `pilot_receipt.json`：配置、协议、运行器、Schema/模板版本和哈希绑定，以及请求计数和安全标记。

不会写入 prompt、完整回复、content ID 明细、API key、headers、provider metadata、request ledger 或私有真值。失败时只返回脱敏错误类别；不会因为错误而重试或继续下一个请求。失败前已写入的安全案例审计会保留，运行器不会覆盖该目录。后续版本的失败摘要会额外给出固定的解析类别（例如 `invalid_schema` 或 `unavailable_content_id`），但仍不会暴露模型原文。

该试跑只回答“协议和模型输出是否可分析”，不估计行为效果，不作因果结论，也不构成论文结果。本次未通过输出门。后续修订已独立放入 [v2 试跑门](conformity-source-behavior-factorization-pilot-v2.zh-CN.md)，使用新的协议、配置和输出目录；v1 失败目录不覆盖、不合并。不能据此直接进入多轮网络实验。
