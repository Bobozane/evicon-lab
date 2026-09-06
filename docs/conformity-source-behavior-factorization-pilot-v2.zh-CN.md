# 来源行为因素分解小规模试跑门（v2）

v2 是对 v1 输出门失败后的独立协议修订。v1 的代码、配置、失败审计和输出目录继续保留，不能与 v2 合并。v2 只修改公开提示模板和版本绑定；场景、条件矩阵、响应字段、JSON Schema 和请求额度保持不变。

## 为什么需要 v2

v1 的第 2 个请求返回了合法的 2xx/`stop` 响应，但没有通过严格解析。由于运行器不保存模型原文，只能确认这是“模型输出不满足契约”，不能安全地猜测具体字段错误。固定顺序中第 2 个分支是自我反思，因此 v2 在提示中明确规定：即使是没有新增材料的自我反思，也必须把实际参考的初始内容写入 `content_ids_used`。

v2 同时把以下规则直接写入系统提示：

- 判断等级和公开行动的对应关系：拒绝、保留、采纳三组固定映射；
- `content_ids_used` 必须非空，且只能引用当前可见内容；
- 不分享时 `share_content_id` 必须为空，分享时必须引用已使用且可见的内容；
- 同根、独立根和未显示来源只表示公开来源关系，不表示真假或证据质量。

解析器仍然严格拒绝未知字段、不可见内容、重复内容 ID 和跨字段矛盾。提示修订不能替代解析和审计。

## 固定范围

- 2 个最小公开合成场景；
- 2 个固定随机种子；
- 8 个条件：初始判断、自我反思、无来源文本重复、同源隐藏、同源显示、独立来源隐藏、独立来源显示、外部证据；
- 共 32 个逻辑请求，每个请求最多一次传输调用；
- `max_tokens=256`、`temperature=0.0`、`max_retries=0`、超时 5 秒、`reasoning_effort=none`；
- 响应 Schema 仍为 `conformity_source_behavior_factorized_response.v1`，提示模板升级为 `conformity_source_behavior_factorized_turn.v2`。

所有材料都来自本协议的公开合成场景。运行器不会读取旧 Pilot、WVS、历史结果或 evaluator-private truth。

## 默认离线预检

默认命令只校验配置、协议哈希、矩阵和额度，不读取 API key，不构造 Provider，不建立输出目录：

```bash
.venv/bin/python -m evicon.conformity_source_behavior_factorization_runner_v2
```

离线测试还覆盖了合法响应、JSON/Schema 错误、内容引用错误、判断/行动矛盾、分享字段错误、截断、超时、连接失败、401、429、500、格式不支持，以及失败后只调用一次传输层。

## 一次性试跑记录

修订完成后的 v2 试跑先后有两次传输中断：第一次完成 4/32 后超时，第二次完成 8/32 后超时。两批次均未生成收据，部分审计保留在各自独立目录中，不能用于行为、因果或论文分析。

随后在新的输出目录 `outputs/conformity-source-behavior-factorization-pilot-v2-rerun-02/` 完成了一次完整试跑：

- 32/32 请求完成；
- `parser_invalid_count=0`；
- `transport_attempt_count=32`；
- `finish_reason=stop`；
- 已生成并通过绑定校验的 `pilot_receipt.json`；
- 总延迟约 70.4 秒，总 token usage 为 25,498。

执行命令为：

```bash
.venv/bin/python -m evicon.conformity_source_behavior_factorization_runner_v2 \
  --allow-network \
  --confirm-run \
  --confirm-request-cap 32 \
  --confirm-completion-reservation-cap 8192
```

这条命令是一次性开发试跑，不是 12-request 行为资格实验，也不是多轮网络主实验。任何一个请求解析失败、截断或传输失败都会停止后续请求；失败目录不会被覆盖。当前 v2 输出门已经通过，不应在同一协议上继续重复联网试跑。

## 输出边界

成功时只在新的输出目录写入脱敏案例审计和绑定收据。收据绑定配置、协议、运行器、Schema/模板版本及哈希，并记录请求计数、token usage、延迟和安全状态。

不会保存 prompt、完整回复、content ID 明细、API key、headers、provider metadata、request ledger 或私有真值。v1 的失败目录保持原样，不能并入 v2 分析。

通过该门只说明“修订后的协议和模型输出可以被安全解析和审计”。完整批次的安全审计可用于开发阶段的描述性检查，但不证明来源效应、从众效应、干预效果或任何论文结论。

## 开发性描述性检查

完整批次包含 4 个配对组，每个条件 4 个案例。安全审计汇总得到：

- 初始判断和自我反思均为 `uncertain / withhold`；
- 无来源文本重复、同源消息和独立来源消息均为 `lean_adopt / adopt`；
- 外部证据也为 `lean_adopt / adopt`，且平均信心更高；
- 文本重复相对于自我反思的预注册序数差异为 `+1.0`（4/4 配对组为正）；
- 外部证据相对于自我反思的预注册序数差异为 `+1.0`（4/4 配对组为正）；
- 同源显示相对于同源隐藏、独立来源显示相对于独立来源隐藏，以及独立来源相对于同源的差异均为 `0`。

这说明当前模型在本开发协议中对“新增文本”和“直接证据”有反应，但尚未显示出可观察的来源独立性差异。由于只有 4 个配对组、单一模型和开发性样本，这只能作为构念效度和刺激设计的信号，不能写成来源效应的负结果。
