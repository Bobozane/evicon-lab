# 来源关系理解兼容性修订（Provider 子集 v2）

## 为什么需要修订

2026-08-29 对原 `conformity_source_relation_comprehension_compatibility.v1` 的一次性检查已经实际发出一次请求。服务端在模型生成前返回了 `4xx`，安全分类为 `response_format_unsupported`。这表示当前端点拒绝了原始 JSON Schema 的表达方式，不表示模型答错，也不表示模型不理解来源关系。

原 v1 的一次性 claim 和失败状态永久保留在：

`outputs/study-locks/conformity_source_relation_comprehension_compatibility.v1.claim`

本修订不覆盖、不重试、不修改该记录，也不把失败请求当作行为数据。

## 修订内容

新增的 v2 只改变发送给 Provider 的 Schema 投影，协议的公开材料和本地严格解析器不变：

- Provider Schema 改为扁平对象和普通 `items` 数组；
- 移除端点常见受限子集不接受的 `prefixItems`、`const`、`$defs`、`$ref`；
- 保留枚举、对象必填字段、数组长度和 `additionalProperties=false` 等约束；
- Provider 返回后仍使用原 v1 的 Pydantic 解析器检查精确顺序、根编号、数量和跨字段关系；
- 不降级为 `json_object`，不使用宽松解析，也不改变 prompt 的任务含义。

因此，“wire Schema 较保守”不等于“本地校验较宽松”。Provider 只负责生成候选 JSON，最终是否合格仍由 canonical parser 决定。

## 独立边界

v2 是一个独立的、一次性的兼容性 gate：

- 默认完全禁网，不读取环境变量或 API key，不构造 Provider，不写 receipt；
- 只有专属审批文件变为 `accepted`，并且命令显式加入 `--allow-network`，才允许进入网络路径；
- 最多一个 HTTP 请求，`max_retries=0`；
- 固定 `max_tokens=128`、`temperature=0.0`、`seed=20261401`、超时 5 秒、`reasoning_effort=none`；
- 只使用新写的最小公开合成材料，不读取历史 Pilot、WVS、evaluator-private truth 或旧结果；
- 失败会保留本次 attempt claim，防止静默重跑；
- 只有 2xx、`finish_reason=stop`、本地严格解析成功且 token usage 完整时，才写不可覆盖的脱敏 receipt。

receipt 只保存状态、模型、完成原因、解析状态、HTTP 分类、token usage、延迟、Schema/模板版本、固定参数和 SHA-256 绑定。它不保存 prompt、完整回复、消息文本、根编号、API key、headers、provider metadata 或 request ledger。

## 当前状态

新版配置和测试已经在离线环境完成。专属审批文件已接受。2026-08-29 在研究者明确授权后实际发出了一次请求，但端点在固定 5 秒超时内没有返回，结果为 `timeout`；因此没有生成 v2 receipt，且 v2 attempt claim 已保留，不能重复运行同一版本。离线默认检查命令为：

```bash
.venv/bin/python -m evicon.conformity_source_relation_comprehension_compatibility_provider_subset_v2
```

预期状态是 `network_disabled`。FakeTransport 测试覆盖合法 2xx、malformed JSON、未知字段、未暴露 content ID、`finish_reason=length`、timeout、connection、401、429、500、`response_format` 不支持、一次调用上限和敏感信息泄漏。

本次超时只说明该次请求没有在 5 秒内完成，不能说明 Provider 不支持该 Schema、模型无法理解来源关系，或任何行为/因果效果。不要删除 claim、伪造 receipt 或再次运行 v2；如需重新检查，必须建立新的版本化兼容性修订并重新审核。

## 后续状态

v2 的审批和单次授权均已消耗，不能再次运行。针对其 5 秒超时，已建立只把等待上限改为 15 秒的独立 v3 修订；v3 说明见 `docs/conformity-source-relation-comprehension-provider-subset-timeout-v3.zh-CN.md`。

v3 必须先经过新的研究者审批，之后还需另行明确授权其唯一一次网络命令。即使 v3 兼容性成功，也只说明该端点接受这份保守 Schema 并返回可被本地解析器接受的结构；它不说明模型已经表现出来源效应。随后仍必须建立并审核绑定 v3 receipt 的独立 24-case 资格运行器版本；现有 v1 运行器不能直接使用 v3 receipt。
