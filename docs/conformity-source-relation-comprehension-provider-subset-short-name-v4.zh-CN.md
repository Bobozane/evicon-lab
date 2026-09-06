# 来源关系理解兼容性短 Schema 名称修订 v4

## 为什么建立 v4

v1、v2、v3 是三个已消耗且不可重跑的父版本：

- v1 已发出一次请求，结果为 `response_format_unsupported`；
- v2 已发出一次请求，结果为 `timeout`；
- v3 已发出一次请求，结果为 `http_client_error`。

基础 `/models` 请求曾返回 200，且同一端点曾接受本项目其他严格 JSON Schema 请求。因此，这三次结果不能简单归因为 API key 失效或端点完全不可达。

v3 的 `response_format.json_schema.name` 为
`conformity_source_relation_comprehension_response_v1_provider_subset_v2`，长度为 71。OpenAI Chat Completions 的公开接口说明要求该名称仅使用字母、数字、下划线或连字符，且最长 64 个字符。v4 将其缩短为 `source_relation_comprehension_v1_subset_v4`，长度为 42。

这是一项高概率但尚未被服务端错误正文直接证实的兼容性假设。项目有意不保存原始错误正文，因此不能把“名称超长”写成已经证明的 v3 失败原因。

## 唯一改动

相对于已经实际运行的 v3，v4 的唯一请求载荷差异是：

```text
response_format.json_schema.name
```

其余内容保持不变：

- Provider-facing JSON Schema 的正文及 SHA-256；
- canonical parser、公开最小合成材料与 prompt 语义；
- `max_tokens=128`、`temperature=0.0`、`seed=20261401`；
- `reasoning_effort=none`、`max_retries=0`、`timeout=15` 秒；
- `response_format=json_schema` 和 `strict=true`；
- 最多一个 HTTP 请求，不重试、不降级到其他格式；
- 不运行来源关系理解资格门、观点更新协议或网络行为实验。

配置将这一个字段差异、父版本的配置/审批/模块哈希、三个空 attempt claim、Provider Schema、canonical schema、协议/模板版本全部绑定。v1-v3 的 claim 和无 receipt 状态会在运行前再次核验。

## 安全边界

默认命令完全离线：不读取环境变量或 API key，不构造 Provider，不创建 claim、receipt 或结果目录。

```bash
.venv/bin/python -m evicon.conformity_source_relation_comprehension_compatibility_provider_subset_short_name_v4
```

只有以下两个条件同时满足时，才可能进行唯一一次真实请求：

1. 研究者明确接受 v4 的专属审批；
2. 研究者之后单独授权一条带 `--allow-network` 的命令。

接受审批不是持续联网权限。真实路径若成功，只写不可覆盖的脱敏 receipt，包含模型名、完成原因、解析状态、HTTP 分类、token usage、延迟、schema、固定参数和哈希绑定；不保存 prompt、完整回复、公开材料文本、来源根、API key、headers、provider metadata 或 request ledger。

## 离线验证

FakeTransport 覆盖：

- 合法 2xx，以及请求载荷与 v3 的逐字段比较；
- malformed JSON、未知字段、未暴露 content ID、`finish_reason=length`；
- timeout、connection、401、429、500、一般 4xx、`response_format` 不支持；
- 最多一次 transport call、claim 防重跑、父版本不变；
- 结果和 receipt 中没有 prompt、完整回复、密钥、headers、provider metadata 或公开材料泄漏。

这些测试只验证运行器和审计边界，不是模型行为、因果效果或论文结果。

## 当前状态

v4 的运行器、配置、离线契约和 FakeTransport 测试已完成；研究者已于 2026-08-30 接受该单变量修订，并在同日完成唯一一次真实兼容性请求。该请求返回 2xx、`finish_reason=stop`、canonical parser 通过，生成了不可覆盖的脱敏 receipt：

```text
outputs/study-locks/conformity_source_relation_comprehension_compatibility_provider_subset_short_name.v4.json
SHA-256: 29b38cca78c8c4c1ff53a89376ae2fd6368329aa80e51528ae80ac1e1dbd8ba2
```

receipt 记录模型为 `gpt-5.6-luna`、总 token usage 为 698、延迟约 2494 ms。它仅证明当前端点在这一份最小公开材料、固定参数和短名称下能接受严格 Schema 并产生可解析结构；它不证明模型已经理解来源关系，也不构成从众、行为、因果或论文结果。

下一步应建立一个独立版本的 24-case 来源关系理解资格运行器，使其明确绑定此 v4 receipt；原 v1 运行器和已消耗的 v1-v3 记录不应修改或重跑。

参考： [OpenAI Chat Completions API reference](https://platform.openai.com/docs/api-reference/chat/create)
