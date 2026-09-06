# 来源关系理解兼容性超时修订 v3

## 修订原因

2026-08-29，已经接受并获单次网络授权的 Provider-subset v2 兼容性检查实际发出了一次请求。该请求在固定 5 秒内没有返回，安全结果为 `provider_error / timeout`。v2 没有生成兼容性 receipt，其空 attempt claim 已保留，不能删除或再次运行同一版本。

这个结果只表示该次请求超过了 5 秒等待上限。它不表示服务端拒绝 v2 Schema，不表示模型不能理解来源关系，也不是行为、因果或论文结果。

## 唯一改动

v3 是独立的超时修订，只把单次 HTTP 请求的等待上限从 5 秒改为 15 秒。以下内容全部保持不变：

- Provider-subset v2 的 JSON Schema、名称和 SHA-256；
- 原 v1 canonical parser；
- 最小公开 Birchline 合成材料及 prompt 语义；
- `max_tokens=128`、`temperature=0.0`、`seed=20261401`；
- `reasoning_effort=none`、`max_retries=0`；
- 最多一个 HTTP 请求，不重试、不回退到其他响应格式；
- 不运行 24-case 来源关系资格门，不运行观点更新或网络行为实验。

v3 配置同时绑定 v2 的配置、审批、执行模块、Provider Schema 模块、空 claim，以及“单次请求超时、没有 v2 receipt”的父状态。v2 失败详情没有被伪造成 receipt；其事实依据仍是研究者保存的脱敏命令输出和不可覆盖的 claim。

## 安全与输出

默认命令完全离线：不读取环境变量或 API key，不构造 Provider，不创建 claim、receipt 或结果目录。

```bash
.venv/bin/python -m evicon.conformity_source_relation_comprehension_compatibility_provider_subset_timeout_v3
```

预期返回 `status=network_disabled` 和 `attempt_count=0`。

只有 v3 专属审批被研究者接受，并且研究者之后对一条带 `--allow-network` 的命令另行明确授权，网络路径才可进入。接受审批本身不是持续网络权限。

真实路径最多调用一次 transport。只有 2xx、`finish_reason=stop`、canonical parser 通过且 token usage 完整时，才写不可覆盖的脱敏 receipt。receipt 只保存安全状态、模型名、完成原因、解析状态、HTTP 分类、token usage、延迟、Schema/模板标识、固定参数和 SHA-256 绑定；不保存 prompt、完整回复、公开材料文本、根编号、API key、headers、provider metadata 或 request ledger。

## 当前状态

v3 的配置、执行模块、审批契约、FakeTransport 测试和文档已离线完成。研究者已于 2026-08-29 接受该超时修订；这不构成持续或自动的联网权限。只有对下一条带 `--allow-network` 的 v3 命令另行明确授权，才可进入唯一一次网络路径。

FakeTransport 覆盖：合法 2xx、malformed JSON、未知字段、未暴露 content ID、`finish_reason=length`、timeout、connection、401、429、500、`response_format` 不支持、最多一次 transport call、脱敏 receipt、父版本绑定和 claim 防重跑。

当前不能运行 v1、v2 或 24-case 资格命令。下一步顺序只能是：研究者另行明确授权一次 v3 兼容性网络命令；兼容成功后再建立绑定 v3 receipt 的独立 24-case 资格运行器版本。

即使 v3 兼容成功，也只说明当前端点能在本请求中接受 Schema 并返回可被严格解析的结构，不说明模型已经表现出来源依赖、从众或干预效果。

## 后续记录

该文档以上内容记录的是 v3 建立时的计划。随后 v3 已在 2026-08-29 实际发出其唯一一次请求，返回 `provider_error / http_client_error`，没有 receipt，空 claim 已保留。v3 不可重跑。

为检查一个新的、单变量的接口兼容性假设，项目另建了独立的 v4：仅把 v3 的 `response_format.json_schema.name` 从 71 个字符缩短到 42 个字符，其余 Schema 正文、parser、公开材料、生成参数及 15 秒 timeout 都不变。v4 的待接受审批、离线测试和说明见 `conformity-source-relation-comprehension-provider-subset-short-name-v4.zh-CN.md`；它不修改或替代 v3 的历史记录。
