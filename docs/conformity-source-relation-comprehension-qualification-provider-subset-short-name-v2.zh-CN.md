# 来源关系理解资格测试 v2

## 当前状态

本资格测试已经完成离线实现、FakeTransport 测试和独立审批边界。人工审批为 `accepted`，并已于 2026-08-30 完成一次真实的 24-request 网络测试。审批文件当前 SHA-256 为 `637cab365c655f8a565f72266a23e8992816fee16b0b17734fed78dd3d3122074`。

本次执行结果为 `source_structure_qualified`：24/24 案例完成，24 次请求全部为 `2xx`、`stop` 且通过本地解析；没有 parser invalid。脱敏 receipt 位于 `outputs/conformity-source-relation-comprehension-qualification-provider-subset-short-name-v2/qualification_attempt_receipt.json`，安全案例审计位于同目录的 `safe_case_audits.jsonl`。本次 receipt SHA-256 为 `fc846174e0dbacb5292caed371aa29b794eb6e63158287b3673a7a6561ccce6c`，审计文件 SHA-256 为 `27b595fd68173840a311b58f674dae56625e0c96c202ba53837f858f0afe22b7`。

它只承接已经成功的 v4 Provider compatibility receipt。该回执仅说明固定端点曾在一次最小公开合成输入上接受了所需的结构化输出请求；它不证明模型理解来源关系，更不构成从众、干预、因果或论文结论。

## 要检查的问题

测试要求固定模型在 24 个最小公开合成案例中，准确复述当前公开视图中的来源结构：

- 2 个合成场景；
- 3 种来源关系：不显示来源、显示同一来源根、显示两个独立来源根；
- 2 种直接 EvidenceCard 状态：不存在、存在；
- 2 种呈现顺序。

每个案例只报告公开可见的社会消息数量、可见来源根数量、消息与公开来源根的对应关系、直接证据卡以及“仅凭来源结构不能判断真值”。它不要求采纳、分享、信心、真实答案或网络传播行为。

通过的含义严格限于：在这个固定的公开结构复述任务中，24 个案例都通过 provider schema 和本地跨字段校验。通过不表示模型会在开放式判断中根据来源独立性改变观点。

## 双层校验

线上请求使用 v4 已验证的 provider subset schema 和短 schema 名称 `source_relation_comprehension_v1_subset_v4`，并保持 `strict=true`。本地仍使用 canonical parser 对完整来源关系进行校验。

运行前会同时锁定：

- v4 compatibility config、approval、module 和成功 receipt；
- 校准协议、模板版本和 canonical schema；
- provider schema 及其模块；
- 固定模型 `gpt-5.6-luna`；
- `max_tokens=128`、`temperature=0.0`、`seed=20261401`、`max_retries=0`、`reasoning_effort=none`、`timeout=15s`；
- 24 个逻辑请求和 3072 个 completion reservation 上限。

每个请求前和写入最终 receipt 前都会重新检查这些绑定。任何绑定变化、transport 异常、格式异常、解析异常或覆盖不完整都会终止本次尝试，不会重试或跳过案例。

## 数据与执行边界

默认 CLI 不读取环境变量或 API key，不构造 Provider，也不创建 claim、输出目录或 receipt。

实际运行只有在以下条件同时满足时才可能进行：本资格测试的人工审批被单独接受；v4 父审批和当前成功 receipt 仍有效；研究者再次明确授权这一次 24-request 网络运行；输出目录和 claim 均为空；命令显式确认两个上限。

运行器只会写脱敏的案例审计和不可覆盖的 attempt receipt。它不会保存 prompt、完整模型回复、API key、headers、provider metadata、request ledger、公开消息文本、来源根编号、历史 Pilot、WVS 或 evaluator-private truth。任何已经发出 transport 请求的失败都会保留 claim，防止静默重复执行。

## 结果解释与下一步

若未通过，只能说明固定端点、请求格式或这个公开结构任务尚未满足资格门槛；不能把 timeout、连接失败或格式失败解释为模型“不理解来源”。

若通过，下一步仍需要另行设计并审批“初始判断到社会暴露后最终判断”的小规模行为试运行，才可检查来源关系是否与公开判断或分享行为有关。本次资格测试本身只证明固定模型能够完成预先定义的来源结构复述任务，不估计任何行为效果、从众效果、干预效果或因果关系。
