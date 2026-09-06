# 来源关系结构解码资格门 v1

## 当前状态

该资格门的运行器、脱敏收据契约、FakeTransport 测试和独立审批文件已经完成。研究者已于 2026-08-29 接受固定范围；尚未进行该资格门的真实网络执行。

注意：本页的 `qualification_runner_v1` 仍然只接受原 v1 compatibility receipt。原 v1 请求已经被端点以 `response_format_unsupported` 拒绝，Provider-subset v2 的唯一请求又因 5 秒超时停止，v3 的唯一请求返回 `http_client_error`；三者均已消耗且没有 receipt。后续短名称 v4 的唯一请求已成功生成有效 receipt，但它不能直接满足本 v1 运行器的前置条件。必须先建立并单独审核一个绑定 v4 receipt 的资格运行器版本；在此之前不要运行下面的真实执行命令。

它是进入后续 J0/J1 观点更新试跑之前的一道工程与构念检查，不是行为实验，不估计来源效应、从众效应、因果效应或干预效果。

## 它检查什么

固定模型端点需要在 24 个最小公开合成案例中，严格复述当前公开视图的结构：

- 2 个新写的合成场景；
- 3 种公开来源关系：不显示来源、显示同一根、显示两个独立根；
- 2 种直接 EvidenceCard 状态：不存在、存在；
- 2 种呈现置换：消息顺序和根编号置换。

每个案例只要求报告可见社会消息数、可见根数、内容与公开根的对应关系、直接 EvidenceCard，以及“来源结构本身不能判定真值”。输出不包含采纳、分享、信心、真实答案或任何网络传播行为。

通过的准确含义是：在固定的、显式规则的公开结构复述任务中，24 个案例都满足 Schema 和跨字段校验。它不表示模型已经会在开放式判断任务中依据来源独立性更新观点，也不表示外部证据会改变模型判断。

## 安全边界

默认命令完全禁网：不读取环境变量或 API key，不构造 Provider，不创建输出目录。

```bash
.venv/bin/python -m evicon.conformity_source_relation_comprehension_qualification_runner_v1
```

真实执行（仅在后续的版本化运行器完成后）必须同时满足以下条件：

- 本资格门的人工审批已接受；
- 对应版本的一次性 Provider compatibility 审批已接受，并且存在当前绑定有效的 compatibility receipt；当前 v1 运行器不能使用 v2、v3 或 v4 receipt；
- 研究者另行明确授权本次 24-request 网络执行；
- 输出目录和 attempt claim 都不存在；
- 命令显式确认 24 个逻辑请求和 3072 个 completion reservation 上限。

原 v1 运行器的命令仅作为历史接口记录，当前不可因 v2 或 v3 修订而执行：

```bash
.venv/bin/python -m evicon.conformity_source_relation_comprehension_qualification_runner_v1 \
  --allow-network \
  --confirm-run \
  --confirm-request-cap 24 \
  --confirm-completion-reservation-cap 3072
```

这条命令在原 v1 compatibility receipt 缺失时会被安全地拦截；不要通过删除 claim、伪造 receipt 或重复运行原 v1 compatibility 来绕过拦截。

每个案例最多一次 transport 调用；`max_retries=0`、`max_tokens=128`、`temperature=0.0`、`seed=20261401`、超时 5 秒。任何 transport、格式、解析、绑定或覆盖失败都会停止，不会跳过案例后继续凑满 24 条。

运行器只写脱敏的案例审计和不可覆盖的 attempt receipt。它们不保存 prompt、完整回复、API key、headers、provider metadata、请求 ledger、根编号、案例文本、历史 Pilot、WVS 或 evaluator-private truth。完成后若锁文件清理失败，收据仍保留真实的完成状态，命令会以非零状态退出并报告安全的清理状态，防止静默重跑。

## 与后续行为实验的关系

若资格门未通过，只能说明固定端点、Schema 或公开来源结构任务尚未达到本门槛；尤其是超时或解析失败，不能解释为模型“不理解来源”。

若资格门通过，下一步仍需单独建立并审批状态化 J0/J1 compatibility，然后才可在另一次明确网络授权下做小规模行为试跑。后续试跑必须检验来源关系是否改变初始判断后的采纳、信心或分享，而不是把结构复述通过当作行为效果。
