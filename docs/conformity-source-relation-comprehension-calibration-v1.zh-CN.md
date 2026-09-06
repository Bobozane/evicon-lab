# 来源关系理解与构念校准门（v1）

## 目的

这个协议是后续来源依赖行为实验之前的一道独立资格门。它只回答一个工程和构念问题：模型能否根据**已经公开展示**的来源关系，准确区分：

- 来源没有展示；
- 两条社会消息来自同一个公开来源根；
- 两条社会消息来自两个不同的公开来源根；
- 两条社会消息与一张直接证据卡是不同类型的材料。

它不要求模型采纳或拒绝任何主张，不测量信心、分享、从众、干预或网络级联，也不包含 evaluator-private truth。因此通过这个门不等于来源会改变模型行为，更不等于存在行为或因果效果。

该协议是新的独立版本。它不读取、不修改、不合并以下任何材料：已完成的 v2 因素分解批次、旧 H-G/H-D/WVS 批次、来源操纵的历史恢复链、ledger、receipt、分析结果或私有真值。

## 为什么现在需要它

已完成的 v2 开发性试跑表明，模型会对新增支持文本和直接证据产生反应，但在同源与独立来源的公开标记之间没有出现可观察的行为分离。这有两种不同解释：

1. 模型读懂了来源关系，但当前任务中没有用它改变行为；
2. 来源呈现本身没有被模型稳定理解。

这两个解释不能靠继续重复同一行为试跑来区分。v1 先单独检查第二点，且不把“独立来源更可靠”或“应该采纳”写进提示。

## 固定矩阵

矩阵由新的、最小公开合成材料构成：

| 因素 | 水平 | 数量 |
|---|---|---:|
| 合成场景 | HarborLink、CedarWorks | 2 |
| 社会来源关系 | 未展示、同一公开根、两个不同公开根 | 3 |
| 直接证据卡 | 不展示、展示一张 | 2 |
| 呈现置换 | 正序、消息顺序和根编号同时交换 | 2 |
| 总案例 | `2 x 3 x 2 x 2` | 24 |

在同一个“场景 x 证据状态 x 呈现置换”组内，三种来源关系看到的社会消息文本完全相同；唯一变化是公开的 `source_root_id`。展示证据卡时，三个来源关系使用同一张卡。这样可把以下概念分开：

- `null` 表示“公开视图没有展示来源”，并不表示现实中没有来源；
- 两个不同社会来源仍然是社会报告，不会自动变成直接证据；
- 直接证据卡是可见材料类型，不是“已经证明是真的”；
- 来源结构和材料类型都不能单独决定真假或来源质量。

正序与反序不仅交换两条社会消息的位置，也交换不透明来源编号。模型若只是记住“第一个编号”或“message-a 对应某个根”，会被严格解析器识别出来。

## 响应契约

响应只包含结构字段：

- 可见社会消息数量；
- 可见来源关系：`not_shown`、`same_shown_root` 或 `distinct_shown_roots`；
- 可见社会来源根数量：`0`、`1` 或 `2`；
- 按可见顺序复述的两条社会消息来源对应关系；
- 可见直接证据卡 ID 列表；
- 支持材料类型：只有社会报告，或包含直接证据；
- `source_structure_alone_decides_truth=false`。

解析器会拒绝未知字段、重复 JSON 字段、错误根数量、错误关系类别、错误顺序或根编号、把独立社会来源误写成证据卡，以及把来源结构说成能决定真值的输出。

安全审计接口只保留案例/场景/条件 ID、解析状态、计数和结构类别；不包含消息文本、prompt、完整模型回复、根编号、API key、headers、provider metadata 或真值。当前离线版本不写任何审计或结果文件。

## 默认离线检查

默认命令只校验配置哈希、Schema 哈希、矩阵和公开/内部字段隔离。它不读取环境变量或 API key，不构造真实 Provider，不联网，也不创建输出目录：

```bash
.venv/bin/python -m evicon.conformity_source_relation_comprehension_calibration_v1
```

更完整的离线烟雾检查使用两个确定性的 FakeProvider：一个正确复述公开结构，另一个故意按固定内容 ID 排序而忽略展示顺序。后者必须在所有反序案例被拒绝：

```bash
.venv/bin/python -m evicon.conformity_source_relation_comprehension_calibration_v1 --offline-smoke
```

该命令仍然没有网络调用，也不构造真实 Provider。FakeProvider 通过只证明契约、解析器和测量线路可以识别预设的正确/错误模式，不是模型行为结果。

## 一次性接口兼容性检查

另有一个独立的一次性兼容性检查模块，用于确认 Provider 能接受本资格门的严格 JSON Schema，并返回可解析的结构结果。它不运行 24 个资格案例，不测量来源效应，也不生成行为结果。

默认命令同样完全离线：不读环境变量或 API key，不构造 Provider，不写 receipt：

```bash
.venv/bin/python -m evicon.conformity_source_relation_comprehension_compatibility_v1
```

网络路径必须同时满足三道边界：审核文件已经从 `pending` 变为 `accepted`、研究者在命令中显式加入 `--allow-network`、默认 receipt 路径尚不存在。它最多发送一次 HTTP 请求，固定使用：

- `response_format=json_schema`，且 Schema 与本协议响应契约完全一致；
- `max_tokens=128`、`temperature=0.0`、`seed=20261401`；
- `max_retries=0`、超时 5 秒、`reasoning_effort=none`；
- 新写的最小公开合成材料，而不是 24 个校准案例、v2 输出、历史 Pilot、WVS 或私有真值。

只有 2xx、`finish_reason=stop`、严格解析成功且 token usage 完整时，才会写入一个不可覆盖的脱敏 receipt。receipt 绑定配置、协议、审核、Schema、模板、兼容性模块和生成参数的 SHA-256，不包含 prompt、完整回复、根编号、案例文本、API key、headers、provider metadata 或 request ledger。

原 v1 审核文件已于 2026-08-29 接受，但实际一次性请求被端点以 `response_format_unsupported` 拒绝。该失败 claim 被永久保留，不能通过重复运行 v1 覆盖。针对受限 Schema 子集的独立 v2 修订、配置和审批说明见 `docs/conformity-source-relation-comprehension-provider-subset-v2.zh-CN.md`；v2 的一次授权请求因 5 秒超时未完成，claim 已保留且没有 receipt。随后 v3 仅将等待上限改为 15 秒，但其唯一请求返回 `http_client_error`，同样没有 receipt。

独立 v4 只将 v3 的 `response_format.json_schema.name` 从 71 个字符缩短到 42 个字符，保留 Schema 正文、canonical parser、公开材料和生成参数。v4 的唯一请求已在 2026-08-30 成功返回 2xx、`finish_reason=stop` 和 parser-valid，并生成脱敏 receipt。说明见 `docs/conformity-source-relation-comprehension-provider-subset-short-name-v4.zh-CN.md`。这一接口兼容成功不能当作模型行为结果，也不能使 v1-v3 重新可运行。

## 通过标准与下一步

每个版本的 compatibility 模块只包含一条严格受控的网络路径：它必须使用该版本已接受的专属审核文件、研究者对该次命令的显式授权和 `--allow-network`；同时 receipt 和执行 claim 都必须尚不存在。它只会发送一次请求，并且只在接口、Schema、解析和 token usage 全部合格时写入脱敏 receipt。v1、v2、v3 的唯一请求都已消耗且失败记录保留；v4 的唯一请求已完成并产生有效 receipt。

真正的 24-case 来源关系理解校准现已具备原 v1 运行器、独立审核和单独请求上限确认；该审核已接受，但仍需先取得与运行器版本匹配的有效 compatibility receipt。当前 v1 运行器只接受原 v1 receipt，不能把 v4 receipt 作为替代。因此下一步必须新建并审核一个独立的、明确绑定 v4 receipt 的运行器版本，而不能修改或重跑 v1。兼容成功不能替代理解校准，也不是行为实验或效果证据。运行器说明见 `docs/conformity-source-relation-comprehension-qualification-v1.zh-CN.md`。

真实资格门的预先标准应是：完整批次的所有案例均通过严格解析，并且正序/反序、无证据/有证据、同源/独立根之间都没有结构性混淆。若不通过，应只修改来源呈现或理解任务，不进入行为实验。

只有来源关系理解通过后，才进入下一版的观点更新协议。下一版的详细设计见 `docs/conformity-source-behavior-update-design-v1.zh-CN.md`，其关键要求是：

1. 先记录同一实验单元的初始公开判断；
2. 再将该初始判断与新的、文本匹配的社会输入一起提供给该单元产生最终判断；
3. 将消息数量、字数、支持强度、顺序和来源独立性分别控制；
4. 加入自我反思、无来源重复、同源重复、独立来源、直接证据、矛盾信息和未决信息等对照；
5. 在结果查看前固定主要比较：同源与独立来源的判断变化、信心变化和分享变化，并保留隐藏来源的 placebo 对照；
6. 只有个体层面更新可测后，再进入无干预网络机制试跑，最后才比较来源感知干预。

这条路线提高的是实验的可解释性和发现真实差异的机会，不保证某个模型一定表现出来源效应。若模型通过来源理解门后仍没有行为差异，这将是“所测模型在该严格协议下来源不敏感”的有边界结果，而不是可以通过反复调参消除的失败。
