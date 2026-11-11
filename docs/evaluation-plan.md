# InterviewPilot 实验与验证流程

项目已经具备功能闭环。本次补充的目标是将实现变成可检查、可复现的项目证据。
新增数据最初由开发辅助生成；项目作者已于2026-10-12逐条复核并确认保留全部原标注。
120条检索与24条评分均已记录为human_reviewed，复核人与日期保存到每条JSON，原文件备份保留。
测试通过也不意味着真实模型回答正确，两者在报告中分开记录。

## 数据与人工复核

- `data/evaluation/knowledge-v1.json`：保留原20题，继续用于已有网页回归流程。
- `data/evaluation/knowledge-v2.json`：120题，80开发/40测试，80个来源关联组。
  86条独立改写、20条双主题查询、14条无资料查询；不施加领域过滤，以保留相似题干扰。
- `data/evaluation/scoring-v1.json`：24个回答案例、72个评分点，四个主题各覆盖六类回答。
  标签为mastered/partial/incorrect/not_addressed，已由项目作者人工复核并认可。

离线校验和生成工作表：

```powershell
.\.venv\Scripts\python.exe scripts/check_evidence.py
```

检查来源是否存在、ID及标签结构、开发/测试的主题与来源交叉。结果和复核工作表位于
`reports/evidence/`。原始题库每个完整参考答案会列在工作表中，便于逐项检查。

还可以运行纯BM25开发基线，无LLM/Embedding调用：

```powershell
.\.venv\Scripts\python.exe scripts/offline_keyword_baseline.py
```

这个基线在当前真实索引上执行原查询，整题取3个父块，便于发现未命中的问题；
不调用问题拆解、生成与拒答，不与线上“每个子问题各取3条”的指标直接宣称提升。
完成本地验证后运行`python scripts/summarize_evidence.py`可生成证据索引。

人工复核时直接修改JSON：

1. 判断问题是否能由题库回答；不要将模型常识当作题库证据。
2. 核对目标来源及所有独立子问题的证据。当前按expected_sources里的每个来源统计覆盖率；
   多个可互换来源若要求“任意命中一个”，应先支持等价来源组，不能直接当作全部必需来源填入。
   新增来源如果跨越split，必须先重新分组。
3. 检查参考答案/要点的事实与完整性；评分集尤其检查等价解法与后续纠错。
4. 每题填写`review.approved=true`、`review.reviewer`和`review.reviewed_at`，备注记录争议。
5. 全部复核后将数据集`label_status`改为`human_reviewed`，保存复核后的文件版本。

`--require-reviewed`会检查状态及每题复核记录。不要仅改顶层状态来跳过复核。
这项检查只验证复核记录，不能认证复核人身份或保证标注正确。

划分规则：同来源改写与双主题题的来源连通组不能跨开发/测试集。分组隔离的是查询评测，
知识索引仍包含全部资料；这是检索任务所需，并非让测试知识在题库中不可见。
同义但引用了不同来源的题仍可能有关联，需要人工检查分组，而不能仅依靠程序。
先在开发集选参数，再冻结参数与代码，用测试集最终评估，不反复按测试结果调参。

## 七种独立消融

| 名称 | BM25+向量/RRF | MQE | HyDE | 精排 |
| --- | --- | --- | --- | --- |
| vector | 仅向量 | 关 | 关 | 关 |
| hybrid | 开 | 关 | 关 | 关 |
| mqe | 开 | 开 | 关 | 关 |
| hyde | 开 | 关 | 开 | 关 |
| hybrid_reranked | 开 | 关 | 关 | 开 |
| expanded | 开 | 开 | 开 | 关 |
| reranked | 开 | 开 | 开 | 开 |

每题只拆解一次，各方案复用相同子问题；轮换方案执行顺序。独立消融首先回答
MQE、HyDE和精排各自是否有效，双扩展与最终方案用于观察组合效果。
网页新增三种可选消融，仍默认选原四种方案，避免默认增加调用量。

先运行小规模开发检查（真实API调用，需要服务可用）：

```powershell
.\.venv\Scripts\python.exe -m techrag eval --ablation --split dev --limit 6 --repeat 1 --no-answer-judge --output-dir reports/experiments
```

limit不是简单取前N题，而是用固定seed按类别轮换抽样；相同seed的选择可复现。
准备好标注后运行正式开发实验，再冻结参数并做测试集实验：

```powershell
.\.venv\Scripts\python.exe -m techrag eval --ablation --split dev --repeat 3 --require-reviewed --output-dir reports/experiments
.\.venv\Scripts\python.exe -m techrag eval --ablation --split test --repeat 3 --require-reviewed --output-dir reports/experiments
```

可以重复传入`--variant hybrid --variant mqe`只比较指定方案，或以`--dataset`使用复核后的副本。

首次试验曾发生DeepSeek HTTP 402余额不足，原始失败与降级记录仍保留。
恢复额度后已串行重跑原6条查询的七方案对比，42次执行全部完成；评分保留原10条有效结果，
仅补跑14条接口降级案例，合并后24条全部完成，并校验了模型、案例与原文证据。
结果见[真实API小规模报告](../reports/evidence/live-pilot.md)。
首次失败请求的快速返回不用于性能比较；新检索耗时来自独立串行试跑。
评分混合了首次10条与新补测14条，不用这组跨批次耗时得出性能或重复稳定性结论。

完整七方案开发实验为80×7×3=1680次问答pipeline，每个pipeline可能包含多次模型调用，
并默认额外评审回答；先用小样本估算时间与费用。关闭额外答案评审会留下检索指标。

每次重复都会保存原始benchmark JSON/Markdown、选中数据集快照、完整逐题轨迹。
总报告保存源数据集/选中数据集/索引指纹、模型、配置、重复次数、分类表现、失败及降级。
独立案例数量不会因重复运行而增加。先对同题配对差值在重复间取平均，再按知识关联组
bootstrap，至少10个组才报告95%区间。失败案例不参与双方成功的配对，需一起看失败率；
端到端Recall将可回答题处理失败记0，避免只报告成功样本造成误导。

`claim_ready`仅说明人工复核、测试集、至少三次重复、至少十组、无失败/降级等前置条件通过。
它不表示候选方案更好，不表示差值显著，更不等于生产准确率。还要检查差值区间、
拒答表现和质量指标。复合题的MRR/Precision/nDCG仍采用原有子问题统计口径，不能当成
逐子问题独立金标准的标准指标；详见原benchmark报告notes。

延迟包含共有拆解和问答pipeline，另列答案评审，不含HTTP排队。保留冷启动，不声称温启动性能。
应用层调用量不等于token费用。调优先查看慢阶段，再选择并行、缓存、条件扩展等措施，
在相同数据与配置下重测，不能为了简历加入未经测量的优化。

## 逐点评分评测

```powershell
.\.venv\Scripts\python.exe -m techrag score-eval --repeat 1
# 人工复核后
.\.venv\Scripts\python.exe -m techrag score-eval --repeat 3 --require-reviewed
```

评分调用生产`InterviewEngine._evaluate`：相同提示、首次及补充回答合并、连续原文与回答编号校验。
仅构造内存中的评分上下文，不保存会话、不更改题库、不重新评价历史面试。
逐个调用完成后保存checkpoint，最终保存JSON/Markdown及逐题证据。

记录全部评分点一致率、已核验点一致率、待核验率、宏F1、混淆矩阵、分类表现和重复一致性。
降级与无效证据仍在全部评分点的分母中。宏F1只平均有标注支持的类别。
重复一致性为同题状态向量的众数比例：稳定答错也会一致，所以必须同时检查标签符合度。
原文校验只证明出处，不能证明语义正确性。当前标注已由项目作者人工复核，
可称“与项目作者人工复核标签的逐点一致率”，同时保留样本规模、评价次数和合成回答范围。

## 面试流程与编辑可靠性

复用现有测试验证确定性约束，包括：

- 单次追问不消耗正式轮次，首次和补充回答共同评分。
- 不会/跳过会换题，排除同知识点与前置依赖；仅回答不完整不会永久屏蔽知识点。
- 请求解释不评分、不消耗轮次；高分仍可针对缺口追问。
- 历史恢复保留正在追问的上下文，提前结束仅结算一次。
- 引申检索使用回答上下文，不能突破题库范围，连续引申有上限。
- 重复提交和过期版本被控制；检索失败回滚本轮状态并可重试。
- 题库修改采用请求去重、版本冲突检查、索引/文件失败恢复，不覆盖原题库和旧评分。

这些测试使用模拟模型，验证流程约束；追问是否针对真实缺口还需要真人或独立评审评价。
不要将模拟模型测试通过率描述为真实Agent决策准确率。

```powershell
.\.venv\Scripts\python.exe -m pytest tests --basetemp .tmp/evidence-full --junitxml reports/evidence/pytest.xml -p no:cacheprovider
node tests/test_evaluation_ui.js
```

本地服务仍为单进程串行写入，评测会阻塞其他业务请求；当前没有并发吞吐量证明。
本次不把全局锁改成并发执行，以免provider包装计数、状态推进和Chroma写入产生新的竞争。
下一阶段若压测确认排队是瓶颈，先隔离评测worker，再测并发请求及失败恢复。
