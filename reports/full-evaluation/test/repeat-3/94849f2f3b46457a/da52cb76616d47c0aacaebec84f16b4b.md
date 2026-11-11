# 检索评测 da52cb76616d47c0aacaebec84f16b4b

- 时间：2026-10-12T15:30:03.996708+00:00
- 数据集：知识库检索分组种子集 v2 · 120题（human_reviewed）
- 题库指纹：516b55a5a68a38e4ee5dd0b94922d3658853545a6eebe13c0b293c3f7ba337da

## 检索质量

| 方案 | 完成 | Recall@3 | Precision@3 | Hit Rate@3 | MRR | nDCG@3 | 正确拒答 | 误拒答 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 混合 + MQE/HyDE | 1/1 | 1.0 | 0.3333 | 1 | 1.0 | 0.6131 | None | None |
| 混合检索 | 1/1 | 1.0 | 0.3333 | 1 | 1.0 | 0.6131 | None | None |

## 回答质量

分数后括号为有效样本数；None为未评估。

| 方案 | 忠实度 | 正确性F1 | 相关性 | 引用准确性 | 引用覆盖 | 评审完成/失败 |
| --- | --- | --- | --- | --- | --- | --- |
| 混合 + MQE/HyDE | None (0) | None (0) | None (0) | None (0) | None (0) | 0/0 |
| 混合检索 | None (0) | None (0) | None (0) | None (0) | None (0) | 0/0 |

## 具体时延（秒）

| 方案 | 平均 | P50 | P95 | 最大 | 评审平均（另计） |
| --- | --- | --- | --- | --- | --- |
| 混合 + MQE/HyDE | 24.415246 | 24.415246 | 24.415246 | 24.415246 | None |
| 混合检索 | 3.885611 | 3.885611 | 3.885611 | 3.885611 | None |

| 阶段（平均/P95） | 混合 + MQE/HyDE | 混合检索 |
| --- | --- | --- |
| decomposition | 3.48721/3.48721 | 3.48721/3.48721 |
| retrieval | 20.927899/20.927899 | 0.398276/0.398276 |
| mqe | 6.84026/6.84026 | 0.0/0.0 |
| hyde | 13.064454/13.064454 | 0.0/0.0 |
| embedding | 0.72908/0.72908 | 0.318903/0.318903 |
| vector_search | 0.059091/0.059091 | 0.022607/0.022607 |
| keyword_search | 0.230626/0.230626 | 0.055174/0.055174 |
| rerank | 0.0/0.0 | 0.0/0.0 |
| generation | 0.0/0.0 | 0.0/0.0 |
| pipeline | 20.928036/20.928036 | 0.398401/0.398401 |
| total | 24.415246/24.415246 | 3.885611/3.885611 |
| judge | None/None | None/None |

- 标注状态记录于dataset.label_status；人工复核不等于生产数据或独立多复核人基准。
- 每个子问题 K=3；复合问题 Recall 统计全部标注来源覆盖率。MRR 为首个相关来源的最佳子问题倒数排名。
- Precision@K 固定以K为分母，缺少结果记为未命中；对子问题求平均。nDCG@K 对子问题求平均，使用整题来源标注，复合题不等同于逐子问标注的标准nDCG。默认二元相关性；source_relevance 可指定1到3级相关程度。
- 标注仅列目标来源，未穷举全部等价证据；Precision和nDCG可能低估实际相关性。
- 失败案例不计效果均值；请结合完成率、降级率和逐题结果判断。
- 耗时包含查询拆解、扩展、检索、精排和生成，以及首次模型加载；方案执行顺序轮换。
- 每题拆解只执行一次，所有方案复用同一子问题计划以控制变量。共有拆解耗时计入各方案耗时；共有调用单独记录，不重复计入方案调用量。
- 分阶段耗时单位为秒，retrieval包含MQE、HyDE、Embedding、向量/BM25检索及融合，这些子阶段不能再与retrieval相加。total=共有拆解+pipeline，不含答案评审、HTTP排队或报告写盘；本接口非流式，不记录首Token延迟。judge耗时与调用单列。
- Faithfulness=有资料支持的事实数/事实数；引用准确性按事实—引用对统计，引用覆盖率=带引用的事实数/事实数。拒答无事实时这些指标为未评估。
- 答案正确性为正确事实精确率与参考要点覆盖率的F1；无参考答案/要点时不评。相关性由评审模型按0到1打分；模型评审与生成分开调用，默认使用同一模型。程序校验连续原文、证据出处和结构，语义判断、事实拆分完整性仍需人工复核。评审失败不影响检索结果，不计入质量均值；每项指标单列有效样本数。
- 调用量为应用层调用次数，不是 token 用量或供应商内部重试次数。引用存在不等于结论受支持，答案正确性仍需复核。
- 断点复用相同输入、配置、代码与题库的已成功结果，失败项重新执行；各记录保留原耗时与调用量，benchmark_wall_seconds仅为当前启动的墙钟时间。
- retrieval_only只运行真实检索链路，未生成答案，不评拒答或回答质量；此模式下total是检索流程耗时，不能当作问答时延。

## 逐题结果

### V2-100 / expanded
模型网关提供哪些治理能力？助手连接数据库又如何限制权限？
- 耗时：24.415 秒；调用：{'embedding_calls': 2, 'embedding_texts': 10, 'chat_calls': 4, 'rerank_calls': 0, 'rerank_pairs': 0}
- 分阶段秒数：{'decomposition': 3.48721, 'retrieval': 20.927899, 'mqe': 6.84026, 'hyde': 13.064454, 'embedding': 0.72908, 'vector_search': 0.059091, 'keyword_search': 0.230626, 'rerank': 0.0, 'generation': 0.0, 'pipeline': 20.928036, 'total': 24.415246, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.3333333333333333, True, 0.6131471927654584, 1.0]
- 标注来源：interview-question://T016, interview-question://A024
- 检索来源：interview-question://T016, interview-question://T004, interview-question://T005, interview-question://A024, C:\Users\orzlc\Desktop\test.pdf
1. 模型网关提供哪些治理能力？


2. 助手连接数据库时如何限制权限？


```json
{
  "status": "disabled",
  "reason": "已关闭答案评审。",
  "metrics": {
    "faithfulness": null,
    "answer_correctness": null,
    "answer_relevancy": null,
    "citation_precision": null,
    "citation_coverage": null
  }
}
```

### V2-100 / hybrid
模型网关提供哪些治理能力？助手连接数据库又如何限制权限？
- 耗时：3.886 秒；调用：{'embedding_calls': 2, 'embedding_texts': 2, 'chat_calls': 0, 'rerank_calls': 0, 'rerank_pairs': 0}
- 分阶段秒数：{'decomposition': 3.48721, 'retrieval': 0.398276, 'mqe': 0.0, 'hyde': 0.0, 'embedding': 0.318903, 'vector_search': 0.022607, 'keyword_search': 0.055174, 'rerank': 0.0, 'generation': 0.0, 'pipeline': 0.398401, 'total': 3.885611, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.3333333333333333, True, 0.6131471927654584, 1.0]
- 标注来源：interview-question://T016, interview-question://A024
- 检索来源：interview-question://T016, interview-question://T004, interview-question://T005, interview-question://A024, C:\Users\orzlc\Desktop\test.pdf, interview-question://R009
1. 模型网关提供哪些治理能力？


2. 助手连接数据库时如何限制权限？


```json
{
  "status": "disabled",
  "reason": "已关闭答案评审。",
  "metrics": {
    "faithfulness": null,
    "answer_correctness": null,
    "answer_relevancy": null,
    "citation_precision": null,
    "citation_coverage": null
  }
}
```
