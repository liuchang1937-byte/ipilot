# 检索评测 ac697abdc36f460eaaccc97841d96abe

- 时间：2026-10-12T15:11:01.268873+00:00
- 数据集：知识库检索分组种子集 v2 · 120题（human_reviewed）
- 题库指纹：516b55a5a68a38e4ee5dd0b94922d3658853545a6eebe13c0b293c3f7ba337da

## 检索质量

| 方案 | 完成 | Recall@3 | Precision@3 | Hit Rate@3 | MRR | nDCG@3 | 正确拒答 | 误拒答 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 混合 + MQE/HyDE | 1/1 | 1.0 | 0.25 | 1 | 1.0 | 0.6577 | None | None |
| 混合检索 | 1/1 | 1.0 | 0.25 | 1 | 1.0 | 0.6577 | None | None |
| 混合 + 精排 | 1/1 | 1.0 | 0.3333 | 1 | 1.0 | 0.9077 | None | None |

## 回答质量

分数后括号为有效样本数；None为未评估。

| 方案 | 忠实度 | 正确性F1 | 相关性 | 引用准确性 | 引用覆盖 | 评审完成/失败 |
| --- | --- | --- | --- | --- | --- | --- |
| 混合 + MQE/HyDE | None (0) | None (0) | None (0) | None (0) | None (0) | 0/0 |
| 混合检索 | None (0) | None (0) | None (0) | None (0) | None (0) | 0/0 |
| 混合 + 精排 | None (0) | None (0) | None (0) | None (0) | None (0) | 0/0 |

## 具体时延（秒）

| 方案 | 平均 | P50 | P95 | 最大 | 评审平均（另计） |
| --- | --- | --- | --- | --- | --- |
| 混合 + MQE/HyDE | 64.066005 | 64.066005 | 64.066005 | 64.066005 | None |
| 混合检索 | 12.707255 | 12.707255 | 12.707255 | 12.707255 | None |
| 混合 + 精排 | 14.016201 | 14.016201 | 14.016201 | 14.016201 | None |

| 阶段（平均/P95） | 混合 + MQE/HyDE | 混合检索 | 混合 + 精排 |
| --- | --- | --- | --- |
| decomposition | 11.926837/11.926837 | 11.926837/11.926837 | 11.926837/11.926837 |
| retrieval | 52.138942/52.138942 | 0.780151/0.780151 | 0.867551/0.867551 |
| mqe | 21.262297/21.262297 | 0.0/0.0 | 0.0/0.0 |
| hyde | 28.797026/28.797026 | 0.0/0.0 | 0.0/0.0 |
| embedding | 1.475115/1.475115 | 0.604985/0.604985 | 0.693977/0.693977 |
| vector_search | 0.116564/0.116564 | 0.053456/0.053456 | 0.049193/0.049193 |
| keyword_search | 0.479049/0.479049 | 0.119317/0.119317 | 0.121157/0.121157 |
| rerank | 0.0/0.0 | 0.0/0.0 | 1.221284/1.221284 |
| generation | 0.0/0.0 | 0.0/0.0 | 0.0/0.0 |
| pipeline | 52.139168/52.139168 | 0.780418/0.780418 | 2.089364/2.089364 |
| total | 64.066005/64.066005 | 12.707255/12.707255 | 14.016201/14.016201 |
| judge | None/None | None/None | None/None |

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

### V2-079 / expanded
函数调用格式、任务技能包和工具连接协议如何在一个系统里配合？
- 耗时：64.066 秒；调用：{'embedding_calls': 4, 'embedding_texts': 19, 'chat_calls': 8, 'rerank_calls': 0, 'rerank_pairs': 0}
- 分阶段秒数：{'decomposition': 11.926837, 'retrieval': 52.138942, 'mqe': 21.262297, 'hyde': 28.797026, 'embedding': 1.475115, 'vector_search': 0.116564, 'keyword_search': 0.479049, 'rerank': 0.0, 'generation': 0.0, 'pipeline': 52.139168, 'total': 64.066005, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.25, True, 0.6577324383928644, 1.0]
- 标注来源：interview-question://T011
- 检索来源：interview-question://T011, interview-question://T003, interview-question://T010, interview-question://T006, interview-question://T001, interview-question://T007, interview-question://T009
1. 在一个系统中，函数调用格式如何与任务技能包配合？


2. 在一个系统中，函数调用格式如何与工具连接协议配合？


3. 在一个系统中，任务技能包如何与工具连接协议配合？


4. 在一个系统中，函数调用格式、任务技能包和工具连接协议三者如何协同配合？


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

### V2-079 / hybrid
函数调用格式、任务技能包和工具连接协议如何在一个系统里配合？
- 耗时：12.707 秒；调用：{'embedding_calls': 4, 'embedding_texts': 4, 'chat_calls': 0, 'rerank_calls': 0, 'rerank_pairs': 0}
- 分阶段秒数：{'decomposition': 11.926837, 'retrieval': 0.780151, 'mqe': 0.0, 'hyde': 0.0, 'embedding': 0.604985, 'vector_search': 0.053456, 'keyword_search': 0.119317, 'rerank': 0.0, 'generation': 0.0, 'pipeline': 0.780418, 'total': 12.707255, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.25, True, 0.6577324383928644, 1.0]
- 标注来源：interview-question://T011
- 检索来源：interview-question://T011, interview-question://T003, interview-question://T010, interview-question://T006, interview-question://T007, interview-question://T004
1. 在一个系统中，函数调用格式如何与任务技能包配合？


2. 在一个系统中，函数调用格式如何与工具连接协议配合？


3. 在一个系统中，任务技能包如何与工具连接协议配合？


4. 在一个系统中，函数调用格式、任务技能包和工具连接协议三者如何协同配合？


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

### V2-079 / hybrid_reranked
函数调用格式、任务技能包和工具连接协议如何在一个系统里配合？
- 耗时：14.016 秒；调用：{'embedding_calls': 4, 'embedding_texts': 4, 'chat_calls': 0, 'rerank_calls': 4, 'rerank_pairs': 48}
- 分阶段秒数：{'decomposition': 11.926837, 'retrieval': 0.867551, 'mqe': 0.0, 'hyde': 0.0, 'embedding': 0.693977, 'vector_search': 0.049193, 'keyword_search': 0.121157, 'rerank': 1.221284, 'generation': 0.0, 'pipeline': 2.089364, 'total': 14.016201, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.3333333333333333, True, 0.9077324383928644, 1.0]
- 标注来源：interview-question://T011
- 检索来源：interview-question://T011, interview-question://T003, interview-question://T001, interview-question://T006, interview-question://T012, interview-question://T010, interview-question://A003
1. 在一个系统中，函数调用格式如何与任务技能包配合？


2. 在一个系统中，函数调用格式如何与工具连接协议配合？


3. 在一个系统中，任务技能包如何与工具连接协议配合？


4. 在一个系统中，函数调用格式、任务技能包和工具连接协议三者如何协同配合？


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
