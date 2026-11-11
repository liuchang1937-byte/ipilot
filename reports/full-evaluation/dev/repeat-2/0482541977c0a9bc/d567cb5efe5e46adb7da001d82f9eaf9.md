# 检索评测 d567cb5efe5e46adb7da001d82f9eaf9

- 时间：2026-10-12T11:13:23.003720+00:00
- 数据集：知识库检索分组种子集 v2 · 120题（human_reviewed）
- 题库指纹：516b55a5a68a38e4ee5dd0b94922d3658853545a6eebe13c0b293c3f7ba337da

## 检索质量

| 方案 | 完成 | Recall@3 | Precision@3 | Hit Rate@3 | MRR | nDCG@3 | 正确拒答 | 误拒答 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 混合 + MQE/HyDE | 1/1 | 1.0 | 0.3333 | 1 | 1.0 | 1.0 | None | None |
| 混合检索 | 1/1 | 1.0 | 0.3333 | 1 | 1.0 | 0.9262 | None | None |
| 混合 + 精排 | 1/1 | 1.0 | 0.2667 | 1 | 1.0 | 0.8 | None | None |

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
| 混合 + MQE/HyDE | 64.694425 | 64.694425 | 64.694425 | 64.694425 | None |
| 混合检索 | 9.718991 | 9.718991 | 9.718991 | 9.718991 | None |
| 混合 + 精排 | 13.275122 | 13.275122 | 13.275122 | 13.275122 | None |

| 阶段（平均/P95） | 混合 + MQE/HyDE | 混合检索 | 混合 + 精排 |
| --- | --- | --- | --- |
| decomposition | 8.46625/8.46625 | 8.46625/8.46625 | 8.46625/8.46625 |
| retrieval | 56.227772/56.227772 | 1.252242/1.252242 | 1.484319/1.484319 |
| mqe | 17.618804/17.618804 | 0.0/0.0 | 0.0/0.0 |
| hyde | 34.68193/34.68193 | 0.0/0.0 | 0.0/0.0 |
| embedding | 2.313211/2.313211 | 0.819122/0.819122 | 0.990532/0.990532 |
| vector_search | 0.284214/0.284214 | 0.107125/0.107125 | 0.135541/0.135541 |
| keyword_search | 1.306373/1.306373 | 0.319846/0.319846 | 0.351182/0.351182 |
| rerank | 0.0/0.0 | 0.0/0.0 | 3.323645/3.323645 |
| generation | 0.0/0.0 | 0.0/0.0 | 0.0/0.0 |
| pipeline | 56.228176/56.228176 | 1.252741/1.252741 | 4.808872/4.808872 |
| total | 64.694425/64.694425 | 9.718991/9.718991 | 13.275122/13.275122 |
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

### V2-007 / expanded
一个请求包含多个依赖步骤，应该如何拆成可验证的小任务？
- 耗时：64.694 秒；调用：{'embedding_calls': 5, 'embedding_texts': 25, 'chat_calls': 10, 'rerank_calls': 0, 'rerank_pairs': 0}
- 分阶段秒数：{'decomposition': 8.46625, 'retrieval': 56.227772, 'mqe': 17.618804, 'hyde': 34.68193, 'embedding': 2.313211, 'vector_search': 0.284214, 'keyword_search': 1.306373, 'rerank': 0.0, 'generation': 0.0, 'pipeline': 56.228176, 'total': 64.694425, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.3333333333333333, True, 1.0, 1.0]
- 标注来源：interview-question://A007
- 检索来源：interview-question://A007, interview-question://L017, interview-question://A023, interview-question://B003, interview-question://A022, interview-question://A019
1. 对于包含多个依赖步骤的请求，拆成可验证小任务的总体方法和步骤是什么？


2. 对于包含多个依赖步骤的请求，如何确定每个小任务的范围和边界？


3. 对于包含多个依赖步骤的请求，如何为每个小任务定义可验证的完成标准？


4. 对于包含多个依赖步骤的请求，如何表示和处理步骤之间的依赖关系？


5. 对于包含多个依赖步骤的请求，如何判断拆分后的小任务是否满足可验证要求？


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

### V2-007 / hybrid
一个请求包含多个依赖步骤，应该如何拆成可验证的小任务？
- 耗时：9.719 秒；调用：{'embedding_calls': 5, 'embedding_texts': 5, 'chat_calls': 0, 'rerank_calls': 0, 'rerank_pairs': 0}
- 分阶段秒数：{'decomposition': 8.46625, 'retrieval': 1.252242, 'mqe': 0.0, 'hyde': 0.0, 'embedding': 0.819122, 'vector_search': 0.107125, 'keyword_search': 0.319846, 'rerank': 0.0, 'generation': 0.0, 'pipeline': 1.252741, 'total': 9.718991, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.3333333333333333, True, 0.9261859507142916, 1.0]
- 标注来源：interview-question://A007
- 检索来源：interview-question://A007, interview-question://L017, interview-question://A023, interview-question://B003, interview-question://B006, interview-question://A018, interview-question://A022, interview-question://A019
1. 对于包含多个依赖步骤的请求，拆成可验证小任务的总体方法和步骤是什么？


2. 对于包含多个依赖步骤的请求，如何确定每个小任务的范围和边界？


3. 对于包含多个依赖步骤的请求，如何为每个小任务定义可验证的完成标准？


4. 对于包含多个依赖步骤的请求，如何表示和处理步骤之间的依赖关系？


5. 对于包含多个依赖步骤的请求，如何判断拆分后的小任务是否满足可验证要求？


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

### V2-007 / hybrid_reranked
一个请求包含多个依赖步骤，应该如何拆成可验证的小任务？
- 耗时：13.275 秒；调用：{'embedding_calls': 5, 'embedding_texts': 5, 'chat_calls': 0, 'rerank_calls': 5, 'rerank_pairs': 60}
- 分阶段秒数：{'decomposition': 8.46625, 'retrieval': 1.484319, 'mqe': 0.0, 'hyde': 0.0, 'embedding': 0.990532, 'vector_search': 0.135541, 'keyword_search': 0.351182, 'rerank': 3.323645, 'generation': 0.0, 'pipeline': 4.808872, 'total': 13.275122, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.26666666666666666, True, 0.8, 1.0]
- 标注来源：interview-question://A007
- 检索来源：interview-question://A007, interview-question://A011, interview-question://A018, interview-question://T009, interview-question://A014, interview-question://A022
1. 对于包含多个依赖步骤的请求，拆成可验证小任务的总体方法和步骤是什么？


2. 对于包含多个依赖步骤的请求，如何确定每个小任务的范围和边界？


3. 对于包含多个依赖步骤的请求，如何为每个小任务定义可验证的完成标准？


4. 对于包含多个依赖步骤的请求，如何表示和处理步骤之间的依赖关系？


5. 对于包含多个依赖步骤的请求，如何判断拆分后的小任务是否满足可验证要求？


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
