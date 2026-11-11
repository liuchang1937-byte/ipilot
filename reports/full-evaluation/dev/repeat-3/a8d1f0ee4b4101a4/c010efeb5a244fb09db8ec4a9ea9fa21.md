# 检索评测 c010efeb5a244fb09db8ec4a9ea9fa21

- 时间：2026-10-12T11:23:08.280516+00:00
- 数据集：知识库检索分组种子集 v2 · 120题（human_reviewed）
- 题库指纹：516b55a5a68a38e4ee5dd0b94922d3658853545a6eebe13c0b293c3f7ba337da

## 检索质量

| 方案 | 完成 | Recall@3 | Precision@3 | Hit Rate@3 | MRR | nDCG@3 | 正确拒答 | 误拒答 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 混合 + MQE/HyDE | 1/1 | 1.0 | 0.5 | 1 | 1.0 | 0.8066 | None | None |
| 混合检索 | 1/1 | 1.0 | 0.5 | 1 | 1.0 | 0.8066 | None | None |
| 混合 + 精排 | 1/1 | 1.0 | 0.5 | 1 | 1.0 | 0.8066 | None | None |

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
| 混合 + MQE/HyDE | 27.347255 | 27.347255 | 27.347255 | 27.347255 | None |
| 混合检索 | 6.383043 | 6.383043 | 6.383043 | 6.383043 | None |
| 混合 + 精排 | 7.542667 | 7.542667 | 7.542667 | 7.542667 | None |

| 阶段（平均/P95） | 混合 + MQE/HyDE | 混合检索 | 混合 + 精排 |
| --- | --- | --- | --- |
| decomposition | 5.73824/5.73824 | 5.73824/5.73824 | 5.73824/5.73824 |
| retrieval | 21.608606/21.608606 | 0.644447/0.644447 | 0.475928/0.475928 |
| mqe | 7.316856/7.316856 | 0.0/0.0 | 0.0/0.0 |
| hyde | 12.903233/12.903233 | 0.0/0.0 | 0.0/0.0 |
| embedding | 0.706245/0.706245 | 0.437994/0.437994 | 0.292673/0.292673 |
| vector_search | 0.120938/0.120938 | 0.052796/0.052796 | 0.047004/0.047004 |
| keyword_search | 0.551014/0.551014 | 0.148325/0.148325 | 0.132411/0.132411 |
| rerank | 0.0/0.0 | 0.0/0.0 | 1.327857/1.327857 |
| generation | 0.0/0.0 | 0.0/0.0 | 0.0/0.0 |
| pipeline | 21.609015/21.609015 | 0.644802/0.644802 | 1.804426/1.804426 |
| total | 27.347255/27.347255 | 6.383043/6.383043 | 7.542667/7.542667 |
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

### V2-101 / expanded
注意力计算的显存如何优化？位置编码又解决什么不同问题？
- 耗时：27.347 秒；调用：{'embedding_calls': 2, 'embedding_texts': 10, 'chat_calls': 4, 'rerank_calls': 0, 'rerank_pairs': 0}
- 分阶段秒数：{'decomposition': 5.73824, 'retrieval': 21.608606, 'mqe': 7.316856, 'hyde': 12.903233, 'embedding': 0.706245, 'vector_search': 0.120938, 'keyword_search': 0.551014, 'rerank': 0.0, 'generation': 0.0, 'pipeline': 21.609015, 'total': 27.347255, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.5, True, 0.8065735963827292, 1.0]
- 标注来源：interview-question://L003, interview-question://L004
- 检索来源：interview-question://L003, interview-question://L002, interview-question://L014, interview-question://L004
1. 注意力计算的显存如何优化？


2. 位置编码解决什么不同于注意力计算显存优化的问题？


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

### V2-101 / hybrid
注意力计算的显存如何优化？位置编码又解决什么不同问题？
- 耗时：6.383 秒；调用：{'embedding_calls': 2, 'embedding_texts': 2, 'chat_calls': 0, 'rerank_calls': 0, 'rerank_pairs': 0}
- 分阶段秒数：{'decomposition': 5.73824, 'retrieval': 0.644447, 'mqe': 0.0, 'hyde': 0.0, 'embedding': 0.437994, 'vector_search': 0.052796, 'keyword_search': 0.148325, 'rerank': 0.0, 'generation': 0.0, 'pipeline': 0.644802, 'total': 6.383043, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.5, True, 0.8065735963827292, 1.0]
- 标注来源：interview-question://L003, interview-question://L004
- 检索来源：interview-question://L003, interview-question://L002, interview-question://L009, interview-question://L004
1. 注意力计算的显存如何优化？


2. 位置编码解决什么不同于注意力计算显存优化的问题？


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

### V2-101 / hybrid_reranked
注意力计算的显存如何优化？位置编码又解决什么不同问题？
- 耗时：7.543 秒；调用：{'embedding_calls': 2, 'embedding_texts': 2, 'chat_calls': 0, 'rerank_calls': 2, 'rerank_pairs': 24}
- 分阶段秒数：{'decomposition': 5.73824, 'retrieval': 0.475928, 'mqe': 0.0, 'hyde': 0.0, 'embedding': 0.292673, 'vector_search': 0.047004, 'keyword_search': 0.132411, 'rerank': 1.327857, 'generation': 0.0, 'pipeline': 1.804426, 'total': 7.542667, 'judge': 0.0}
- 答案评审调用：{'chat_calls': 0}
- Recall/Precision/Hit/nDCG/RR：[1.0, 0.5, True, 0.8065735963827292, 1.0]
- 标注来源：interview-question://L003, interview-question://L004
- 检索来源：interview-question://L003, interview-question://L002, interview-question://L009, interview-question://L004
1. 注意力计算的显存如何优化？


2. 位置编码解决什么不同于注意力计算显存优化的问题？


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
