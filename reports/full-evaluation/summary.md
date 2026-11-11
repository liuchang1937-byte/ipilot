# 完整评测状态与结果

| 阶段 | 案例轮次完成 | 已保存方案结果 | 失败 | 带提示 | 有效/无效评审 |
| --- | --- | --- | --- | --- | --- |
| dev | 240/240 | 720/720 | 0 | 0 | 0/0 |
| test | 120/120 | 240/240 | 0 | 0 | 0/0 |
| quality | 40/40 | 40/40 | 0 | 4 | 40/0 |

## dev

独立问题80，来源关联组54，重复3，模式retrieval_only。

| 方案 | Recall@3 | MRR | 平均流程秒 | P95秒 | 完成 |
| --- | --- | --- | --- | --- | --- |
| 混合检索 | 0.9879 | 0.9477 | 5.314 | 12.164 | 240/240 |
| 混合 + 精排 | 0.9758 | 0.9235 | 6.376 | 13.369 | 240/240 |
| 混合 + MQE/HyDE | 0.9976 | 0.9718 | 26.986 | 60.282 | 240/240 |

配对差值和分组置信区间：

```json
[
  {
    "baseline": "hybrid",
    "candidate": "hybrid_reranked",
    "metric": "recall_at_k",
    "delta": -0.012077,
    "paired_cases": 69,
    "paired_observations": 207,
    "independent_groups": 43,
    "bootstrap_95_ci": [
      -0.057778,
      0.018779
    ],
    "direction": "higher"
  },
  {
    "baseline": "hybrid",
    "candidate": "hybrid_reranked",
    "metric": "reciprocal_rank",
    "delta": -0.024155,
    "paired_cases": 69,
    "paired_observations": 207,
    "independent_groups": 43,
    "bootstrap_95_ci": [
      -0.068376,
      0.015152
    ],
    "direction": "higher"
  },
  {
    "baseline": "hybrid",
    "candidate": "hybrid_reranked",
    "metric": "elapsed_seconds",
    "delta": 1.0629,
    "paired_cases": 80,
    "paired_observations": 240,
    "independent_groups": 54,
    "bootstrap_95_ci": [
      0.942065,
      1.170943
    ],
    "direction": "lower"
  },
  {
    "baseline": "hybrid",
    "candidate": "expanded",
    "metric": "recall_at_k",
    "delta": 0.009662,
    "paired_cases": 69,
    "paired_observations": 207,
    "independent_groups": 43,
    "bootstrap_95_ci": [
      0.0,
      0.024242
    ],
    "direction": "higher"
  },
  {
    "baseline": "hybrid",
    "candidate": "expanded",
    "metric": "reciprocal_rank",
    "delta": 0.024155,
    "paired_cases": 69,
    "paired_observations": 207,
    "independent_groups": 43,
    "bootstrap_95_ci": [
      0.004566,
      0.052632
    ],
    "direction": "higher"
  },
  {
    "baseline": "hybrid",
    "candidate": "expanded",
    "metric": "elapsed_seconds",
    "delta": 21.672875,
    "paired_cases": 80,
    "paired_observations": 240,
    "independent_groups": 54,
    "bootstrap_95_ci": [
      19.320275,
      23.807048
    ],
    "direction": "lower"
  }
]
```

## test

独立问题40，来源关联组26，重复3，模式retrieval_only。

| 方案 | Recall@3 | MRR | 平均流程秒 | P95秒 | 完成 |
| --- | --- | --- | --- | --- | --- |
| 混合检索 | 1.0 | 0.9775 | 5.143 | 9.319 | 120/120 |
| 混合 + MQE/HyDE | 1.0 | 1.0 | 26.094 | 40.114 | 120/120 |

配对差值和分组置信区间：

```json
[
  {
    "baseline": "hybrid",
    "candidate": "expanded",
    "metric": "recall_at_k",
    "delta": 0.0,
    "paired_cases": 37,
    "paired_observations": 111,
    "independent_groups": 23,
    "bootstrap_95_ci": [
      0.0,
      0.0
    ],
    "direction": "higher"
  },
  {
    "baseline": "hybrid",
    "candidate": "expanded",
    "metric": "reciprocal_rank",
    "delta": 0.022523,
    "paired_cases": 37,
    "paired_observations": 111,
    "independent_groups": 23,
    "bootstrap_95_ci": [
      0.0,
      0.061905
    ],
    "direction": "higher"
  },
  {
    "baseline": "hybrid",
    "candidate": "expanded",
    "metric": "elapsed_seconds",
    "delta": 20.951067,
    "paired_cases": 40,
    "paired_observations": 120,
    "independent_groups": 26,
    "bootstrap_95_ci": [
      18.441284,
      23.254348
    ],
    "direction": "lower"
  }
]
```

## quality

独立问题40，来源关联组26，重复1，模式end_to_end。

| 方案 | Recall@3 | MRR | 平均流程秒 | P95秒 | 完成 |
| --- | --- | --- | --- | --- | --- |
| 混合 + MQE/HyDE | 1.0 | 0.9865 | 40.335 | 72.572 | 40/40 |

自动回答质量评审（独立调用，默认与生成使用同一模型）：

评审完成40/40，评审失败0。

| 指标 | 均值 | 有效样本 |
| --- | --- | --- |
| 事实受资料支持比例 | 0.996 | 37 |
| 事实精确率与参考要点覆盖F1（自动） | 0.6667 | 37 |
| 回答相关性（模型评分） | 0.9587 | 40 |
| 引用支持比例 | 0.9965 | 37 |
| 事实引用覆盖率 | 0.8798 | 37 |
| 无资料题资料不足状态比例 | 1 | 3 |
| 可回答题全部分节资料不足比例 | 0 | 37 |

首轮有效35/40；累计46次评审尝试，其中6次未取得有效评审。上表为断点恢复后的最终记录，不表示单次请求成功率100%。
只重试无效项，原35条有效记录经比对完全一致；重试重新生成该条回答。低分和引用拦截项保留，详见[恢复分析](analysis.md)。

配对差值和分组置信区间：

```json
[]
```

## 统计口径

- 未保存整阶段报告时只展示进度，不给最终选型或效果结论。
- 检索阶段不生成答案，其流程耗时不能称为问答时延，也不评价语义拒答。
- retrieval计时字段对应召回阶段，精排另计；总流程包含拆解、召回和精排。
- 检索耗时在三个并行案例进程负载下测得；质量阶段为单进程。
- 重复次数不增加独立样本数量，差值区间按来源关联组bootstrap。
- Recall@3按各子问题Top 3目标来源合并命中，复合题MRR取最佳子问题倒数排名，不是整条复合问题的全局Top 3排名口径。
- 原标注人工复核已完成；生成答案的质量评审为同模型独立调用的自动辅助评审。
- 程序校验结构和原文出处，不能证明语义正确；质量指标不标为人工准确率。
- 质量指标先按每条回答计算，再对有有效值的回答取均值；有效样本列为回答数。
- 本数据集部分参考要点是完整段落，二元覆盖判断较粗；其F1不能解释为真人答案准确率。
