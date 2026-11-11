# 开发集预分析（非最终结果）

独立查询80条；仅采用完整轮次[1, 2]。

| 方案 | 完成记录 | 目标来源Recall@3 | MRR | 平均检索流程秒 | P95秒 |
| --- | --- | --- | --- | --- | --- |
| 混合检索 | 160/160 | 0.9855 | 0.9553 | 5.321 | 11.614 |
| 混合 + 精排 | 160/160 | 0.9783 | 0.9287 | 6.466 | 14.171 |
| 混合 + MQE/HyDE | 160/160 | 1.0 | 0.9734 | 26.265 | 51.696 |

## 配对差值与分组区间

```json
[
  {
    "baseline": "hybrid",
    "candidate": "hybrid_reranked",
    "metric": "recall_at_k",
    "delta": -0.007246,
    "paired_cases": 69,
    "paired_observations": 138,
    "independent_groups": 43,
    "bootstrap_95_ci": [
      -0.052239,
      0.026667
    ],
    "direction": "higher"
  },
  {
    "baseline": "hybrid",
    "candidate": "hybrid_reranked",
    "metric": "reciprocal_rank",
    "delta": -0.02657,
    "paired_cases": 69,
    "paired_observations": 138,
    "independent_groups": 43,
    "bootstrap_95_ci": [
      -0.073333,
      0.011538
    ],
    "direction": "higher"
  },
  {
    "baseline": "hybrid",
    "candidate": "hybrid_reranked",
    "metric": "elapsed_seconds",
    "delta": 1.144619,
    "paired_cases": 80,
    "paired_observations": 160,
    "independent_groups": 54,
    "bootstrap_95_ci": [
      1.012718,
      1.263303
    ],
    "direction": "lower"
  },
  {
    "baseline": "hybrid",
    "candidate": "expanded",
    "metric": "recall_at_k",
    "delta": 0.014493,
    "paired_cases": 69,
    "paired_observations": 138,
    "independent_groups": 43,
    "bootstrap_95_ci": [
      0.0,
      0.036364
    ],
    "direction": "higher"
  },
  {
    "baseline": "hybrid",
    "candidate": "expanded",
    "metric": "reciprocal_rank",
    "delta": 0.018116,
    "paired_cases": 69,
    "paired_observations": 138,
    "independent_groups": 43,
    "bootstrap_95_ci": [
      0.0,
      0.044974
    ],
    "direction": "higher"
  },
  {
    "baseline": "hybrid",
    "candidate": "expanded",
    "metric": "elapsed_seconds",
    "delta": 20.943938,
    "paired_cases": 80,
    "paired_observations": 160,
    "independent_groups": 54,
    "bootstrap_95_ci": [
      18.766705,
      23.124214
    ],
    "direction": "lower"
  }
]
```

- 仅使用完整轮次，各方案具有相同案例与重复数量；未完成的第三轮不参与。
- 未冻结选型，未查看测试结果，不能替代协议要求的三次开发与三次测试。
- 纯检索不生成答案，其总流程时间包含拆解、召回与精排，不是问答时延。
- 数据为项目作者人工复核的合成查询，按标注来源关联组bootstrap，不能泛化为生产表现。
