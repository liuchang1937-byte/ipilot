# 完整评测：dev

- 独立问题：80；分组：54；重复：3。
- 模式：retrieval_only。

| 方案 | 完成 | Recall@3 | MRR | 平均流程秒 | P95秒 |
| --- | --- | --- | --- | --- | --- |
| 混合检索 | 240/240 | 0.9879 | 0.9477 | 5.314 | 12.164 |
| 混合 + 精排 | 240/240 | 0.9758 | 0.9235 | 6.376 | 13.369 |
| 混合 + MQE/HyDE | 240/240 | 0.9976 | 0.9718 | 26.986 | 60.282 |

## 配对比较

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

- 检索阶段三进程并行，每个案例内各方案串行并轮换顺序；耗时在此并发负载下测得。
- 纯检索不生成答案，不评价正确拒答或回答质量，也不将其耗时称为问答时延。
- 质量阶段单进程、独立40条测试问题、各一次生成与评审，不能当作三次稳定性实验。
- 重复记录先按案例汇总，配对区间按来源关联组bootstrap，避免重复样本扩大样本量。
- 断点保留原输入、调用量、耗时和成功响应，当前墙钟时间只代表本次启动。
