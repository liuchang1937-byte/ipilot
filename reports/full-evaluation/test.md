# 完整评测：test

- 独立问题：40；分组：26；重复：3。
- 模式：retrieval_only。

| 方案 | 完成 | Recall@3 | MRR | 平均流程秒 | P95秒 |
| --- | --- | --- | --- | --- | --- |
| 混合检索 | 120/120 | 1.0 | 0.9775 | 5.143 | 9.319 |
| 混合 + MQE/HyDE | 120/120 | 1.0 | 1.0 | 26.094 | 40.114 |

## 配对比较

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

- 检索阶段三进程并行，每个案例内各方案串行并轮换顺序；耗时在此并发负载下测得。
- 纯检索不生成答案，不评价正确拒答或回答质量，也不将其耗时称为问答时延。
- 质量阶段单进程、独立40条测试问题、各一次生成与评审，不能当作三次稳定性实验。
- 重复记录先按案例汇总，配对区间按来源关联组bootstrap，避免重复样本扩大样本量。
- 断点保留原输入、调用量、耗时和成功响应，当前墙钟时间只代表本次启动。
