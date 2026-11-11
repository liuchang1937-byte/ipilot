# InterviewPilot 本轮项目证据

- 完整Python回归：151项，失败0，错误0，跳过0；原始记录：[pytest.xml](pytest.xml)。
- 修订后评测相关测试：{'tests': 27, 'failures': 0, 'errors': 0}，记录：[targeted-tests.xml](targeted-tests.xml)。
- 前端评测展示测试通过：True，记录：[ui-check.log](ui-check.log)。
- Ruff检查通过：True，记录：[lint-check.log](lint-check.log)。

## 数据准备

- 检索120题，开发80/测试40，来源关联组80个；无跨集标注来源、无缺失来源。
- 评分24个回答案例，覆盖六类回答。
- 人工复核完成：True；标注状态详见dataset-check.json。
- [检索复核工作表](retrieval-review.md) / [评分复核工作表](scoring-review.md)。

## 实际离线结果

- BM25开发基线：80条原查询，69条有标注来源；来源Recall@3=0.8551，MRR=0.8261。
- 纯检索平均34.6毫秒；未拆解、未扩展、未生成，不是端到端时延。
- [原始结果与未完整命中案例](530edae547254b63ab56575dc4ef376b.md)。

## 完成情况与后续方向

1. 人工复核已完成：120条检索、24条评分，项目作者认可原标注，未修改来源或标签。
2. 正式开发、独立测试和回答质量报告均已保存，结合有效样本量解释结果。
3. 已按开发集规则冻结选型，并完成独立测试的三轮重复，原始报告与实现快照保留。
4. 后续可研究条件扩展、并行或缓存；当前报告不把这些未测量的改动写成优化成果。

失败的API尝试仅用于检查完成率，不能用于效果结论。
本地自动化测试使用模拟模型，仅验证流程和约束；真实追问质量仍需实际评价。

## 完整评测阶段

- 已完成阶段：['dev', 'test', 'quality']；三阶段完整完成：True。
- 启动状态：completed。
- [正式评测协议与调用范围](../../docs/full-evaluation-protocol.md)。
- 完整阶段与已完成的小规模API试跑分别记录，不将准备完成视为评测完成。

## 可直接使用的材料

- [实验命令与统计口径](../../docs/evaluation-plan.md)
- [架构图、演示脚本和简历草稿](../../docs/project-evidence.md)
- [可直接使用的简历项目条目](../../docs/resume-project.md)
- [本轮真实API实验与失败分析](live-pilot.md)

## 回归覆盖

| 模块 | 收集用例 |
| --- | --- |
| tests.test_answer_evaluation | 17 |
| tests.test_benchmark | 17 |
| tests.test_cli | 4 |
| tests.test_database | 8 |
| tests.test_experiments | 7 |
| tests.test_features | 10 |
| tests.test_full_evaluation | 3 |
| tests.test_interview | 23 |
| tests.test_interview_questions | 12 |
| tests.test_knowledge | 10 |
| tests.test_question_editor | 10 |
| tests.test_rag | 3 |
| tests.test_reranker | 3 |
| tests.test_scoring | 6 |
| tests.test_scoring_benchmark | 3 |
| tests.test_text | 3 |
| tests.test_web | 12 |
