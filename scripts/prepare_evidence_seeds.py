"""Rebuild pending seed datasets from curated independent queries and local answers.

Do not rerun after human review: it intentionally regenerates pending seed labels.
Run from the repository root with the project's Python environment.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from techrag.interview_questions import load_question_directory

ROOT = Path(__file__).resolve().parents[1]

QUERIES = {
    "A001": "聊天模型能回答问题就算智能体吗？执行任务还缺哪些能力？",
    "A002": "准备从零实现任务助手，模型之外需要规划、工具和哪些模块？",
    "A003": "固定流程、会自主决定下一步的助手和外部函数，分别承担什么职责？",
    "A004": "审批步骤很固定的业务，应该让模型自由规划还是预先定义流程？",
    "A005": "助手先推理再调用工具、读完结果继续推理，这个循环怎么落地？",
    "A006": "边想边做、先规划后执行和完成后反思，在延迟和可控性上有什么取舍？",
    "A007": "一个请求包含多个依赖步骤，应该如何拆成可验证的小任务？",
    "A008": "助手要记住用户偏好和当前任务进度，记忆模块如何设计？",
    "A009": "正在进行的任务和几周前的历史信息，应该用同一种方式保存吗？",
    "A010": "让多个专门的助手共同完成一个任务，核心组织方式是什么？",
    "A011": "什么情况下一个助手已经足够，什么情况下值得拆成多个角色？",
    "A012": "历史对话放不下了，除了截掉旧消息还有哪些压缩方法？",
    "A013": "为何有的团队不采用现成智能体框架，选择自己实现执行循环？",
    "A014": "如何让模型产出可执行计划，并在执行中调整步骤？",
    "A015": "让模型检查自己的结果再修改，何时有收益、怎样避免反复自我修正？",
    "A016": "不同专长的助手如何交接任务，什么时候动态切换角色？",
    "A017": "工具描述、历史对话、检索资料与当前目标如何组织进上下文？",
    "A018": "对话进行一半服务重启，如何恢复任务阶段并防止重复执行？",
    "A019": "任务助手上线前，如何设计案例与指标检验它是否真的完成任务？",
    "A020": "助手反复选同一个工具且没有推进，怎样识别并终止这种循环？",
    "A021": "一次任务耗时突然翻倍，如何从执行轨迹定位慢在哪一环？",
    "A022": "多个助手并行写同一结果，其中一个超时，如何处理交接与冲突？",
    "A023": "助手说文件已经保存，但没有任何工具执行记录，如何避免这种情况？",
    "A024": "给模型接数据库时，如何限制查询权限并防止敏感字段泄漏？",
    "L001": "生成式语言模型与传统文本分类或序列标注模型有何不同？",
    "L002": "Transformer中的编码器、解码器和注意力层分别做什么？",
    "L003": "推理时注意力占用太多显存，MQA、GQA与Flash Attention各优化什么？",
    "L004": "模型为何需要知道词的位置？旋转位置编码和线性偏置有什么差别？",
    "L005": "文字送入模型前如何转换为token，分词方式为何影响长度与成本？",
    "L006": "从大量原始文本到能遵循指令的模型，通常经历哪些训练阶段？",
    "L007": "扩大数据、参数和计算量时能力如何变化，所谓涌现如何解释？",
    "L008": "只有有限显存，如何在全参数、LoRA等微调方式之间选择？",
    "L009": "用低秩矩阵适配模型，除了少训练参数还有什么工程收益？",
    "L010": "指令微调完成之后，偏好优化和拒绝采样各解决哪些问题？",
    "L011": "直接用偏好数据优化与基于奖励模型做PPO，有哪些训练差异？",
    "L012": "生成答案时，确定性选词、束搜索和随机采样如何取舍？",
    "L013": "temperature、top_p与top_k对生成分布各有什么影响？",
    "L014": "KV缓存与重复提示缓存分别复用了什么，为什么能减少计算？",
    "L015": "部署时怎样选择四位或八位量化，AWQ和GPTQ如何权衡？",
    "L016": "想让模型稳定输出结构化结果，提示中应明确哪些约束？",
    "L017": "要求模型展示中间推理过程有什么价值，为什么也会增加风险与成本？",
    "L018": "模型为什么会编造事实，检索与验证分别能缓解哪些问题？",
    "L019": "混合专家模型为何只激活部分参数，路由和负载均衡有哪些代价？",
    "L020": "CPU本地部署与GPU高并发服务，怎样选择推理运行框架？",
    "L021": "评估语言模型时，除了通用考试分数还需要测哪些维度？",
    "L022": "为业务选择模型时如何比较效果、延迟、费用与部署限制？",
    "L023": "资料明明在长提示里却被忽略，为什么中间位置容易丢失信息？",
    "R001": "从资料入库到带证据回答，检索增强系统的完整数据链路是什么？",
    "R002": "模型不知道企业最新资料，用检索增强能解决哪些限制？",
    "R003": "知识经常更新且需要出处，检索增强与参数微调该怎么选？",
    "R004": "长文档怎么拆成可检索片段，片段长度与重叠如何权衡？",
    "R005": "检索到了答案的一半，另一半在相邻片段，怎样保留完整语义？",
    "R006": "选文本向量模型时，如何验证它适合中文技术题检索？",
    "R007": "除了神经网络向量，文本表示与相似匹配还用过哪些算法？",
    "R008": "为向量索引选择存储引擎，需要比较哪些功能和工程约束？",
    "R009": "向量库的规模和响应速度应该怎样测量，瓶颈通常怎么排查？",
    "R010": "用户提交问题后，系统如何依次检索资料并构造生成提示？",
    "R011": "精确型号匹配和同义表达匹配，关键词搜索与语义搜索各有什么长处？",
    "R012": "用户问法太口语化，怎样改写查询且不改变原始需求？",
    "R013": "同时走关键词与向量召回后，如何融合排名并去重？",
    "R014": "召回效果不好时，应该从查询、切块、混合召回还是重排开始检查？",
    "R015": "Self-RAG、Corrective RAG和Agentic RAG控制流程有何差别？",
    "R016": "问题要沿实体依赖关系查多跳路径，图检索比纯向量有什么优势？",
    "R017": "检索到了材料，模型仍然编造结论，如何约束证据和拒答？",
    "R018": "如何分别衡量检索命中、排序和最终回答质量？",
    "R019": "资料修改或删除后，怎样让关键词与向量索引持续保持最新？",
    "R020": "检索增强落地时，数据质量、权限和评测哪些难点最容易被忽视？",
    "R021": "两篇检索材料给出矛盾结论，系统应按什么依据选择或展示差异？",
    "T001": "模型给出函数名与参数之后，谁真正执行函数并回填结果？",
    "T002": "语言模型为何能够生成工具调用请求，这项能力从哪里来？",
    "T003": "训练一个可靠工具调用模型，需要哪些数据与执行反馈？",
    "T004": "想用统一接口把外部资源和工具接给助手，MCP规范了哪些事情？",
    "T005": "MCP里的Host、Client、Server分别承担什么职责？",
    "T006": "模型选择函数与MCP提供工具连接，为什么不是同一层的机制？",
    "T007": "仅接一个内部函数与接多个外部服务，何时采用MCP更合适？",
    "T008": "推理模型没有原生工具格式就无法接MCP吗，适配层应该放在哪里？",
    "T009": "可按需加载的任务说明与配套资源，如何帮助助手遵循操作流程？",
    "T010": "提供工具访问协议与提供任务操作指南，MCP和Skill各负责什么？",
    "T011": "函数调用格式、任务技能包和工具连接协议如何在一个系统里配合？",
    "T012": "助手之间交接任务与助手访问工具，A2A和MCP的边界是什么？",
    "T013": "MCP在本地进程与远程服务之间怎样传输消息？",
    "T014": "服务器持续推送文本与双向交互，SSE和WebSocket有什么差异？",
    "T015": "实时语音助手为何考虑WebRTC，它与WebSocket的传输取舍是什么？",
    "T016": "统一模型网关怎样处理路由、限流和服务失败回退？",
    "T017": "工具目录很大，把所有Schema放进提示有什么问题，如何按需选工具？",
    "T018": "模型传错参数、工具执行超时，如何校验、重试并防止重复副作用？",
}

COMPOUND = [
    ("A001", "A002", "模型变成能执行任务的助手需要什么变化？另外说明它的模块组成。"),
    ("A005", "A006", "ReAct执行循环怎么工作？再比较先规划后执行与反思。"),
    ("A008", "A012", "助手要保留记忆时怎样设计存储？历史太长时又怎样压缩？"),
    ("A018", "T018", "对话服务重启如何恢复阶段？工具请求失败重试时如何避免重复执行？"),
    ("A019", "R018", "如何评估助手任务完成情况？另外怎样拆开衡量检索与回答质量？"),
    ("A021", "L014", "怎样通过轨迹定位助手的慢调用？另外KV缓存和提示缓存分别复用什么？"),
    ("R003", "L009", "频繁更新的知识应该检索还是微调？另外LoRA有哪些工程收益？"),
    ("R004", "R005", "资料切块时如何选长度与重叠？答案跨片段时如何保留完整上下文？"),
    ("R011", "R013", "关键词与向量召回各适合什么问题？两个结果列表如何融合？"),
    ("R017", "L018", "模型幻觉有哪些成因？有检索资料时怎样约束回答与拒答？"),
    ("T004", "T005", "MCP解决什么连接问题？Host、Client和Server职责是什么？"),
    ("T006", "T011", "MCP与函数调用有什么边界？再说明技能包怎样与它们配合。"),
    ("T014", "T015", "SSE与WebSocket适合什么通信？实时语音为什么还会考虑WebRTC？"),
    ("T016", "A024", "模型网关提供哪些治理能力？助手连接数据库又如何限制权限？"),
    ("L003", "L004", "注意力计算的显存如何优化？位置编码又解决什么不同问题？"),
    ("L010", "L011", "指令微调后的偏好优化有哪些路线？DPO与PPO主要差别是什么？"),
    ("L012", "L013", "生成时如何选解码策略？温度和截断采样参数如何影响随机性？"),
    ("L015", "L020", "部署模型时如何选择量化方案？CPU和GPU服务框架如何取舍？"),
    ("R019", "R021", "更新资料后如何同步检索索引？两份资料冲突时如何处理来源？"),
    ("A020", "A023", "怎样阻止助手循环调用工具？如何识别没有执行却宣称完成的情况？"),
]
ABSENT = [
    "资料中记录的Nebula-Interview内部服务2026年9月实际P99延迟是多少？",
    "给出代号IP-PRIVATE-73实验在未公开题库上的原始逐题召回结果。",
    "请提供不存在于题库的ZetaEmbed-2027的官方中文评测分数。",
    "这套平台维护者上周在私有服务器上部署了多少个GPU节点？",
    "请按知识库提供Orion-RAG内部采购合同的向量数据库成交价格。",
    "哪份资料记载了虚构数据集InterviewSecret-19的训练样本数量？",
    "请查询内部未发布的LambdaAgent团队下个月的招聘岗位名额。",
    "请给出XQ-UNPUBLISHED项目2026年8月生产故障的精确时间线。",
    "资料中有用户ID PRIVATE-8842最近一次模拟面试的个人得分吗？",
    "请根据已有题库列出私有模型Aster-Internal的未公开训练超参数。",
    "请提供私有仓库ZetaHost的实际工具调用失败率及统计日期。",
    "知识库中隐藏版本V9-NOT-RECORDED的重排阈值最终设置为多少？",
    "内部评审未公开的AtlasMemo记忆压缩实验到底节约了多少token？",
    "请给出未入库文档SecretCost-042中的本月API费用账单。",
]


def retrieval_dataset(questions):
    by_id = {q.id: q for q in questions}
    cases = []
    for number, (qid, query) in enumerate(QUERIES.items(), 1):
        q = by_id[qid]
        cases.append(
            {
                "id": f"V2-{number:03}",
                "question": query,
                "category": "paraphrase" if number % 3 else "terminology",
                "domains": [],
                "expected_sources": [f"interview-question://{qid}"],
                "answerable": True,
                "reference_answer": q.reference_answer,
            }
        )
    for a, b, query in COMPOUND:
        cases.append(
            {
                "id": f"V2-{len(cases) + 1:03}",
                "question": query,
                "category": "cross_domain" if by_id[a].domain != by_id[b].domain else "compound",
                "domains": [],
                "answerable": True,
                "expected_sources": [f"interview-question://{a}", f"interview-question://{b}"],
                "reference_points": [by_id[a].reference_answer, by_id[b].reference_answer],
            }
        )
    for query in ABSENT:
        cases.append(
            {
                "id": f"V2-{len(cases) + 1:03}",
                "question": query,
                "category": "unanswerable",
                "domains": [],
                "expected_sources": [],
                "answerable": False,
            }
        )
    # Connected expected evidence stays together, including compound queries.
    parent = {qid: qid for qid in QUERIES}

    def find(qid):
        while parent[qid] != qid:
            qid = parent[qid]
        return qid

    for a, b, _ in COMPOUND:
        parent[find(b)] = find(a)
    for case in cases:
        sources = case["expected_sources"]
        case["group"] = find(sources[0].split("//")[1]) if sources else case["id"]
        case["review"] = {
            "approved": False,
            "reviewer": "",
            "reviewed_at": "",
            "notes": "需核对问题可回答性、所有等价来源与参考要点。",
        }
    groups = sorted({c["group"] for c in cases})
    random.Random(20261012).shuffle(groups)
    counts = {group: sum(c["group"] == group for c in cases) for group in groups}
    test_groups = set()
    total = 0
    for group in groups:
        if total + counts[group] <= 40:
            test_groups.add(group)
            total += counts[group]
    for case in cases:
        case["split"] = "test" if case["group"] in test_groups else "dev"
    corpus_hash = hashlib.sha256(
        "\n".join(q.id + q.reference_answer for q in questions).encode()
    ).hexdigest()
    return {
        "name": "知识库检索分组种子集 v2 · 120题",
        "schema_version": 2,
        "label_status": "seed_requires_human_review",
        "source_answers_sha256": corpus_hash,
        "description": "86条独立改写、20条双主题查询、14条无资料查询。"
        "按标注来源连通组划分80开发/40测试，"
        "均不按领域过滤。参考答案来自本地题库，非人工金标准，尚未穷举等价来源。"
        "原20题回归集继续保留；本集主要检查改写、干扰、跨主题与拒答。",
        "cases": cases,
    }


def scoring_dataset(questions):
    by_id = {q.id: q for q in questions}
    definitions = [
        (
            "T001",
            "模型选择工具并生成参数；宿主校验参数和权限并执行工具；执行结果回填模型",
            [
                (
                    "correct",
                    ["模型选工具并给参数，宿主校验参数和权限后执行，再将结果回填模型。"],
                    ["mastered"] * 3,
                ),
                (
                    "partial",
                    ["模型会给出工具名和参数，程序会调用这个函数。"],
                    ["mastered", "partial", "not_addressed"],
                ),
                (
                    "incorrect",
                    ["模型在自己的参数中直接执行函数，宿主不需要检查权限，也不用把结果给模型。"],
                    ["incorrect"] * 3,
                ),
                (
                    "omission",
                    ["模型选择工具并生成参数。"],
                    ["mastered", "not_addressed", "not_addressed"],
                ),
                (
                    "equivalent",
                    [
                        "语言模型发出调用请求，应用层检查入参并鉴权后运行外部函数，返回值作为工具消息加入后续对话。"
                    ],
                    ["mastered"] * 3,
                ),
                (
                    "correction",
                    [
                        "是模型直接执行函数。",
                        "我纠正前面的话：模型只决定函数名与参数，宿主检查参数和权限再执行，并把返回值交给模型继续回答。",
                    ],
                    ["mastered"] * 3,
                ),
            ],
        ),
        (
            "R011",
            "关键词检索擅长精确词项匹配；向量检索擅长语义同义匹配；混合检索需要融合排序并去重",
            [
                (
                    "correct",
                    [
                        "关键词适合精确术语和编号，向量适合语义与同义问法，两路召回可以用RRF融合排名并去重。"
                    ],
                    ["mastered"] * 3,
                ),
                (
                    "partial",
                    ["关键词按词匹配，向量按语义匹配，把两路结果合在一起。"],
                    ["mastered", "mastered", "partial"],
                ),
                (
                    "incorrect",
                    [
                        "关键词检索不依赖词项而只理解语义，向量只能匹配相同字面，融合不用处理排序和重复项。"
                    ],
                    ["incorrect"] * 3,
                ),
                (
                    "omission",
                    ["向量可以找到同义表达，查询与文档被映射为向量再比较相似度。"],
                    ["not_addressed", "mastered", "not_addressed"],
                ),
                (
                    "equivalent",
                    [
                        "倒排召回抓住型号和术语，稠密表示找到不同措辞下的同一含义；两个有序列表按名次合并，同一资料只保留一份。"
                    ],
                    ["mastered"] * 3,
                ),
                (
                    "correction",
                    [
                        "向量只能匹配原词。",
                        "刚才说反了：倒排擅长精确词项，向量覆盖同义语义，结果用排序融合并去掉重复资料。",
                    ],
                    ["mastered"] * 3,
                ),
            ],
        ),
        (
            "R001",
            "资料清洗切块并建立向量索引；查询检索相关资料；将检索证据加入提示生成带来源回答",
            [
                (
                    "correct",
                    [
                        "清洗资料、切块并嵌入建索引，用户查询后召回相关内容，把证据放进提示让模型生成带来源的答案。"
                    ],
                    ["mastered"] * 3,
                ),
                (
                    "partial",
                    ["先把资料存起来，查询时找相关内容，再让模型回答。"],
                    ["partial", "mastered", "partial"],
                ),
                (
                    "incorrect",
                    [
                        "RAG就是直接把资料训练进模型参数，不需要查询检索任何资料，回答也不使用外部证据。"
                    ],
                    ["incorrect"] * 3,
                ),
                (
                    "omission",
                    ["先清洗资料，按语义切块，生成向量并建立索引。"],
                    ["mastered", "not_addressed", "not_addressed"],
                ),
                (
                    "equivalent",
                    [
                        "离线把文档整理成片段、编码为稠密表示存入库；在线用问句取回匹配片段，把正文与出处作为上下文交给生成器。"
                    ],
                    ["mastered"] * 3,
                ),
                (
                    "correction",
                    [
                        "RAG不做检索。",
                        "修正一下：先清洗切块生成向量索引，再检索与问题相关的资料，证据和出处进入提示用于生成有引用的回答。",
                    ],
                    ["mastered"] * 3,
                ),
            ],
        ),
        (
            "A018",
            "保存会话阶段和当前任务状态；持久化状态以支持中断恢复；使用请求去重或幂等避免重复执行",
            [
                (
                    "correct",
                    [
                        "记录当前阶段与任务进度，状态持久化后重启可恢复，使用请求ID去重和幂等操作避免重复执行。"
                    ],
                    ["mastered"] * 3,
                ),
                (
                    "partial",
                    ["保存当前处在哪一步，重启后继续执行。"],
                    ["mastered", "partial", "not_addressed"],
                ),
                (
                    "incorrect",
                    [
                        "完全不保存当前任务状态，所有信息只在内存里所以重启也不会丢，重试请求重复执行也不需要控制。"
                    ],
                    ["incorrect"] * 3,
                ),
                (
                    "omission",
                    ["给每个请求唯一编号，重复请求只处理一次。"],
                    ["not_addressed", "not_addressed", "mastered"],
                ),
                (
                    "equivalent",
                    [
                        "将流程节点及其进度写入数据库检查点，进程重建后从检查点续跑，以业务唯一键检查已经执行的步骤。"
                    ],
                    ["mastered"] * 3,
                ),
                (
                    "correction",
                    [
                        "服务重启只能重新开始。",
                        "更正：保存阶段与任务状态到持久存储，重启后读取恢复，并使用幂等键防止相同请求再次执行。",
                    ],
                    ["mastered"] * 3,
                ),
            ],
        ),
    ]
    cases = []
    for qid, points, variants in definitions:
        q = by_id[qid]
        for category, answers, statuses in variants:
            cases.append(
                {
                    "id": f"S-{len(cases) + 1:03}",
                    "question": q.text,
                    "reference_answer": q.reference_answer,
                    "scoring_points": points,
                    "category": category,
                    "answers": answers,
                    "expected_statuses": {f"P{i}": s for i, s in enumerate(statuses, 1)},
                    "review": {"approved": False, "reviewer": "", "reviewed_at": ""},
                }
            )
    return {
        "name": "逐点评分种子集 v1 · 24题",
        "label_status": "seed_requires_human_review",
        "description": "四个主题各六种回答：正确、部分、错误、遗漏、等价和补充纠错。"
        "预期标签由项目开发辅助生成，待人工复核；不能称为人工金标准。",
        "cases": cases,
    }


def main():
    questions = load_question_directory(ROOT / "data/interview_questions")
    destination = ROOT / "data/evaluation"
    for filename, data in (
        ("knowledge-v2.json", retrieval_dataset(questions)),
        ("scoring-v1.json", scoring_dataset(questions)),
    ):
        path = destination / filename
        if path.exists():
            raise SystemExit(f"保留现有标注，未覆盖：{path}")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{filename}: {len(data['cases'])} cases")


if __name__ == "__main__":
    main()
