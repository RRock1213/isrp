# ISRP: 面向长文生成的迭代自优化规划器

ISRP（Iterative Self-Refining Planner）是一种基于树结构的大纲规划与优化框架，其核心思想是将内容构建与篇幅分配解耦，通过**生成-评估-优化**（Generate-Evaluate-Refine）迭代循环，结合深度优先的节点扩展、两级验证机制及抗遗忘注入策略，产出结构连贯、满足字数约束的长文本。

## 整体架构

```
                           ┌──────────────────────┐
                           │   阶段一：预处理           │
                           │  查询压缩                  │
                           │  风格识别                  │
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼───────────┐
                           │   阶段二：大纲树生成      │
                           │  根节点建立                │
                           │  DFS 节点扩展              │
                           │  章节级检查点验证          │
                           │  全局验证                  │
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼───────────┐
                           │   阶段三：大纲树优化      │
                           │  抗遗忘注入                │
                           │  BFS 字数分配             │
                           └──────────────────────┘
```

规划器首先通过查询压缩和风格识别提炼用户意图。随后，以深度优先搜索（DFS）构建树形大纲，交替进行纵向扩展（子树生成）与横向扩展（覆盖度填补），并引入两级验证机制——章节级检查点（Checkpoint）在完成一个一级章节后即时触发纠错，全局验证在大纲生成完毕后进行规则与语义层面的系统性校验。规划完成后，抗遗忘注入在每次 LLM 调用前强制注入全局记忆与局部上下文，避免长链式迭代中的意图漂移；BFS 按层分配字数，依据节点重要性权重分配篇幅预算。

## 目录结构

```
isrp-submit/
├── isrp/                         # ISRP 核心框架
│   ├── __init__.py               # 包入口，暴露 SyncISRPPlanner 等
│   ├── schemas.py                # 数据结构 (OutlineNode, PlanSession 等)
│   ├── engines.py                # 引擎配置、温度调度、参数常量
│   ├── prompts.py                # 各步骤 Prompt 模板
│   ├── planner.py                # 高层规划编排器
│   ├── evaluators/               # 评估器
│   │   ├── base.py               # 评估基类
│   │   ├── isrp_evaluator.py     # ISRP 内部评估器
│   │   ├── writingbench_evaluator.py  # WritingBench 外部评估器
│   │   └── factory.py            # 评估器工厂
│   ├── steps/                    # 管线步骤
│   │   ├── query_style_recognition.py   # 查询风格识别
│   │   ├── root_node_establishment.py   # 根节点建立
│   │   ├── dfs_expansion_engine.py      # DFS 节点扩展引擎
│   │   ├── global_abstract.py           # 全局摘要生成
│   │   ├── chapter_level_validation.py  # 章节级检查点验证
│   │   ├── global_validation.py         # 全局验证
│   │   ├── description_optimize.py      # 节点描述优化
│   │   ├── word_count_allocation.py     # BFS 字数分配
│   │   └── format.py                    # 格式规范化
│   └── utils/                    # 工具函数
│       ├── llm_client.py         # 结构化 LLM 调用
│       ├── node_ops.py           # 节点级操作
│       ├── tree_ops.py           # 树级操作
│       ├── prompt_compressor.py  # 查询压缩
│       └── hash_utils.py         # 样本 ID 生成
├── isrp_ablation/                 # 消融实验分支（fork 自 isrp/）
│   ├── ablation_config.py        # 消融开关配置
│   ├── planner.py                # 含消融钩子的规划器
│   ├── engines.py                # 含消融钩子的引擎
│   ├── steps/                    # 步骤处理器（同 isrp/steps/）
│   ├── evaluators/               # 评估器（同 isrp/evaluators/）
│   └── utils/                    # 工具（同 isrp/utils/）
├── e2e/                          # E2E (AgentWrite) 基线
│   ├── planner.py                # 端到端大纲生成
│   └── prompts/plan.txt          # Prompt 模板
├── evaluation/                   # 评估脚本
│   ├── eval_outline.py           # 大纲质量评估 (6 维度)
│   ├── eval_body.py              # 正文质量评估 (动态 checklist)
│   └── pairwise/                 # 成对偏好评估
├── rag_module/                   # RAG 模块 (可选)
│   ├── chunker.py                # 文本切片
│   ├── embedder.py               # 向量嵌入
│   ├── retriever.py              # 检索器
│   ├── vector_store.py           # 向量存储 (zvec)
│   ├── cache_manager.py          # 缓存管理
│   └── rag_service.py            # RAG 服务入口
├── api_client.py                 # 统一多平台 API 客户端
├── api_strategies.py             # API 策略抽象
├── config.py                     # 全局配置 (模型/路径/参数)
├── checkpoint_manager.py         # 断点续传支持
├── run_plan.py                   # 入口：大纲生成
├── run_write.py                  # 入口：正文生成
├── run_ablation_test.py          # 入口：消融实验
├── scripts/                      # 分析工具
├── outputs/                      # 实验结果
├── requirements.txt
└── .env.example
```

## 环境配置

```bash
pip install -r requirements.txt
```

将 `.env.example` 复制为 `.env`，填入 API Key：

```env
OPENAI_API_KEY=
DASHSCOPE_API_KEY=sk-xxxxxxxx
```

模型选择在 `config.py` 中配置。默认使用 **Qwen3-Max** 作为规划模型，**GLM-5** 作为评估模型，均通过阿里云百炼（DashScope）API 调用。

## 使用方法

### 大纲生成

```bash
# ISRP 规划器（默认）
python run_plan.py

# E2E 基线
python e2e/planner.py

# 自定义参数
python run_plan.py \
    --input data/WritingBench-120.jsonl \
    --output outputs/plan.jsonl \
    --model qwen3-max
```

### 正文生成

```bash
python run_write.py \
    --plan outputs/plan.jsonl \
    --output outputs/write.jsonl
```

### 评估

```bash
# 大纲质量评估（6 个静态维度）
python evaluation/eval_outline.py --input outputs/plan.jsonl

# 正文质量评估（每条样本动态 checklist）
python evaluation/eval_body.py --input outputs/write.jsonl

# 成对偏好评估
python evaluation/pairwise/eval_pairwise_outline.py
python evaluation/pairwise/eval_pairwise_write.py
```

### 分析工具

```bash
# 生成统计文件
python scripts/gen_eval_outline_stats.py <eval_outline.jsonl>
python scripts/gen_eval_quality_stats.py <eval_quality.jsonl>

```

### 消融实验

复现论文 Table 3 的消融研究：

```bash
# 禁用迭代评估 (w/o Iteration Evaluation)
python run_ablation_test.py --ablation iteration_eval --limit 10

# 禁用全局记忆注入 (w/o Global Memory Injection)
python run_ablation_test.py --ablation global_memory --limit 10

# 禁用局部上下文注入 (w/o Local Context Injection)
python run_ablation_test.py --ablation local_context --limit 10

# 禁用BFS字数分配 (w/o Word Count Allocation)
python run_ablation_test.py --ablation word_allocation --limit 10

# 完整对照组 (Full ISRP)
python run_ablation_test.py --ablation full --limit 10
```

## 主要实验结果

**正文质量**（WritingBench 基准，120 条样本，六个领域）：

| 方法 | 正文均分 | 高分率≥8 | 金融 | 政法 | 文学 | 广告 | 教育 | 工程 |
|------|---------|---------|------|------|------|------|------|------|
| LongWriter | 6.53 | 30.8% | 7.04 | 6.72 | 6.75 | 5.13 | 7.32 | 6.92 |
| WritingModel | 7.72 | 56.2% | 8.84 | 7.27 | 7.87 | 6.45 | 8.77 | 8.66 |
| SuperWriter | 8.10 | 69.6% | 8.62 | 7.90 | 8.07 | 7.52 | 8.72 | 8.54 |
| E2E (AgentWrite) | 8.78 | 89.2% | 8.98 | 8.68 | 8.93 | 8.72 | 9.11 | 8.48 |
| **ISRP (Ours)** | **9.33** | **95.8%** | **9.74** | **9.40** | **9.36** | **9.09** | **9.58** | **9.11** |

**大纲质量**（六个维度对比）：

| 维度 | E2E | ISRP | Δ |
|------|-----|------|-----|
| 相关性 Relevance | 9.16 | 9.75 | +0.59 |
| 具体性 Specificity | 8.50 | 9.17 | +0.67 |
| 指导性 Guidance | 8.75 | 9.41 | +0.66 |
| 语言 Language | 9.08 | 9.67 | +0.59 |
| 覆盖度 Coverage | 9.18 | 9.50 | +0.32 |
| 语义正确性 Semantics | 8.72 | 9.17 | +0.45 |
| **大纲均分** | 8.90 | **9.44** | +0.54 |
| 高分率≥8 | 92.5% | **98.3%** | +5.8% |

## 实验数据组织

```
outputs/
├── main_experiment/
│   ├── e2e/              # E2E 基线结果
│   ├── isrp/             # ISRP 完整结果
│   └── baselines/        # LongWriter, SuperWriter, WritingModel
├── ablation_experiments/
│   ├── no_iteration_evaluation/
│   ├── no_global_memory_injection/
│   ├── no_local_memory_injection/
│   └── no_word_count_allocation/
├── domain_experiments/
│   └── Impact of base LLMs/         # 跨基座模型对比 (deepseek_v4_pro, kimi-k2.5, qwen3_max)
└── domain_experiments/
    └── Sensitivity of tree widths/  # 树宽度 ∈ {3,4,5,6}
```

## 核心设计决策

- **解耦规划**：大纲质量与字数分配独立求解。大纲阶段专注内容结构与逻辑，字数在后处理阶段由 BFS 逐层权重分配完成，避免早期固定字数对结构灵活性的约束。
- **DFS + 检查点**：深度优先的节点扩展策略使得在完成一个一级章节子树时即可触发章节级验证（Checkpoint），在缺陷向兄弟节点级联传播前实现即时遏制。
- **抗遗忘注入**：每次 LLM 调用前，上下文被强制增强为 (a) 压缩后的全局记忆（主题、约束、风格）与 (b) 动态检索到的局部文本块。这有效抑制了长链式迭代中的意图漂移与信息遗漏。
- **LLM-as-a-Judge 评估**：大纲与正文均使用独立评估模型（GLM-5）进行打分，每条样本使用实例化 rubric，避免了自评偏差。评估维度与内部优化信号维度相互独立。
