# ISRP: Iterative Self-Refining Planner for Long-Text Generation

ISRP is a tree-based outline planning and optimization framework that decouples content construction from length allocation. It employs an iterative **Generate-Evaluate-Refine** loop with DFS-based node expansion, a two-tiered validation mechanism, and an anti-forgetting injection strategy to produce coherent, constraint-satisfying long-form text.

## Architecture

```
                           ┌──────────────────────┐
                           │   Phase I: Preprocessing   │
                           │  Query Compression         │
                           │  Style Recognition         │
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼───────────┐
                           │ Phase II: Outline Tree  │
                           │  Root Establishment      │
                           │  DFS Expansion           │
                           │  Chapter-Level Checkpoint│
                           │  Global Validation       │
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼───────────┐
                           │ Phase III: Optimization │
                           │  Anti-Forgetting Injection│
                           │  BFS Word-Count Allocation│
                           └──────────────────────┘
```

The planner first distills user intent through query compression and style recognition. It then constructs a tree-structured outline via depth-first search, interleaving vertical expansion (child generation), horizontal expansion (coverage gap filling), and a two-tiered validation mechanism (chapter-level checkpoint + global validation). Finally, anti-forgetting injection preserves long-range consistency while BFS-based word-count allocation assigns length budgets proportional to node importance.

## Directory Structure

```
isrp-submit/
├── isrp/                         # Core ISRP framework
│   ├── __init__.py               # Package entry
│   ├── schemas.py                # Data models (OutlineNode, PlanSession, etc.)
│   ├── engines.py                # Engine config & temperature schedules
│   ├── prompts.py                # Prompt templates for each step
│   ├── planner.py                # High-level planner orchestrator
│   ├── evaluators/               # Outline & WritingBench evaluators
│   ├── steps/                    # Pipeline steps
│   │   ├── query_style_recognition.py
│   │   ├── root_node_establishment.py
│   │   ├── dfs_expansion_engine.py
│   │   ├── global_abstract.py
│   │   ├── chapter_level_validation.py
│   │   ├── global_validation.py
│   │   ├── description_optimize.py
│   │   ├── word_count_allocation.py
│   │   └── format.py
│   └── utils/                    # Utilities
│       ├── llm_client.py         # Structured LLM invocation
│       ├── node_ops.py           # Node-level operations
│       ├── tree_ops.py           # Tree-level operations
│       ├── prompt_compressor.py  # Query compression
│       └── hash_utils.py         # Sample ID generation
├── isrp_ablation/                 # Ablation experiment fork of ISRP
│   ├── ablation_config.py        # Ablation feature gates
│   ├── planner.py                # Planner with ablation hooks
│   ├── engines.py                # Engines with ablation hooks
│   ├── steps/                    # Step processors (same as isrp/steps/)
│   ├── evaluators/               # Evaluators (from isrp/evaluators)
│   └── utils/                    # Utilities (from isrp/utils)
├── e2e/                          # E2E (AgentWrite) baseline
│   ├── planner.py
│   └── prompts/plan.txt
├── evaluation/                   # Evaluation scripts
│   ├── eval_outline.py           # Outline quality evaluation
│   ├── eval_body.py              # Body text quality evaluation
│   └── pairwise/                 # Pairwise preference evaluation
├── rag_module/                   # Optional RAG module
├── api_client.py                 # Unified multi-provider API client
├── api_strategies.py             # API strategy abstractions
├── config.py                     # Global configuration
├── checkpoint_manager.py         # Resume-from-checkpoint support
├── run_plan.py                   # Entry: outline generation
├── run_write.py                  # Entry: body text generation
├── run_ablation_test.py          # Entry: ablation experiments
├── scripts/                      # Analysis utilities
├── outputs/                      # Experiment results
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```env
OPENAI_API_KEY=
DASHSCOPE_API_KEY=sk-xxxxxxxx
```

Model selection is configured in `config.py`. The default uses **Qwen3-Max** as the planner and **GLM-5** as the evaluator, both served via the DashScope (Alibaba Cloud Bailian) API.

## Usage

### Outline Generation

```bash
# ISRP planner (default)
python run_plan.py

# E2E baseline
python e2e/planner.py

# With custom parameters
python run_plan.py --input data/WritingBench-120.jsonl --output outputs/plan.jsonl --model qwen3-max
```

### Body Text Generation

```bash
python run_write.py --plan outputs/plan.jsonl --output outputs/write.jsonl
```

### Evaluation

```bash
# Outline quality (6 static dimensions)
python evaluation/eval_outline.py --input outputs/plan.jsonl

# Body text quality (dynamic checklist per sample)
python evaluation/eval_body.py --input outputs/write.jsonl

# Pairwise preference evaluation
python evaluation/pairwise/eval_pairwise_outline.py
python evaluation/pairwise/eval_pairwise_write.py
```

### Analysis Scripts

```bash
# Generate statistics for eval files
python scripts/gen_eval_outline_stats.py <eval_outline.jsonl>
python scripts/gen_eval_quality_stats.py <eval_quality.jsonl>

```

### Ablation Experiments

Reproduce the ablation study (Table 3 in paper):

```bash
# w/o Iteration Evaluation
python run_ablation_test.py --ablation iteration_eval --limit 10

# w/o Global Memory Injection
python run_ablation_test.py --ablation global_memory --limit 10

# w/o Local Context Injection
python run_ablation_test.py --ablation local_context --limit 10

# w/o Word Count Allocation
python run_ablation_test.py --ablation word_allocation --limit 10

# Full ISRP (control)
python run_ablation_test.py --ablation full --limit 10
```

## Main Results

**Body text quality** across six domains (WritingBench benchmark, 120 samples):

| Method | Body Avg | H(B)≥8 | Finance | Politics | Literature | Ad&Mar | Education | Engineering |
|--------|----------|--------|---------|----------|------------|--------|-----------|-------------|
| LongWriter | 6.53 | 30.8% | 7.04 | 6.72 | 6.75 | 5.13 | 7.32 | 6.92 |
| WritingModel | 7.72 | 56.2% | 8.84 | 7.27 | 7.87 | 6.45 | 8.77 | 8.66 |
| SuperWriter | 8.10 | 69.6% | 8.62 | 7.90 | 8.07 | 7.52 | 8.72 | 8.54 |
| E2E (AgentWrite) | 8.78 | 89.2% | 8.98 | 8.68 | 8.93 | 8.72 | 9.11 | 8.48 |
| **ISRP (Ours)** | **9.33** | **95.8%** | **9.74** | **9.40** | **9.36** | **9.09** | **9.58** | **9.11** |

**Outline quality** (6-dimension evaluation):

| Dimension | E2E | ISRP | Δ |
|-----------|-----|------|-----|
| Relevance | 9.16 | 9.75 | +0.59 |
| Specificity | 8.50 | 9.17 | +0.67 |
| Guidance | 8.75 | 9.41 | +0.66 |
| Language | 9.08 | 9.67 | +0.59 |
| Coverage | 9.18 | 9.50 | +0.32 |
| Semantics | 8.72 | 9.17 | +0.45 |
| **Outline Avg** | 8.90 | **9.44** | +0.54 |
| H(O)≥8 | 92.5% | **98.3%** | +5.8% |

## Experiment Organization

```
outputs/
├── main_experiment/
│   ├── e2e/              # E2E baseline results
│   ├── isrp/             # ISRP full results
│   └── baselines/        # LongWriter, SuperWriter, WritingModel
├── ablation_experiments/
│   ├── no_iteration_evaluation/
│   ├── no_global_memory_injection/
│   ├── no_local_memory_injection/
│   └── no_word_count_allocation/
├── domain_experiments/
│   └── Impact of base LLMs/    # Cross-LLM comparison (deepseek_v4_pro, kimi-k2.5, qwen3_max)
└── domain_experiments/
    └── Sensitivity of tree widths/  # Width ∈ {3,4,5,6}
```

## Key Design Decisions

- **Decoupled planning**: outline quality and word-count allocation are solved independently. The outline stage focuses purely on content structure; word counts are assigned post-hoc by a BFS layer-by-layer weighting mechanism.
- **DFS with checkpoint**: chapter-level validation triggers immediately when a Level-1 subtree completes, enabling early error containment before defects cascade to sibling branches.
- **Anti-forgetting injection**: before every LLM call, the context is augmented with (a) compressed global memory (topic, requirements, style) and (b) dynamically retrieved local chunks. This prevents topic drift in long iterative chains.
- **LLM-as-a-Judge evaluation**: both outline and body text are scored by a separate evaluator model (GLM-5) with instance-specific rubrics, avoiding the self-enhancement bias common in self-evaluation.
