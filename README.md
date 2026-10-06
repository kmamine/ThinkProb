# ThinkProb

A framework for profiling LLM cognitive behavior through CoT trace analysis. ThinkProb extracts cognitive profiles from language model reasoning traces, characterizing how different models approach open-ended problems.

## What is ThinkProb?

ThinkProb transforms raw Chain-of-Thought (CoT) reasoning traces into structured directed graphs, then computes a 22-dimensional cognitive profile vector that characterizes each model's thinking style.

**Key capabilities:**
- Collect reasoning traces from any OpenAI-compatible LLM endpoint (vLLM, OpenAI, etc.)
- Extract Thought Graphs with 12 node types and 8 edge types using a **non-generative pipeline** (regex + TextTiling + NLI — no parser LLM required)
- Compute 22 cognitive metrics across 5 categories (Breadth, Depth, Structure, Metacognitive, Efficiency)
- Generate radar chart visualizations for model comparison

## Installation

```bash
git clone https://github.com/your-org/ThinkProb.git
cd ThinkProb
pip install -e .
```

**Requirements:**
- Python 3.11+
- `pydantic` (data models)
- `networkx` (graph operations)
- `sentence-transformers` (embeddings for segmentation and metrics)
- `transformers` + `torch` (DeBERTa NLI cross-encoder)
- `scipy` (TextTiling peak detection)
- `scikit-learn` (agglomerative clustering)
- `matplotlib` + `numpy` (visualization)

## Quick Start

```bash
python scripts/ThinkProb_full.py \
    --questions data/questions/ethical_dilemmas.jsonl \
    --runs 3 \
    --output data
```

This will:
1. Collect traces from questions (K runs each) via the configured LLM endpoint
2. Extract Thought Graphs — non-generatively, using regex + MiniLM + DeBERTa NLI
3. Compute 22 cognitive metrics per trace and aggregate by model
4. Generate a markdown report and radar chart in `docs/`

---

# The ThinkProb Pipeline

## Pipeline Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Questions  │────▶│   Collect   │────▶│   Extract   │────▶│   Profile   │
│   (.jsonl)  │     │   Traces    │     │   Graphs    │     │   Metrics   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                    LLM endpoint        Non-generative        22 metrics
                    (vLLM/OpenAI)       regex+NLI only
```

---

## Phase 1: Data Collection

The **Collection Phase** sends questions to an LLM endpoint and captures raw Chain-of-Thought reasoning output.

### Input Format

```json
{"id": "D1_001", "domain": "ethical_dilemmas", "text": "A self-driving car's braking system..."}
{"id": "D1_002", "domain": "ethical_dilemmas", "text": "A doctor has one dose of medication..."}
```

### Configuration

| Parameter | Environment Variable | Default |
|-----------|---------------------|---------|
| API Key | `VLLM_API_KEY` | `sofia-token-j7y6qXDOTJ6grLvo` |
| Endpoint | `VLLM_ENDPOINT` | `http://10.17.1.57:8978` |
| Model | `VLLM_MODEL` | `Qwen/Qwen3.5-35B-A3B` |

### Output Format

```json
{
  "trace_id": "fff05748-34b0-4664-a483-9af3a7c6c4ea",
  "model": "Qwen/Qwen3.5-35B-A3B",
  "question_id": "D1_001",
  "domain": "ethical_dilemmas",
  "run": 1,
  "raw_cot": "This is a classic ethical dilemma...",
  "token_count": 2847,
  "collected_at": "2025-04-15T16:39:00Z"
}
```

---

## Phase 2: Graph Extraction (Non-generative)

The **Extraction Phase** transforms raw CoT text into structured Thought Graphs using three passes — no generative LLM involved.

```
raw_cot text ──▶ Pass 1: Regex ──▶ Pass 2: TextTiling ──▶ Pass 3: NLI ──▶ ThoughtGraph
                 hard boundaries   soft boundaries         edge promotion
```

### Pass 1: Hard Boundaries (Regex)

Cue-phrase patterns are matched at sentence starts to assign a `BoundaryClass` to each span:

| BoundaryClass | Cue Phrases | → NodeType | → EdgeType |
|---|---|---|---|
| BACKTRACK | "Wait,", "Actually,", "Let me reconsider" | CRT | BACK |
| BRANCH | "Alternatively,", "Another approach", "What if" | HYP | BRCH |
| META | "I'm going in circles", "Let me step back" | MET | SEQ |
| CONVERGENCE | "In conclusion", "Bringing this together" | SYN | SYNT |
| ELABORATION | "Specifically,", "For example,", "In practice" | SPC | ELAB |
| CONTRAST | "On the other hand", "However,", "But" | CMP | CONT |
| SUPPORT | "Because,", "This is supported by", "The reason is" | JUS | SUPP |

Spans shorter than 3 sentences or 50 tokens are merged with the following span. Paragraph breaks produce SEQ boundaries.

### Pass 2: Soft Boundaries (TextTiling with MiniLM)

On spans without a hard boundary, sentences are encoded with `all-MiniLM-L6-v2`. A sliding window (size=3) computes cosine similarity between adjacent windows. Local minima detected by `scipy.signal.find_peaks` (prominence ≥ 0.15) below the 30th percentile of within-trace similarities become soft boundaries (`BoundaryClass.NONE`, edge `SEQ`).

### Pass 3: NLI Edge Promotion (DeBERTa)

`cross-encoder/nli-deberta-v3-large` promotes sequential edges to typed semantic edges. For adjacent TU pairs within window=4, three hypotheses are tested (SUPP/CONT/ELAB) at threshold 0.75. For BACKTRACK spans > 4 positions apart, a BACK hypothesis is tested at threshold 0.70. All pairs are submitted as a **single batched** `predict()` call.

### Node Type Taxonomy (12 types, 4 families)

| Family | Type | Code | Definition |
|--------|------|------|------------|
| EXPLORATION | Hypothesis | HYP | Proposes a possible answer/approach |
| EXPLORATION | Reframing | RFR | Re-states problem from different angle |
| EXPLORATION | Analogy | ANA | Draws parallel to another domain |
| EXPLORATION | Brainstorm | BRS | Lists options without evaluation |
| ELABORATION | Justification | JUS | Provides evidence/reasoning |
| ELABORATION | Specification | SPC | Makes idea more concrete |
| ELABORATION | Implication | IMP | Derives consequence |
| ELABORATION | Constraint | CON | Identifies limits/caveats |
| EVALUATION | Critique | CRT | Points out weaknesses |
| EVALUATION | Comparison | CMP | Weighs alternatives |
| EVALUATION | Meta-reflection | MET | Reflects on reasoning process |
| CONVERGENCE | Synthesis | SYN | Combines insights from multiple threads |

### Edge Type Taxonomy

| Type | Code | Definition |
|------|------|------------|
| Sequential | SEQ | Default adjacency (no semantic relationship) |
| Elaboration | ELAB | Child deepens/specifies parent |
| Branching | BRCH | Child explores new direction |
| Backtracking | BACK | Explicitly revises earlier idea |
| Synthesis | SYNT | Combines ideas from multiple sources |
| Critique | CRIT | Evaluates/challenges target |
| Support | SUPP | Provides evidence for target |
| Contrast | CONT | Opposing perspectives |

---

## Phase 3: Profile Computation

### The 22 Cognitive Metrics

#### Breadth Metrics (4)

| Metric | Formula | Intuition |
|--------|---------|-----------|
| `branching_factor` | `\|E_BRCH\| / \|V\|` | How many distinct directions explored? |
| `unique_perspective_count` | Count of RFR nodes | Independent viewpoints |
| `domain_spread` | Agglomerative clusters of BRS+HYP embeddings (cosine threshold=0.45) | Topic coverage |
| `first_idea_diversity` | Cosine distance between first 3 HYP node embeddings | How different are initial ideas? |

#### Depth Metrics (4)

| Metric | Formula | Intuition |
|--------|---------|-----------|
| `max_elaboration_chain` | Longest path following ELAB edges only | Deepest single-thread reasoning |
| `mean_branch_depth` | Average depth across branches (semantic subgraph) | Typical elaboration depth |
| `specificity_gradient` | Linear regression slope of entity density vs. depth | Do ideas become more concrete? |
| `reasoning_density` | `\|nodes with ≥1 semantic edge\| / \|V\|` | Fraction of logically connected thinking |

#### Structure Metrics (6)

| Metric | Formula | Intuition |
|--------|---------|-----------|
| `exploration_exploitation_ratio` | EXPLORATION nodes / ELABORATION nodes | Breadth-first vs depth-first? |
| `backtracking_rate` | `\|E_BACK\| / max(\|E_sem\|, 1)` | How often does it revise itself? |
| `cross_branch_connectivity` | Fraction of cross-branch pairs with SYNT/SUPP edge | Does it connect different threads? |
| `convergence_index` | `sum(d_in(v) for SYN nodes) / (\|V\| × mean_d_in)` | Convergence pattern |
| `graph_density` | `\|E_sem\| / (\|V\|×(\|V\|-1))` (semantic edges only) | Overall semantic connectivity |
| `revision_depth` | `mean(\|pos(u) - pos(v)\|) for BACK edges` | How far back does it revise? |

#### Metacognitive Metrics (4)

| Metric | Formula | Intuition |
|--------|---------|-----------|
| `self_reflection_rate` | MET nodes / total | How often monitors own reasoning? |
| `critique_to_hypothesis_ratio` | CRT nodes / HYP nodes | Evaluate as much as generate? |
| `hedging_density` | TUs with uncertainty markers / total | Epistemic humility level |
| `perspective_taking` | RFR nodes / total | How often shifts viewpoint? |

#### Efficiency Metrics (2)

| Metric | Formula | Intuition |
|--------|---------|-----------|
| `token_per_idea` | `avg_tokens / max(unique_perspective_count, 1)` | Verbosity relative to ideas |
| `redundancy_ratio` | TU pairs with cosine similarity > 0.90 | How repetitive? |

### Aggregation

Individual trace profiles are grouped by model and averaged across all numeric metrics.

**Output**: `data/profiles/results_<timestamp>.json`

---

## Phase 4: Visualization & Report

### Radar Chart

- Normalize all 22 metrics to [0,1] using empirical bounds
- Plot on polar chart for visual comparison
- Save as `docs/benchmark_radar_<timestamp>.png`

### Markdown Report

- Summary statistics and all 22 metrics with values
- Save as `docs/benchmark_report.md`

---

## Usage Examples

### Full Pipeline (Recommended)

```bash
python scripts/ThinkProb_full.py \
    --questions data/questions/ethical_dilemmas.jsonl \
    --runs 3 \
    --output data
```

### Individual Phases

```bash
# Phase 1: Collect traces only
python scripts/run_experiment.py \
    --questions data/questions/ethical_dilemmas.jsonl \
    --runs 3 --collect

# Phase 2: Extract graphs (non-generative — no LLM endpoint required)
python scripts/extract_graphs.py \
    --input data/traces/ --output data/graphs/

# Phase 3: Compute profiles
python scripts/run_experiment.py --compute
```

### Custom Configuration

```bash
export VLLM_ENDPOINT=http://localhost:8080
export VLLM_MODEL=deepseek-r1
export VLLM_API_KEY=your-key-here

python scripts/ThinkProb_full.py \
    --questions data/questions/your_questions.jsonl \
    --runs 5
```

---

## Directory Structure

```
ThinkProb/
├── src/ThinkProb/
│   ├── collect/               # CoT collection (LLM calls here only)
│   │   ├── models.py
│   │   └── collector.py
│   ├── extract/               # Non-generative graph extraction
│   │   ├── schemas.py         # Pydantic data models (BoundaryClass, ThoughtGraph, etc.)
│   │   ├── segmenter.py       # 3-pass: regex + TextTiling + NLI
│   │   ├── classifier.py      # boundary_class → node_type (rule-based)
│   │   └── linker.py          # Graph assembly
│   ├── metrics/               # 22 metric computations
│   │   ├── breadth.py
│   │   ├── depth.py
│   │   ├── structure.py
│   │   ├── metacognitive.py
│   │   ├── efficiency.py
│   │   └── profile.py
│   └── utils/
│       └── models.py          # Shared lazy singletons (embed + NLI models)
├── scripts/
│   ├── ThinkProb_full.py     # Standalone full pipeline
│   ├── run_experiment.py      # Modular experiment runner
│   ├── extract_graphs.py      # Batch extraction
│   └── generate_report.py     # Visualization + report
├── data/
│   ├── questions/
│   ├── traces/                # Raw CoT traces (generated)
│   ├── graphs/                # Extracted graphs (generated)
│   └── profiles/              # Computed profiles (generated)
└── docs/
    ├── benchmark_report.md
    └── benchmark_radar_*.png
```

---

## Troubleshooting

### No graphs extracted

- Check that traces have `raw_cot` field with sufficient text
- Run `python scripts/extract_graphs.py --input data/traces/ --output data/graphs/` directly to see per-file output

### NLI model slow on first run

DeBERTa downloads on first use (~1.5 GB). Subsequent runs use the cached model. To skip NLI (faster, regex+TextTiling only):
```bash
python scripts/extract_graphs.py --no-nli
```

### Metrics returning 0

- `first_idea_diversity`, `redundancy_ratio`, `domain_spread`: require sentence-transformers (included by default)
- `specificity_gradient`: uses linear regression on entity density — returns 0 if no depth variation

### Radar chart looks skewed

Normalization bounds may not match your data distribution. Adjust bounds in `scripts/generate_report.py`.

---

## License

Apache 2.0

---

## Citation

```
@misc{kerkouri2026thinkprobeaccuracystructural,
      title={ThinkProbe: Beyond Accuracy -- Structural Profiling of Open-Ended LLM Reasoning Traces via Non-Generative Thought Graphs}, 
      author={Mohamed Amine Kerkouri and Simon D. Hernandez and Marouane Tliba and Yann Dauxais and Maha Ben-Fares and Pierre Holat},
      year={2026},
      eprint={2606.29067},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.29067}, 
}
```
