# Phase 0 — State-of-the-Art Research Report
## RAG Retrieval Methods Benchmark on T²-RAGBench

**Date**: 2026-03-29
**Status**: Draft for review

---

## Table of Contents
1. [T²-RAGBench Deep-Dive](#1-t2-ragbench-deep-dive)
2. [Retrieval Methods Survey](#2-retrieval-methods-survey)
3. [Embedding Model Comparison](#3-embedding-model-comparison)
4. [Reranker Comparison](#4-reranker-comparison)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Method Matrix](#6-method-matrix)
7. [Experiment Plan](#7-experiment-plan)
8. [Paper Positioning & Novelty](#8-paper-positioning--novelty)
9. [Recommended Venue](#9-recommended-venue)
10. [Open Questions for Discussion](#10-open-questions-for-discussion)

---

## 1. T²-RAGBench Deep-Dive

### 1.1 Paper Overview
- **Title**: T²-RAGBench: Text-and-Table Benchmark for Evaluating Retrieval-Augmented Generation
- **Authors**: Jan Strich, Enes Kutay Isgorur, Maximilian Trescher, Chris Biemann, Martin Semmann
- **Venue**: EACL 2026 (Main Conference)
- **arXiv**: 2506.12071 (v2: Jan 16, 2026)
- **License**: CC-BY-4.0
- **GitHub**: https://github.com/uhh-hcds/g4kmu-paper
- **HuggingFace**: https://huggingface.co/datasets/G4KMU/t2-ragbench
- **Downloads**: 12k+ on HuggingFace

### 1.2 Dataset Statistics

| Subset | Domain | Documents | QA Pairs | Avg Tokens/Doc | Avg Q Tokens (Reformulated) |
|--------|--------|-----------|----------|-----------------|---------------------------|
| FinQA | Finance | 2,789 | 8,281 | 950.4 | 39.2 |
| ConvFinQA | Finance | 1,806 | 3,458 | 890.9 | 30.9 |
| TAT-DQA | Finance | 2,723 | 11,349 | 915.3 | 31.7 |
| ~~VQAonBD~~ | ~~Finance~~ | ~~1,777~~ | ~~9,820~~ | ~~460.3~~ | ~~43.5~~ |
| **Total (active)** | | **7,318** | **23,088** | **~920** | **~34** |

> **Important**: VQAonBD was removed from the dataset due to low quality of question reformulations. The active dataset has 3 subsets.

### 1.3 Data Format (HuggingFace Parquet)

Key columns:
- `id`, `context_id` — identifiers
- `question` — reformulated context-independent question
- `program_answer`, `original_answer` — numerical ground truth
- `context` — full document in **markdown format** (text + tables)
- `table` — table portion only (markdown)
- `pre_text`, `post_text` — text before/after table
- `file_name` — source PDF reference
- Company metadata: `company_name`, `company_symbol`, `company_sector`, `company_industry`, `report_year`

**Splits available**: FinQA and TAT-DQA have train/dev/test. ConvFinQA has `turn_0` only.

### 1.4 Key Characteristics
- **All answers are numerical** (yes/no → 0/1 conversion)
- **Financial domain only** — SEC filings, earnings reports (2000-2019)
- **Context-independent questions**: 91.3% validated by human experts (reformulated from original context-dependent questions)
- **Documents contain both text AND tables** — this is the core challenge
- **Inter-annotator agreement**: Cohen's κ = 0.58 (moderate-substantial)

### 1.5 Methods Tested in Original Paper

| Method | NM (Weighted Avg) | MRR@3 | Notes |
|--------|-------------------|-------|-------|
| Pretrained-Only | 3.9% | — | No retrieval, parametric knowledge only |
| Oracle Context | 72.3% | 100.0 | Upper bound — gold context given |
| Base-RAG | 37.2% | 36.9 | Dense retrieval (E5-Large) |
| **Hybrid BM25** | **41.3%** | **37.8** | **Best method** — BM25 + Dense |
| Reranker | 31.8% | 30.3 | Cross-encoder reranker |
| HyDE | 34.0% | 32.0 | Hypothetical document — worse than Base-RAG |
| Summarization | 18.8% | 36.5 | Summarize then answer |
| SumContext | 37.4% | 36.5 | Summarize + context |

**Embedding models tested (Base-RAG, k=5)**:
| Model | R@1 | R@5 | MRR@5 |
|-------|-----|-----|-------|
| Stella-EN-1.5B | 2.7% | 6.5% | 4.0% |
| GTE-Qwen2 1.5B | 14.5% | 23.2% | 14.5% |
| Multilingual E5-Instruct | 29.4% | 53.3% | 38.6% |
| Gemini Text-Embedding-004 | 32.5% | 52.8% | 41.4% |
| OpenAI text-embedding-3-large | 33.8% | 56.1% | 43.6% |

**LLMs**: Llama 3.3-70B, QwQ-32B (no significant difference, <0.3%)

**Hardware**: 2× NVIDIA H100

### 1.6 Critical Gaps in the Original Paper
1. **Only 6 retrieval methods tested** — we can add 10+ more
2. **No ColBERT / late interaction** retrieval
3. **No Contextual Retrieval** (Anthropic's approach)
4. **No Late Chunking** (Jina's approach)
5. **No GraphRAG** or KG-enhanced retrieval
6. **No Multi-Query / RAG-Fusion**
7. **No RAPTOR** tree-based retrieval
8. **No Parent-Child** hierarchical retrieval
9. **No HyPE** (index-time synthetic queries)
10. **No Agentic/Adaptive RAG** (Self-RAG, CRAG)
11. **Limited embedding model comparison** (only 5 models)
12. **No reranker comparison** — only one unnamed reranker
13. **No chunking ablation** — they use whole documents (~920 tokens avg)
14. **No statistical significance tests**
15. **Limited metrics** — only NM and MRR@3; no Recall@k, nDCG, MAP, or generation-quality metrics
16. **No SPLADE** or learned sparse retrieval
17. **No fusion method comparison** (they only tried one hybrid approach)

---

## 2. Retrieval Methods Survey

### 2.1 Sparse Retrieval

#### BM25
- **Paper**: Robertson et al. (1994, 2009) — Okapi BM25
- **Implementation**: Pyserini (Lucene-based), rank_bm25 (Python), Elasticsearch
- **Parameters**: k1 (term frequency saturation, default 1.2), b (length normalization, default 0.75)
- **Strengths**: Fast, no training needed, strong baseline for exact keyword match, works well for technical/financial terminology
- **Weaknesses**: No semantic understanding, fails on paraphrases, vocabulary mismatch
- **T²-RAGBench relevance**: Financial documents have specific terminology — BM25 should work well for exact numerical references

#### SPLADE / SPLADE++
- **Paper**: Formal et al. (SIGIR 2021, 2022); SPLADE-v3 (Lassance et al., 2024)
- **Mechanism**: Learned sparse representations — uses MLM head to predict term expansion weights. Produces sparse vectors with semantic expansion.
- **Latest**: Echo-Mistral-SPLADE (Doshi et al., 2024) — decoder-only backbone, surpasses all prior SPLADE variants on BEIR
- **Implementation**: naver/splade (GitHub), Qdrant native support
- **Strengths**: Semantic expansion beyond exact match while maintaining sparsity; matches BM25 latency with superior quality
- **Weaknesses**: Requires training, larger index than BM25
- **T²-RAGBench relevance**: Could help with financial terminology expansion

### 2.2 Dense Retrieval (Bi-Encoder)

See Section 3 for detailed embedding model comparison.

**General approach**: Encode query and document separately → cosine similarity / dot product search via ANN (FAISS, Qdrant, etc.)

**Strengths**: Semantic understanding, handles paraphrases, works across languages
**Weaknesses**: Embedding quality varies by domain; may miss exact keyword matches; requires GPU for fast encoding

### 2.3 Late Interaction (ColBERT)

#### ColBERTv2
- **Paper**: Santhanam et al. (NAACL 2022)
- **Mechanism**: Token-level embeddings for both query and document; MaxSim operation computes relevance at token granularity
- **PLAID Engine**: Performance-optimized Late Interaction Driver — 7× GPU speedup, 45× CPU speedup via centroid interaction
- **Implementation**: colbert-ai (official), RAGatouille (user-friendly wrapper by AnswerDotAI)
- **Strengths**: Fine-grained token-level matching; better than bi-encoders for nuanced queries; can serve as both retriever and reranker
- **Weaknesses**: 6-10× larger index than dense; higher latency than bi-encoder; complex setup
- **Variants**: Jina-ColBERT-v2, ColBERT-XM (multilingual)
- **T²-RAGBench relevance**: Token-level matching could help with table cell retrieval where specific numbers matter

### 2.4 Hybrid Search (Sparse + Dense Fusion)

#### Fusion Methods
| Method | Description | Key Parameter |
|--------|-------------|---------------|
| **RRF (Reciprocal Rank Fusion)** | score = Σ 1/(k + rank_i) | k (default 60) |
| **Convex Combination (CC)** | score = α·sparse + (1-α)·dense | α ∈ [0, 1] |
| **DBSF (Distribution-Based Score Fusion)** | Normalize by score distribution then combine | — |
| **Learned Fusion** | Train a model to learn optimal weights | Requires training data |

- **Paper**: Cormack et al. (2009) for RRF
- **Implementation**: LangChain EnsembleRetriever, LlamaIndex, Qdrant hybrid, Weaviate hybrid
- **Key finding from T²-RAGBench**: Hybrid BM25 was the best method (41.3% NM vs 37.2% Base-RAG)
- **Recent insight**: Hybrid consistently outperforms single-method in most benchmarks
- **Our opportunity**: Original paper only tested one fusion approach — we can compare RRF, CC, DBSF

### 2.5 Cross-Encoder Reranking

#### Current Reranker Leaderboard (Agentset, Feb 2026)
| Rank | Model | ELO | Latency | Open Source |
|------|-------|-----|---------|-------------|
| 1 | Zerank-2 | 1638 | 265ms | Yes (CC-BY-NC) |
| 2 | Cohere Rerank 4 Pro | 1629 | 614ms | No |
| 3 | Zerank-1 | 1573 | 266ms | Yes (CC-BY-NC) |
| 4 | Voyage AI Rerank 2.5 | 1544 | 613ms | No |
| 5 | Cohere Rerank 4 Fast | 1510 | 447ms | No |
| 8 | Qwen3 Reranker 8B | 1473 | 4687ms | Yes (Apache 2.0) |
| 10 | Cohere Rerank 3.5 | 1451 | 392ms | No |
| 11 | BGE-reranker-v2-m3 | 1327 | 2383ms | Yes (Apache 2.0) |
| 12 | Jina Reranker v2 | 1327 | 746ms | Yes (CC-BY-NC) |

**Reranking approaches**:
- **Pointwise**: Score each (query, doc) pair independently — most common
- **Listwise**: Score a list of documents jointly — RankGPT (Sun et al., 2023)
- **Pairwise**: Compare documents in pairs
- **LLM-as-Reranker**: Use GPT-4 / Claude to rerank — expensive but powerful

**Key finding**: Reranking can improve retrieval by up to 48% (Databricks research). However, in T²-RAGBench, the reranker actually HURT performance (31.8% vs 37.2% Base-RAG), suggesting the reranker used was suboptimal or the domain is challenging.

### 2.6 HyDE (Hypothetical Document Embeddings)
- **Paper**: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
- **Mechanism**: Query → LLM generates hypothetical answer document → embed hypothetical doc → retrieve similar real docs
- **Implementation**: LangChain HypotheticalDocumentEmbedder, Haystack
- **Strengths**: Works well for zero-shot settings; bridges query-document vocabulary gap
- **Weaknesses**: Adds LLM latency per query; quality depends on LLM's domain knowledge; hallucinated pseudo-docs can mislead retrieval
- **T²-RAGBench result**: HyDE HURT performance (34.0% vs 37.2% Base-RAG) — likely because financial/numerical queries are hard for LLM to hypothesize about
- **Cost**: ~$0.001-0.01 per query (depends on LLM)

### 2.7 HyPE (Hypothetical Passage Embeddings)
- **Paper**: Vake et al. (2025)
- **Mechanism**: At INDEX TIME, generate synthetic questions for each chunk → embed questions alongside chunks → at query time, match query to synthetic questions
- **Key difference from HyDE**: Index-time (one-time cost) vs query-time (per-query cost)
- **Results**: Up to +42pp precision and +45pp recall improvement on certain datasets
- **Implementation**: No established library yet — custom implementation needed
- **Strengths**: No query-time overhead; query-question matching is more natural than query-document matching
- **Weaknesses**: Index-time LLM cost (N questions per chunk); storage overhead; quality depends on synthetic question diversity
- **T²-RAGBench relevance**: Could be very effective — financial questions have specific patterns that could be generated

### 2.8 Contextual Retrieval (Anthropic)
- **Blog**: Anthropic, "Introducing Contextual Retrieval" (September 2024)
- **Mechanism**: For each chunk, use an LLM to prepend document-level context summary → embed the contextualized chunk. Combined with BM25 + reranking.
- **Results**: 49% fewer retrieval failures (contextual embedding alone), 67% fewer with reranking
- **Cost**: ~$1.02 per million document tokens (one-time indexing cost, assuming 800-token chunks, 8k-token docs)
- **Implementation**: Custom — use Claude/GPT to generate context prefix per chunk
- **Prompt pattern**: "Given this document, provide a short context for the following chunk that would help retrieve it"
- **T²-RAGBench relevance**: Financial documents have shared context (company name, year, metric types) that could greatly benefit chunks

### 2.9 Late Chunking (Jina AI)
- **Paper**: Jina AI (2024), arXiv:2409.04701
- **Mechanism**: Embed full document through long-context model → get token-level embeddings → THEN chunk the embedding sequence → mean-pool each chunk's tokens
- **Advantage**: Each chunk's embedding preserves full document context naturally (no LLM needed)
- **Requires**: Long-context embedding model (jina-embeddings-v2/v3 with 8K context)
- **Implementation**: jina-ai/late-chunking (GitHub), Jina API (`late_chunking=true`)
- **Results**: Consistent improvement over naive chunking; improvement increases with document length
- **T²-RAGBench relevance**: Documents are ~920 tokens avg — within 8K context window, so late chunking is applicable. However, since original paper uses whole documents (not chunks), late chunking's advantage may be limited unless we also implement chunking.

### 2.10 GraphRAG / KG-Enhanced Retrieval

#### Microsoft GraphRAG
- **Paper**: Edge et al. (2024)
- **Mechanism**: Extract entities/relations → build KG → Leiden clustering → community summaries → two search modes (Global: community summaries, Local: entity neighborhoods)
- **Implementation**: microsoft/graphrag (Python), nano-graphrag
- **Strengths**: Excellent for multi-hop reasoning, thematic/global questions, entity-centric queries
- **Weaknesses**: Very expensive (many LLM calls for extraction), slow indexing, complex setup

#### LightRAG
- **Paper**: Guo et al. (2024)
- **Mechanism**: Lightweight graph construction + dual-level retrieval (entity-level + relation-level)
- **Performance**: 70-90% of GraphRAG quality at 1/100th cost
- **Implementation**: HKUDS/LightRAG (GitHub)

#### GraphRAG-Bench (ICLR 2026)
- Recent benchmark shows: NaiveRAG slightly outperformed LightRAG after eliminating evaluation biases
- GraphRAG yields ~10% accuracy increase on relational QA but at much higher cost
- **T²-RAGBench relevance**: Financial tables have entity-relation structures (company → metric → value → year), but the overhead may not justify the gain for primarily numerical QA

### 2.11 Multi-Query / RAG-Fusion
- **Paper**: Raudaschl (2023) for RAG-Fusion
- **Mechanism**: LLM generates N query variants → retrieve for each → merge via RRF
- **Implementation**: LangChain MultiQueryRetriever
- **Recent finding (2025)**: In production deployment, "fusion variants failed to outperform single-query baselines after reranking" — retrieval recall gains were "neutralized after re-ranking and truncation"
- **Alternative**: MQRF-RAG (2025) — multiple rewriting strategies combined, +14.45% P@5 on FreshQA
- **T²-RAGBench relevance**: Financial questions are already specific — multi-query may not help much. Worth testing.

### 2.12 Parent-Child / Hierarchical Retrieval
- **Mechanism**: Index small child chunks (100-500 tokens), retrieve on children, return parent chunks (500-2000 tokens) as context
- **Implementation**: LlamaIndex (ParentDocumentRetriever), LangChain (ParentDocumentRetriever), Dify v0.15.0
- **Strengths**: Fine-grained matching + broad context; decouples retrieval precision from context completeness
- **T²-RAGBench note**: Original paper uses whole documents (~920 tokens) as retrieval units. If we chunk, parent-child could help preserve table context while matching on specific cells.

### 2.13 RAPTOR
- **Paper**: Sarthi et al. (ICLR 2024)
- **Mechanism**: Cluster chunks → summarize clusters → recursively build tree → retrieve at multiple abstraction levels
- **Clustering**: GMM with UMAP dimensionality reduction; BIC for optimal cluster count
- **Results**: +20% absolute accuracy on QuALITY benchmark (with GPT-4); ROUGE-L 30.87% on NarrativeQA
- **Implementation**: parthsarthi03/raptor (GitHub)
- **Strengths**: Excels at broad/thematic questions requiring multi-document reasoning
- **Weaknesses**: Many LLM calls for summarization; complex tree structure; may lose numerical precision in summaries
- **T²-RAGBench relevance**: RISK — summarization of financial tables may lose numerical precision. The original paper showed summarization methods underperform on VQAonBD and TAT-DQA.

### 2.14 Agentic / Adaptive RAG

#### Self-RAG
- **Paper**: Asai et al. (2023)
- **Mechanism**: Train LLM with special reflection tokens to decide: (a) when to retrieve, (b) whether retrieved docs are relevant, (c) whether output is supported
- **Requires**: Fine-tuned model with reflection tokens — not easily usable with arbitrary LLMs

#### CRAG (Corrective RAG)
- **Paper**: Yan et al. (2024)
- **Mechanism**: Retrieve → evaluate relevance → if low confidence, trigger web search fallback → refine
- **Modular**: Plug-and-play, works with any LLM/retriever
- **Results**: Self-CRAG outperformed Self-RAG by 20% on PopQA, 36.9% on Biography

#### Adaptive RAG
- Route between: no-retrieval / single-step / multi-step based on query complexity
- **Implementation**: LangGraph workflows

**T²-RAGBench relevance**: All questions require retrieval (parametric-only is 3.9%), so adaptive routing is less relevant. CRAG's confidence-based re-retrieval could help for hard cases.

### 2.15 Sentence Window Retrieval
- **Mechanism**: Embed individual sentences → retrieve top-k sentences → expand to N surrounding sentences for context
- **Implementation**: LlamaIndex SentenceWindowNodeParser (window_size parameter)
- **Strengths**: Very fine-grained matching; good for factoid questions
- **Weaknesses**: May miss table context; window expansion is heuristic
- **T²-RAGBench note**: May not work well — tables span multiple "sentences" and window expansion may cut tables mid-row

### 2.16 Step-Back Prompting + Retrieval
- **Paper**: Zheng et al. (2023) — "Take a Step Back"
- **Mechanism**: LLM generates abstract/high-level question → retrieve for abstract question → use retrieved context + original question to answer
- **Example**: "What was AAPL revenue in 2019?" → Step-back: "What were Apple's key financial metrics in recent years?"
- **Strengths**: Gets broader context; helps with questions requiring background
- **Weaknesses**: Abstract questions may retrieve irrelevant docs; extra LLM call
- **T²-RAGBench relevance**: Financial questions are already specific — step-back may retrieve too broadly

### 2.17 2025-2026 Innovations

#### Mixture of Retrievers (MoR)
- **Paper**: EMNLP 2025
- **Mechanism**: Combine BM25, DPR, SimCSE with learned trustworthiness weights; zero-shot routing
- **Strengths**: Handles diverse query types without manual fusion tuning

#### ExpertRAG (Mixture of Experts for RAG)
- **Paper**: 2025
- **Mechanism**: Gating expert routes queries to specialized retrieval/generation experts
- **Results**: Dynamic routing outperforms static pipelines

#### Speculative Pipelining
- **Paper**: Wang et al. (2025)
- **Mechanism**: Overlap retrieval and generation — start generating with partial results
- **Focus**: Latency reduction, not quality improvement

#### RAGRoute (Federated RAG)
- **Mechanism**: Neural classifier for dynamic data source selection — reduces queries by 77.5%

---

## 3. Embedding Model Comparison

### Comprehensive Comparison Table (March 2026)

| Model | Provider | MTEB Overall | Retrieval Score | Dims | Max Context | Open Source | Cost/1M tokens |
|-------|----------|-------------|-----------------|------|-------------|-------------|----------------|
| Qwen3-Embedding-8B | Alibaba | **70.58** | High | 7,168 (flex→32) | 32,000 | Yes (Apache 2.0) | Free |
| NV-Embed-v2 | NVIDIA | 69.32 | 69.32 | 4,096 | 32,768 | Partial (CC-BY-NC) | Free |
| Gemini embedding-001 | Google | 68.32 | 67.71 | 3,072 (flex→768) | 2,048 | No | $0.15 |
| Voyage-3-large | Voyage AI | ~67+ | Top tier | 2,048 (flex) | 32,000 | No | $0.06 |
| Cohere embed-v4 | Cohere | 65.2 | — | 1,024 | 128,000 | No | $0.10 |
| OpenAI text-embed-3-large | OpenAI | 64.6 | — | 3,072 (flex) | 8,192 | No | $0.13 |
| BGE-M3 | BAAI | 63.0 | Strong | 1,024 | 8,192 | Yes (MIT) | Free |
| Jina embeddings-v3 | Jina AI | ~62+ | ~62+ | 1,024 (flex→32) | 8,192 | Partial (CC-BY-NC) | $0.018 |
| E5-Mistral-7B-Instruct | Microsoft | ~62 | — | 4,096 | 32,768 | Yes (MIT) | Free |
| Multilingual E5-Large-Instruct | Microsoft | ~60 | 38.6 (T²-RAG) | 1,024 | 512 | Yes (MIT) | Free |
| Nomic embed-text-v1.5 | Nomic | ~62+ | ~62+ | 768 (flex→64) | 8,192 | Yes (Apache 2.0) | Free |
| all-MiniLM-L6-v2 | SBERT | 56.3 | — | 384 | 512 | Yes (Apache 2.0) | Free |

**T²-RAGBench tested**: E5-Large-Instruct (primary), OpenAI text-embed-3-large (best at R@1=33.8%)

### Recommended Models for Our Study
1. **OpenAI text-embedding-3-large** — best T²-RAGBench result, commercial baseline
2. **BGE-M3** — best open-source option, MIT license, free
3. **Cohere embed-v4** — competitor, 128K context
4. **Jina embeddings-v3** — for late chunking experiments
5. **E5-Mistral-7B-Instruct** — large instruction-tuned open model
6. **(Optional) Qwen3-Embedding-8B** — MTEB leader, but large

---

## 4. Reranker Comparison

| Model | ELO | Latency | Cost/1M | Open Source | Best For |
|-------|-----|---------|---------|-------------|----------|
| Zerank-2 | 1638 | 265ms | $0.025 | CC-BY-NC | Best overall quality |
| Cohere Rerank 4 Pro | 1629 | 614ms | $0.050 | No | Production API |
| Voyage Rerank 2.5 | 1544 | 613ms | $0.050 | No | Balanced quality/latency |
| BGE-reranker-v2-m3 | 1327 | 2383ms | $0.020 | Apache 2.0 | Self-hosted, free |
| Jina Reranker v2 | 1327 | 746ms | $0.045 | CC-BY-NC | Multilingual |

**For our study**: BGE-reranker-v2-m3 (free, open) + Cohere Rerank 4 Pro (best API) as primary comparison pair.

---

## 5. Evaluation Metrics

### 5.1 Retrieval Metrics (we will report all)

| Metric | Formula | When to Use | Priority |
|--------|---------|-------------|----------|
| **Recall@k** (k=1,3,5,10,20) | Relevant retrieved in top-k / Total relevant | Primary — measures retrieval completeness | ★★★ |
| **MRR@k** | Mean(1/rank of first relevant) | Measures ranking quality | ★★★ |
| **nDCG@k** (k=5,10) | Graded relevance with position discount | When relevance is graded | ★★☆ |
| **Precision@k** | Relevant in top-k / k | When precision matters (small k) | ★★☆ |
| **MAP** | Mean of AP across queries | Overall ranking quality | ★★☆ |
| **Hit Rate@k** | % queries with ≥1 relevant in top-k | Simple coverage metric | ★☆☆ |

### 5.2 Generation Metrics

| Metric | Description | Library | Priority |
|--------|-------------|---------|----------|
| **Number Match (NM)** | T²-RAGBench primary metric — relative tolerance matching | Custom | ★★★ |
| **Exact Match (EM)** | Strict string matching (after normalization) | evaluate | ★★★ |
| **Token F1** | Token-level precision/recall | evaluate | ★★★ |
| **ROUGE-L** | Longest common subsequence | rouge-score | ★★☆ |
| **BERTScore** | Contextual embedding similarity | bert-score | ★★☆ |

### 5.3 RAG-Specific Metrics (via RAGAS)

| Metric | What it measures | Requires |
|--------|-----------------|----------|
| **Faithfulness** | Is answer grounded in retrieved context? | Answer + context |
| **Answer Relevancy** | Does answer address the question? | Answer + question |
| **Context Precision** | Are relevant chunks ranked higher? | Context + ground truth |
| **Context Recall** | Were all relevant chunks retrieved? | Context + ground truth |

### 5.4 Statistical Significance
- **Paired Bootstrap Test** — standard in IR (10,000 resamples)
- **Bonferroni correction** for multiple comparisons
- Report p-values and confidence intervals
- Standard practice in ACL/EMNLP papers

### 5.5 Recommended Metrics for T²-RAGBench
- **Primary retrieval**: Recall@3, Recall@5, MRR@3 (matches original paper's k=3)
- **Extended retrieval**: Recall@1, Recall@10, nDCG@5, nDCG@10
- **Primary generation**: Number Match (NM), Token F1
- **Extended generation**: EM, ROUGE-L, BERTScore
- **RAG quality**: RAGAS Faithfulness, Context Precision
- **Efficiency**: Latency (ms/query), Index size (MB), Indexing time (s)

---

## 6. Method Matrix

| # | Method | Category | Implementation | Compute Cost | Text Strength | Table Strength | Expected on T²-RAGBench |
|---|--------|----------|---------------|-------------|--------------|---------------|------------------------|
| 1 | BM25 | Sparse | rank_bm25 / Pyserini | ★☆☆ | Good (keyword) | Good (exact numbers) | Strong baseline |
| 2 | SPLADE-v3 | Learned Sparse | naver/splade | ★★☆ | Very Good | Good | Better than BM25 |
| 3 | Dense (OpenAI) | Dense | text-embedding-3-large | ★★☆ | Very Good | Moderate | Similar to original |
| 4 | Dense (BGE-M3) | Dense | BAAI/bge-m3 | ★★☆ | Very Good | Moderate | Slightly below OpenAI |
| 5 | Dense (E5-Mistral) | Dense | E5-Mistral-7B | ★★★ | Very Good | Moderate | Potentially strong |
| 6 | ColBERTv2 | Late Interaction | RAGatouille | ★★★ | Excellent | Good (token-level) | **High potential** |
| 7 | Hybrid BM25+Dense (RRF) | Hybrid | Custom | ★★☆ | Very Good | Good | Matches original best |
| 8 | Hybrid BM25+Dense (CC) | Hybrid | Custom | ★★☆ | Very Good | Good | Compare with RRF |
| 9 | Hybrid + Reranker | Hybrid+Rerank | Cohere/BGE | ★★★ | Excellent | Good | **Strong contender** |
| 10 | HyDE | Query Expansion | LangChain | ★★★ | Good | Poor (hallucination) | Likely underperforms |
| 11 | HyPE | Index Augmentation | Custom | ★★★ (index) | Good | Moderate | Worth testing |
| 12 | Contextual Retrieval | Context Augment | Custom (Claude) | ★★★ (index) | Excellent | **Very Good** | **High potential** |
| 13 | Late Chunking | Chunking Strategy | Jina API | ★★☆ | Very Good | Good | Moderate improvement |
| 14 | Multi-Query + RRF | Query Expansion | LangChain | ★★★ | Good | Moderate | Mixed — may not help |
| 15 | Parent-Child | Hierarchical | LlamaIndex | ★★☆ | Good | **Good** | Moderate improvement |
| 16 | RAPTOR | Tree Summary | raptor-py | ★★★★ | Very Good (broad) | Poor (loses numbers) | Likely weak on numerical |
| 17 | GraphRAG | KG-Enhanced | LightRAG | ★★★★ | Good (relations) | Moderate | Mixed — high cost |
| 18 | Agentic (CRAG) | Adaptive | LangGraph | ★★★★ | Good | Moderate | Marginal improvement |
| 19 | Sentence Window | Fine-Grained | LlamaIndex | ★★☆ | Good | Poor (breaks tables) | Likely weak |
| 20 | Best Combo | Ensemble | Custom | ★★★ | Excellent | **Very Good** | **Best expected** |

---

## 7. Experiment Plan

### 7.1 Tier 1 — Core Experiments (Must Have)

These form the main comparison table of the paper:

| # | Experiment | Details |
|---|-----------|---------|
| E1 | BM25 Baseline | rank_bm25, default params |
| E2 | Dense (OpenAI) | text-embedding-3-large, FAISS, top-k=[3,5,10] |
| E3 | Dense (BGE-M3) | Open-source comparison |
| E4 | ColBERTv2 | RAGatouille, PLAID index |
| E5 | Hybrid (BM25 + OpenAI, RRF) | α=0.5, k=60 |
| E6 | Hybrid (BM25 + OpenAI, CC) | Compare fusion methods |
| E7 | Hybrid + BGE Reranker | Free reranker |
| E8 | Hybrid + Cohere Rerank 4 | Best API reranker |
| E9 | HyDE + Dense | GPT-4o-mini pseudo-doc |
| E10 | HyPE + Dense | Index-time synthetic queries |
| E11 | Contextual Retrieval | Claude context + BM25 + rerank |
| E12 | Multi-Query + RRF | 3 sub-queries, GPT-4o-mini |
| E13 | Parent-Child | Child=256tok, Parent=full doc |
| E14 | Best Combination | Best retriever + best reranker + best query expansion |

### 7.2 Tier 2 — Extended Experiments (Important but Optional)

| # | Experiment | Details |
|---|-----------|---------|
| E15 | SPLADE-v3 | Learned sparse |
| E16 | Late Chunking (Jina) | jina-embeddings-v3 |
| E17 | RAPTOR | Tree-based, GPT-4o-mini for summaries |
| E18 | LightRAG | Graph-based retrieval |
| E19 | CRAG (Corrective) | Confidence-based re-retrieval |
| E20 | Sentence Window | window_size=[1,3,5] |

### 7.3 Ablation Studies

| # | Ablation | Variables |
|---|---------|-----------|
| A1 | **Embedding model effect** | Same hybrid+rerank pipeline, swap: OpenAI / BGE-M3 / E5-Mistral / Cohere / Jina |
| A2 | **Chunk size effect** | 256, 512, 1024, full-doc (with best method) |
| A3 | **Top-k effect** | k=1,3,5,10,20 retrieval depth curves |
| A4 | **Reranker effect** | No reranker / BGE / Cohere / ColBERT-rerank |
| A5 | **Table representation** | Markdown table / Linearized text / Table-as-sentences |
| A6 | **Fusion weight sensitivity** | RRF k∈{10,30,60,100}; CC α∈{0.3,0.5,0.7} |
| A7 | **Corpus size scaling** | 25%, 50%, 75%, 100% (random subsets) |

### 7.4 End-to-End Generation Pipeline

- **LLM**: GPT-4o-mini (cost-effective, deterministic with temperature=0)
- **Alternative**: GPT-4.1-mini if available
- **Same prompt template** across all methods
- **Top N retrieval methods** (top 5-7 based on retrieval results) → full generation pipeline
- **Measure**: NM, EM, F1, ROUGE-L, BERTScore, RAGAS Faithfulness

### 7.5 Cost Estimate

| Component | Estimated Cost | Notes |
|-----------|---------------|-------|
| OpenAI Embeddings (23K queries + 7K docs) | ~$2-3 | text-embedding-3-large |
| Cohere Embeddings | ~$1-2 | embed-v4 |
| Cohere Reranker | ~$5-10 | 23K queries × top-50 docs |
| HyDE (GPT-4o-mini) | ~$2-3 | 23K pseudo-docs |
| HyPE (GPT-4o-mini) | ~$3-5 | 5 questions per chunk |
| Contextual Retrieval (Claude) | ~$5-8 | Context for all chunks |
| Multi-Query (GPT-4o-mini) | ~$1-2 | 3 queries × 23K |
| RAPTOR summaries | ~$3-5 | Recursive summarization |
| Generation (GPT-4o-mini) | ~$5-10 | 23K generations × 5-7 methods |
| **Total estimated** | **~$30-50** | |

### 7.6 Execution Order
1. **Phase 1a**: Setup, data download, preprocessing (no cost)
2. **Phase 1b**: BM25, Dense (BGE-M3, free) — verify pipeline works
3. **Phase 2a**: Tier 1 core experiments (E1-E8) — ~$10
4. **Phase 2b**: Index-augmentation methods (E9-E11) — ~$10-15
5. **Phase 2c**: Remaining Tier 1 + ablations — ~$10
6. **Phase 2d**: Tier 2 experiments — ~$5-10
7. **Phase 3**: Top-N end-to-end generation — ~$5-10

---

## 8. Paper Positioning & Novelty

### 8.1 Gap in Literature
The T²-RAGBench paper (Strich et al., 2026) introduced a valuable benchmark but tested only 6 retrieval methods with limited metrics. There is **no comprehensive comparison** of modern retrieval strategies on text+table heterogeneous financial data.

### 8.2 Our Contribution (4 bullet points for the paper)
1. **Most comprehensive RAG retrieval benchmark to date**: We systematically evaluate 15-20 retrieval methods on T²-RAGBench, including 10+ methods not tested in the original paper (ColBERT, Contextual Retrieval, HyPE, Late Chunking, GraphRAG, RAPTOR, Parent-Child, SPLADE, etc.)
2. **Multi-dimensional evaluation**: We go beyond NM and MRR to provide Recall@k, nDCG, MAP, generation quality (EM, F1, BERTScore), and RAG-specific metrics (RAGAS Faithfulness) with statistical significance testing.
3. **Extensive ablation studies**: We isolate the effect of embedding models, chunk sizes, rerankers, fusion methods, and table representation strategies — providing actionable guidelines for RAG practitioners.
4. **Practical recommendations**: We provide a decision framework (cost vs. accuracy trade-off, Pareto analysis) for choosing retrieval methods for text+table RAG systems.

### 8.3 How This Differs from Existing Work
| Existing Work | Our Contribution |
|--------------|-----------------|
| T²-RAGBench (Strich et al., 2026): 6 methods, 2 metrics | 15-20 methods, 10+ metrics, ablations |
| BEIR (Thakur et al., 2021): Text-only benchmark | Focus on text+table heterogeneous documents |
| MTEB (Muennighoff et al., 2023): Embedding comparison only | Full pipeline comparison (retrieval + generation) |
| RAGBench (various): General QA | Financial domain, numerical reasoning |
| GraphRAG-Bench (ICLR 2026): Graph methods only | All retrieval paradigms compared fairly |

### 8.4 Expected Key Findings (Hypotheses)
1. Hybrid (BM25 + Dense + Reranker) will be the overall best, confirming T²-RAGBench
2. Contextual Retrieval will significantly improve over vanilla hybrid
3. ColBERTv2 will excel at table-specific queries due to token-level matching
4. HyDE will underperform (confirmed by original paper)
5. GraphRAG will have high cost but modest improvement for numerical QA
6. Better retrieval will clearly lead to better generation quality (strong correlation)
7. Table representation strategy will significantly affect results

---

## 9. Recommended Venue

### Primary Recommendation: **EMNLP 2026** (Main Conference or Findings)
- **Deadline**: Likely June 2026
- **Fit**: Strong — EMNLP values empirical studies, benchmarking, and resource papers
- **Format**: 8 pages + references + appendix

### Alternative Options:
| Venue | Deadline (est.) | Fit | Notes |
|-------|----------------|-----|-------|
| **ACL 2026 ARR** | Rolling (submit by April 2026) | ★★★★ | Top venue, competitive |
| **EMNLP 2026** | ~June 2026 | ★★★★★ | Best fit for empirical benchmark |
| **NAACL 2026** | ~January 2026 | ★★★ | Already passed |
| **SIGIR 2026** | ~Feb 2026 | ★★★★ | IR-focused, good fit |
| **ECIR 2027** | ~Oct 2026 | ★★★ | European IR venue |
| **COLM 2026** | TBD | ★★★ | New venue for LM research |

### Recommendation
Start with **ACL 2026 ARR** (rolling submission) — if reviews suggest improvements, revise and resubmit to **EMNLP 2026**. The paper scope and contribution level are appropriate for a main conference.

---

## 10. Open Questions for Discussion

Before proceeding to Phase 1, I need your input on these decisions:

### Q1: Scope of Methods
The full 20-method plan is ambitious. Should we:
- (a) Test all 20 methods (comprehensive, but takes longer)
- (b) Focus on Tier 1 (14 methods) — still very comprehensive
- (c) Prioritize differently?

### Q2: Chunking Strategy
The original paper uses **whole documents** (~920 tokens) as retrieval units. Should we:
- (a) Keep whole documents (matches original, but limits chunking-based methods)
- (b) Also test with chunked documents (enables parent-child, sentence window, late chunking)
- (c) Both — whole-doc experiments first, then chunked ablation

### Q3: Embedding Models
Testing 5+ embedding models across all methods is expensive. Should we:
- (a) Use one primary model (OpenAI text-embedding-3-large) for all methods, then ablate on the best method
- (b) Test 2-3 models (OpenAI + BGE-M3 + one more) across all
- (c) Other preference?

### Q4: Generation LLM
- (a) GPT-4o-mini only (cheapest, sufficient)
- (b) GPT-4o-mini + GPT-4.1-mini (compare)
- (c) Also include an open model (Llama 3.3-70B to match original paper)?

### Q5: API vs Self-Hosted
Do you have GPU access? This affects:
- ColBERTv2 indexing (needs GPU)
- BGE-M3 embedding (GPU preferred)
- E5-Mistral (needs GPU)
- SPLADE (needs GPU)

### Q6: Budget Approval
Estimated total cost: **$30-50** for API calls. Is this budget acceptable?

### Q7: Timeline
What is our target deadline? This determines how many experiments we can run.

---

## References (Key Papers)

1. Strich et al. (2026). T²-RAGBench: Text-and-Table Benchmark for RAG. EACL 2026.
2. Gao et al. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE). arXiv:2212.10496.
3. Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. NAACL 2022.
4. Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model. SIGIR 2021.
5. Sarthi et al. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. ICLR 2024.
6. Asai et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
7. Yan et al. (2024). Corrective Retrieval Augmented Generation (CRAG).
8. Edge et al. (2024). From Local to Global: A Graph RAG Approach. Microsoft Research.
9. Anthropic (2024). Introducing Contextual Retrieval. Blog post.
10. Jina AI (2024). Late Chunking in Long-Context Embedding Models. arXiv:2409.04701.
11. Raudaschl (2023). RAG-Fusion: Multi-query + RRF.
12. Sun et al. (2023). RankGPT: Is ChatGPT Good at Search? (LLM-as-Reranker).
13. Cormack et al. (2009). Reciprocal Rank Fusion (RRF).
14. Es et al. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation.
15. Zheng et al. (2023). Take a Step Back: Evoking Reasoning via Abstraction.
16. Guo et al. (2024). LightRAG: Simple and Fast Retrieval-Augmented Generation.
17. Vake et al. (2025). Hypothetical Passage Embeddings (HyPE).
