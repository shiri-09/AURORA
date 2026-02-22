<div align="center">

# 🔥 AURORA

### **Auditable Unlearning for Relational & Orchestrated Reasoning Architectures**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qiskit 1.0+](https://img.shields.io/badge/Qiskit-1.0+-6929C4?style=for-the-badge&logo=ibm&logoColor=white)](https://qiskit.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7DF1E?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Bounded multi-hop irrecoverable unlearning in transformer-based LLMs**
**with adversarial robustness, quantum verification, and cryptographic auditability.**

[Quick Start](#-quick-start) · [Architecture](#-system-architecture) · [Modules](#-module-deep-dive) · [Demo](#-interactive-dashboard) · [Citation](#-citation)

---

</div>

## 🧬 What is AURORA?

AURORA is a **certified machine unlearning framework** that surgically removes specific knowledge from pretrained language models — not by retraining from scratch, but by precisely identifying and modifying the minimal set of parameters responsible for storing a target fact, while provably preserving all other capabilities.

> **The Problem:** GDPR's "Right to be Forgotten" (Art. 17) and similar regulations require the ability to delete specific learned information from AI models. Naive approaches (full retraining, output filtering) are either computationally prohibitive or trivially bypassed.
>
> **AURORA's Solution:** A 7-step pipeline that maps knowledge topology → localizes responsible parameters → cascade-deletes via gradient descent on an inverted loss → validates against adversarial attacks → certifies with cryptographic proofs → verifies via quantum-inspired trace distance metrics.

### Key Differentiators

| Feature | Naive Fine-Tuning | Output Filtering | **AURORA** |
|---|:---:|:---:|:---:|
| True parameter-level deletion | ❌ | ❌ | ✅ |
| Multi-hop indirect leakage prevention | ❌ | ❌ | ✅ |
| Adversarial robustness (5 attack classes) | ❌ | ❌ | ✅ |
| Retain utility preservation | ⚠️ Degrades | ✅ | ✅ |
| Cryptographic audit certificate | ❌ | ❌ | ✅ |
| Quantum-verified embedding separation | ❌ | ❌ | ✅ |

---

## 🏗️ System Architecture

```
                          ┌──────────────────────────────────┐
                          │       AURORA Pipeline (7 Steps)   │
                          └──────────────┬───────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
    ┌────▼─────┐                   ┌─────▼──────┐                 ┌──────▼──────┐
    │  Step 1   │                   │   Step 2    │                 │   Step 3     │
    │  Target   │──────────────────▶│    RKG      │────────────────▶│  Parameter   │
    │  Formal.  │  (s, r, o) triple │   Builder   │  Graph G(V,E)  │  Localizer   │
    └──────────┘                   └────────────┘                 └──────┬──────┘
                                                                         │
                                                                    θ_risk ⊂ θ
                                                                         │
    ┌──────────┐                   ┌────────────┐                 ┌──────▼──────┐
    │  Step 7   │                   │   Step 6    │                 │   Step 4     │
    │  Quantum  │◀─────────────────│   Crypto    │◀────────────────│  Cascade     │
    │  Verify   │  Trace Distance  │   Certify   │  θ' (unlearned) │  Optimizer   │
    └──────────┘                   └────────────┘                 └─────────────┘
                                         │
                                   ┌─────▼──────┐
                                   │   Step 5    │
                                   │ Adversarial │
                                   │   Suite     │
                                   └────────────┘
```

### Mathematical Foundation

**Convergence Objective:**

```
min_θ' [ L_forget(θ') + λ·L_retain(θ') + γ·‖θ' − θ‖² ]

where:
  L_forget = E[ −log(1 − P(y_t | x, θ')) ]     ← pushes P(target) → 0
  L_retain = KL( P_θ(·|x_r) ‖ P_θ'(·|x_r) )   ← preserves utility
  ‖θ' − θ‖²                                      ← stability regularizer
```

**Formal Guarantee:**

```
∀ a ∈ A_tested:  sup P(y_t | a(x), θ') ≤ ε

where A_tested = { Correlated FT, Prompt Injection, Multi-Hop, Quantization, LoRA Recovery }
```

---

## ⚡ Quick Start

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA optional, CPU supported)
- 8GB+ RAM recommended

### Installation

```bash
git clone https://github.com/shiri-09/AURORA.git
cd AURORA

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
# Full pipeline with GPT-2 (runs on CPU in ~40s)
python scripts/run_demo.py

# Interactive dashboard with live visualization
python scripts/demo_server.py
# → Open http://localhost:5000
```

### API Server

```bash
# Start FastAPI backend
uvicorn aurora.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 📂 Project Structure

```
AURORA/
│
├── aurora/                             # Core framework
│   ├── config.py                       # Global configuration & hyperparameters
│   ├── types.py                        # Shared data types (ForgetSet, RetainSet, etc.)
│   │
│   ├── fact_formalization/             # Module 1: Target Fact Manager
│   │   └── formalizer.py               #   (subject, relation, object) → ForgetSet
│   │
│   ├── knowledge_graph/                # Module 2: Relational Knowledge Graph
│   │   ├── rkg_builder.py              #   Builds G(V,E) from embeddings & attention
│   │   └── graph_analyzer.py           #   Computes influence, centrality, paths
│   │
│   ├── parameter_localization/         # Module 3: Influence-Based Parameter Selection
│   │   ├── gradient_tracer.py          #   Gradient attribution analysis
│   │   └── fisher_selector.py          #   Fisher Information masking → θ_risk
│   │
│   ├── cascade_optimizer/              # Module 4: Cascade Unlearning Engine
│   │   ├── optimizer.py                #   Iterative masked gradient descent
│   │   └── losses.py                   #   L_forget + λ·L_retain + γ·L_stability
│   │
│   ├── adversarial_suite/              # Module 5: Adversarial Robustness Testing
│   │   ├── evaluator.py                #   Orchestrates all 5 attacks
│   │   └── attacks/
│   │       ├── correlated_finetuning.py    # Attack 1: Recovery via related fine-tuning
│   │       ├── prompt_injection.py         # Attack 2: Adversarial prompt extraction
│   │       ├── multihop_reconstruction.py  # Attack 3: Multi-hop reasoning chains
│   │       ├── quantization_attack.py      # Attack 4: 8-bit quantization leakage
│   │       └── lora_attack.py              # Attack 5: LoRA adapter recovery
│   │
│   ├── certification/                  # Module 6: Cryptographic Certification
│   │   ├── certificate_generator.py    #   Generates signed compliance certificates
│   │   └── merkle_tree.py              #   Merkle tree for metric verification
│   │
│   ├── quantum_distinguishability/     # Module 7: Quantum Verification
│   │   ├── analyzer.py                 #   Layer A: numpy/scipy density matrices
│   │   └── qiskit_analyzer.py          #   Layer B: IBM Qiskit 4-qubit circuit sim
│   │
│   ├── pipeline/                       # End-to-end orchestrator
│   │   └── orchestrator.py             #   Steps 1→7 execution engine
│   │
│   ├── api/                            # FastAPI REST backend
│   │   ├── app.py                      #   CORS, middleware, lifecycle
│   │   └── routes.py                   #   /unlearn, /results, /certificate
│   │
│   └── utils/                          # Shared utilities
│       ├── embeddings.py               #   Token probability extraction
│       ├── math_ops.py                 #   Leakage probability computation
│       └── logging.py                  #   Module-specific loggers
│
├── scripts/
│   ├── run_demo.py                     # Headless demo runner
│   ├── demo_server.py                  # Flask dashboard server
│   ├── verify_unlearning.py            # Standalone verification script
│   └── templates/
│       └── dashboard.html              # Interactive D3.js visualization UI
│
├── tests/
│   ├── test_fact_formalization.py       # Unit tests: fact parsing
│   ├── test_knowledge_graph.py          # Unit tests: RKG construction
│   ├── test_certification.py            # Unit tests: certificate generation
│   ├── test_quantum.py                  # Unit tests: quantum metrics
│   └── test_math_ops.py                # Unit tests: math utilities
│
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Build configuration
└── AURORA_PITCH_GUIDE.md              # Comprehensive presentation guide
```

---

## 🔬 Module Deep Dive

### Module 1 — Target Fact Formalization

Transforms natural language into structured knowledge triples:

```python
Input:  subject="Eiffel Tower", relation="location", object="Paris"
Output: ForgetSet(
    subject="Eiffel Tower",
    relation="location",
    object="Paris",
    direct_prompts=["The Eiffel Tower is located in", ...],
    indirect_prompts=["What European city has the famous iron tower?", ...],
    multihop_prompts=["The architect of the tower in ___ also designed...", ...]
)
```

### Module 2 — Relational Knowledge Graph (RKG)

Constructs a graph `G(V, E)` mapping how knowledge is stored relationally:

- **Prompt Nodes** — test prompts probing the model
- **Entity Nodes** — subject/object entities from the triple
- **Fact Nodes** — hashed knowledge triples (SHA-256 truncated for privacy)
- **Edge Types** — semantic similarity & retrieval paths
- **Metrics** — attention importance, gradient sensitivity, frequency scores

### Module 3 — Parameter Localization

Identifies `θ_risk ⊂ θ` using a dual-signal approach:

1. **Gradient Attribution** — computes `∂L/∂θ` for target prompts to identify high-gradient parameters
2. **Fisher Information** — estimates `F_ii = E[(∂log P/∂θ_i)²]` to weigh parameter importance
3. **Intersection Masking** — selects top-K parameters appearing in both signals

### Module 4 — Cascade Optimizer

The unlearning engine uses **gradient descent on an inverted loss** (not gradient ascent):

```
L_forget = E[ −log(1 − P(y_t | x, θ')) ]
```

This achieves the same suppression as gradient ascent but with **stable convergence**:
- Natural saturation as `P → 0`
- Compatible with retain/stability losses in a single optimization step
- No adversarial min-max instability

**Cascade expansion:** if indirect leakage persists after the initial unlearning pass, AURORA expands `θ_risk` by adding high-influence parameters from neighboring RKG nodes and re-optimizes.

### Module 5 — Adversarial Robustness Suite

Five attack classes validate unlearning durability:

| # | Attack | Method | What It Tests |
|---|--------|--------|---------------|
| 1 | **Correlated Fine-Tuning** | Train on related docs, measure fact recovery | Parametric resilience |
| 2 | **Prompt Injection** | Crafted adversarial prompts to extract target | Input-space robustness |
| 3 | **Multi-Hop Reconstruction** | Chain indirect reasoning paths | Relational completeness |
| 4 | **Quantization (8-bit)** | Quantize model, check leakage | Compression stability |
| 5 | **LoRA Recovery** | Low-rank adaptation on correlated data | Efficient fine-tuning resistance |

### Module 6 — Cryptographic Certification

Generates tamper-proof compliance certificates:

- **Merkle Tree** — hashes all metrics into a verifiable tree structure
- **Digital Signature** — RSA-2048 signing of the certificate payload
- **Compliance Tagging** — `COMPLIANT` / `NON_COMPLIANT` based on threshold checks
- **Audit Trail** — model hashes (before/after), timestamps, configuration snapshot

### Module 7 — Quantum Distinguishability Verification

Dual-layer quantum-inspired verification:

| Layer | Engine | Approach | Dimension |
|-------|--------|----------|-----------|
| **A** | numpy/scipy | Density matrix trace distance | Full embedding space |
| **B** | IBM Qiskit | 4-qubit circuit simulation (RY + CX) | PCA-compressed |

**What it measures:** The quantum trace distance between the original and unlearned model's embedding distributions. A high `D_forget/D_retain` ratio (>100×) confirms the forget set is quantum-distinguishable while the retain set is unchanged.

> **Note:** Qiskit runs in **simulator mode only** — no quantum hardware required. If Qiskit is not installed, the system gracefully falls back to Layer A (numpy/scipy) only.

---

## 🖥️ Interactive Dashboard

AURORA includes a full-featured interactive dashboard built with **D3.js** and **Flask**:

- **Live Demo Mode** — run the full pipeline from the browser
- **Real-Time Pipeline Progress** — step-by-step execution monitoring with terminal logs
- **Core Metrics Visualization** — animated bar charts for all unlearning metrics
- **Adversarial Resistance Bars** — visual breakdown of all 5 attack results
- **Force-Directed RKG Graph** — interactive D3.js visualization of the knowledge graph with draggable nodes
- **Merkle Verification Tree** — cryptographic certificate visualization
- **Quantum Distinguishability Panel** — trace distance metrics and compliance status
- **Before/After Chat** — query the model before and after unlearning

```bash
python scripts/demo_server.py
# → Open http://localhost:5000
```

---

## 📊 Benchmark Results

Results from the default demo configuration (GPT-2, CPU, "Eiffel Tower → location → Paris"):

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Direct Forget Accuracy | 0.0000 | ≤ 0.30 | ✅ PASS |
| Indirect Leakage Rate | 0.0000 | ≤ 0.30 | ✅ PASS |
| Retain Utility Drop | 0.0066 | ≤ 0.10 | ✅ PASS |
| Reconstruction Probability | 0.0040 | ≤ 0.10 | ✅ PASS |
| Parameter Drift Norm | 0.7921 | ≤ 1.00 | ✅ PASS |

| Adversarial Attack | Leakage | Status |
|---|---|---|
| Correlated Fine-Tuning | 0.0040 | ✅ RESISTED |
| Prompt Injection | 0.0000 | ✅ RESISTED |
| Multi-Hop Reconstruction | 0.0000 | ✅ RESISTED |
| Quantization (8-bit) | 0.0040 | ✅ RESISTED |
| LoRA Recovery | 0.0000 | ✅ RESISTED |

| Quantum Verification | Value |
|---|---|
| D_forget (Trace Distance) | 0.9985 |
| D_retain (Trace Distance) | 0.0010 |
| Ratio (D_forget / D_retain) | **1010×** |
| Qiskit Circuit Ratio | **1400×** |
| Certificate | `COMPLIANT` |

---

## ⚙️ Configuration

Key hyperparameters in `aurora/config.py`:

| Parameter | Symbol | Description | Default |
|---|---|---|---|
| `alpha` | α | Allowable utility drift (KL bound) | `0.01` |
| `epsilon` | ε | Leakage threshold | `0.05` |
| `lambda_retain` | λ | Retain loss weight | `1.0` |
| `gamma_stability` | γ | L2 regularization weight | `0.1` |
| `top_k_params` | K | Number of parameters to modify | `1000` |
| `cascade_threshold` | τ | Influence score cutoff for cascade expansion | `0.1` |
| `unlearn_epochs` | — | Gradient descent iterations per cascade | `5` |
| `distinguishability_threshold` | — | Quantum trace distance threshold | `0.1` |

---

## 🧪 Tech Stack

| Component | Technology |
|---|---|
| **Core ML** | PyTorch 2.0+, HuggingFace Transformers |
| **Knowledge Graph** | NetworkX, FAISS (vector similarity) |
| **Quantum Verification** | IBM Qiskit (simulator), numpy/scipy |
| **Cryptography** | Python `cryptography` (RSA-2048, SHA-256) |
| **API Backend** | FastAPI, Uvicorn |
| **Dashboard** | Flask, D3.js v7, Vanilla CSS |
| **Testing** | pytest, pytest-asyncio |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📖 Citation

If you use AURORA in your research, please cite:

```bibtex
@software{aurora2025,
  title   = {AURORA: Auditable Unlearning for Relational & Orchestrated Reasoning Architectures},
  author  = {Vijay},
  year    = {2025},
  url     = {https://github.com/shiri-09/AURORA},
  note    = {Bounded multi-hop irrecoverable unlearning with adversarial robustness and cryptographic auditability}
}
```

---

<div align="center">

**Built with precision. Engineered for compliance. Verified by quantum.**

*AURORA — Because forgetting should be as rigorous as learning.*

</div>
