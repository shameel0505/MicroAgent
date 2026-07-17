# MicroAgent Framework 🔬

**MicroAgent** is an enterprise-grade autonomous AI framework explicitly engineered to maximize the reliability and reasoning capabilities of small-parameter local LLMs (e.g., `7B` to `9B` parameter classes). By abandoning standard reactive loops in favor of a strictly deterministic state machine, MicroAgent achieves zero-shot execution stability typically reserved for massive API-gated models.

## 🌟 Enterprise Architecture

- **Deterministic State Machine**: Replaces standard ReAct loops with a strict `EVALUATE → RESEARCH → SYNTHESIS → ACTION` pipeline. This architecture prevents "hallucination loops" and forces smaller models into structured, deliberate execution.
- **Failproof Heuristics**: Advanced anomaly detection instantly identifies infinite loops, alternating failure states, or API timeouts. The orchestrator automatically applies "Context Pruning" (amnesia) to clear the model's polluted context and injects a `Critical System Override` to force a strategy pivot.
- **Sandboxed Execution**: Python code and bash commands are strictly isolated in a designated workspace subprocess, complete with hard timeouts to protect host integrity.
- **Dynamic Plugin Pipeline**: A hot-reloadable plugin architecture allows real-time capability expansion (e.g., MS Office document generation, PDF OCR) without requiring core logic restarts.

## 🧠 Cognitive Memory Systems

MicroAgent implements a sophisticated 3-tier memory architecture to simulate continuous learning:
1. **Scratchpad (Volatile)**: In-context buffer for chain-of-thought reasoning and immediate summarization.
2. **Semantic Memory (Persistent)**: A key-value knowledge graph persisting facts, configurations, and user preferences across system reboots.
3. **Episodic Memory (Append-Only)**: A chronological ledger of all past actions and observations, allowing the agent to perform historical retrospectives.

## 🔮 Future Scopes & Possibilities

To push the boundaries of what local, edge-deployed AI can accomplish, the roadmap for MicroAgent includes:

### 1. Model Sparsity & Extreme Edge Deployment
By leveraging **Sparsity** (e.g., MoE architectures) and aggressive quantization techniques (e.g., GGUF 4-bit/2-bit), MicroAgent will be optimized to run entirely in constrained edge environments (IoT devices, mobile processors) with minimal energy consumption and memory overhead, while maintaining enterprise-level reasoning via the state machine.

### 2. Open-Source Ecosystem Integration
MicroAgent is designed to serve as the cognitive core of a broader toolchain. Future iterations will include native hooks for:
- **Vector Databases**: Integration with *ChromaDB* or *Milvus* for massive RAG (Retrieval-Augmented Generation) scaling.
- **LangChain & LlamaIndex**: Seamless interoperability with existing open-source data loaders and unstructured data pipelines.

### 3. Multi-Agent Swarm Orchestration
Scaling the state machine to support a "Swarm" topology where multiple specialized MicroAgents (e.g., a dedicated `ResearcherAgent`, a `CoderAgent`, and an `ExecutorAgent`) can negotiate and pass context asynchronously over a local message broker (like Redis or MQTT).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) running locally with the `qwen2.5-coder:7b` (or equivalent) model loaded.

### Quickstart

1. **Clone & Setup**:
   ```bash
   git clone https://github.com/shameel0505/MicroAgent.git
   cd MicroAgent
   python3 -m venv qwen_venv
   source qwen_venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch the Core**:
   ```bash
   python auto_agent.py
   ```

3. **Usage**:
   Interact directly via the CLI, or drop batch tasks into `inbox.txt` for headless automation. Watch as the model exposes its internal `<thought>` process securely in real-time.

---
*Developed with a focus on efficiency, transparency, and uncompromising reliability.*
