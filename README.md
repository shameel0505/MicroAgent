# 🔬 MicroAgent: Local ReAct Autonomous Agent Runtime

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Runtime-Ollama%20(Local)-blueviolet.svg)](https://ollama.ai/)
[![Local Inference](https://img.shields.io/badge/Model-Qwen%202.5%20Coder%207B-brightgreen.svg)](#)

A modular, local-first autonomous agent engine designed specifically to maximize the task execution reliability of **small-parameter local models (7B - 9B)** like `qwen2.5-coder:7b`. Features sandboxed execution, observation offloading, a 3-tier memory model, and autonomous office deliverable synthesis (PowerPoint & Excel).

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Objective                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 ReAct Reasoning & State Loop                │
│  - Observation Offloading (Large payloads written to disk)  │
│  - Dynamic Context Pruning (Sliding window of K=8 turns)    │
│  - Loop Breaker Heuristics (Repetition counter / Backoff)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       Tool Execution                        │
│  - Isolated Python Sandbox (Subprocess with 30s timeout)    │
│  - DuckDuckGo Web Search & Clean HTML Traversal             │
│  - Dynamic Plugin Architecture (Hot-reloadable)             │
│  - Office Automation (openpyxl, python-pptx)                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   3-Tier Memory Persistence                 │
│  1. Scratchpad (Volatile chain-of-thought)                  │
│  2. Semantic Store (Key-value persistent entity graph)      │
│  3. Episodic Ledger (Chronological audit history)           │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Key Engineering Mechanisms

### 1. Observation Offloading
Small 7B parameter models quickly experience attention degradation and hallucination loops when in-context history is flooded with massive web scrapes or raw terminal outputs.
- MicroAgent monitors observation payload size.
- Outputs exceeding `3,000` characters are automatically offloaded to `workspace/observations/` on disk.
- A concise summary pointer is fed back to the LLM prompt, keeping attention sharp while preserving full data access via file tools.

### 2. Sandboxed Subprocess Execution
Rather than invoking `exec()` or `eval()` in the main agent thread, code execution is isolated:
```python
subprocess.run(
    [sys.executable, "-c", code],
    capture_output=True,
    text=True,
    timeout=30,
    cwd="workspace"
)
```
This isolates execution inside a sandboxed workspace directory and enforces hard execution timeouts.

### 3. Context Pruning for 7B/9B Models
Local coder models lose coherence beyond 8-12 dialogue turns. MicroAgent implements a sliding-window context policy configured via `agent_config.yaml`:
- Maintains recent $K=8$ turns in raw fidelity.
- Compresses earlier reasoning history into episodic memory summaries when threshold is reached.

---

## 📊 Proven Case Study: Autonomous Deliverable Synthesis

In a single multi-step run with zero human intervention, the agent successfully researched a topic, populated a structured Excel workbook, and compiled a presentation slide deck:
- 📄 **Case Study Documentation:** [examples/autonomous_office_deliverables/](examples/autonomous_office_deliverables/)
- **Deliverables Produced:**
  - `Octopus_Research.xlsx` — Formatted data table with sources.
  - `Science_News_This_Week.pptx` — Presentation deck generated via `python-pptx`.

---

## 🛠️ Quickstart

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) running locally:
  ```bash
  ollama pull qwen2.5-coder:7b
  ```

### 2. Installation
```bash
git clone https://github.com/shameel0505/MicroAgent.git
cd MicroAgent

python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch Agent
```bash
python auto_agent.py
```
Interact via the live streaming terminal interface or provide tasks programmatically.

---

## ⚠️ Known Operational Boundaries

| Factor | Observed Threshold | Built-in Mitigation |
| :--- | :--- | :--- |
| **Max Trajectory Steps** | > 25 sequential actions | Strict `max_steps` termination cap with state summary. |
| **Repeated Action Loops** | 2 identical tool calls | Anti-repetition counter forces system-override prompt shift. |
| **Long Web Pages** | > 20,000 DOM characters | Traversed via BeautifulSoup with link/text pruning before offloading. |

---

## 📄 License
Licensed under the [MIT License](LICENSE).
