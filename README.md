# 🔬 MicroAgent: Local ReAct Autonomous Agent Runtime (v5.1)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Runtime-Ollama%20(Local)-blueviolet.svg)](https://ollama.ai/)
[![Local Inference](https://img.shields.io/badge/Model-Qwen%202.5%20Coder%207B-brightgreen.svg)](#)

A high-performance, local-first autonomous agent engine designed specifically to maximize task execution reliability on **small-parameter local models (7B - 9B)** like `qwen2.5-coder:7b` running on Apple Silicon.

### 🌟 What's New in v5.1:
- **Two-Phase Hybrid Grammar Execution (`enable_hybrid_grammar`):** Unconstrained Chain-of-Thought (`<thought>`) live-streaming for deep reasoning, followed immediately by logit-masked JSON schema dispatch (`_build_action_schema`) to eliminate syntax and tag hallucinations while keeping reasoning sharp.
- **Closed-Loop Ground-Truth Verification Gate (`enable_state_verification`):** Intercepts premature task conclusion (`<finish>`) whenever a physical mutation was requested (e.g., brightness, volume, file write) but not empirically verified by an inspection tool. Cures hallucinated completions.
- **Native macOS Hardware Controls:**
  - `set_display_brightness` / `get_display_brightness`: Native `ctypes` bindings to macOS PrivateFramework `DisplayServices.framework` (`DisplayServicesSetBrightness` / `DisplayServicesGetBrightness`).
  - `set_system_volume` / `get_system_volume`: Native CoreAudio/AppleScript volume control with verified state return.
  - `get_system_state`: Unified system telemetry (display brightness, audio volume, battery status, frontmost application).
- **Autonomous Deep Research Engine (`deep_research` / `/research`):** Decomposes topics into orthogonal sub-queries, executes concurrent multi-engine searches, scrapes high-signal pages with Trafilatura, and synthesizes publication-grade Markdown research deliverables in `workspace/`.
- **Resilient Multi-Engine Search:** Eliminates rate-limit dead ends with automatic fallback chaining: Primary DDGS $\rightarrow$ DuckDuckGo HTML Lite $\rightarrow$ Tavily/Brave APIs.
- **Trafilatura SOTA Content Extraction:** Distills main body articles, preserves structural Markdown tables, and discards cookie notices, navigation menus, and web clutter.
- **Self-Healing Diagnostic Reflex:** When execution tools fail with macOS/script error codes (e.g., AppleScript `-1728`), AutoAgent automatically queries resilient search to identify modern fixes and prevent loop hallucinations.
- **Parallel Tool Execution:** Read-only operations (`deep_research`, `search_web`, `read_url`, `read_file`, etc.) execute concurrently via `ThreadPoolExecutor`.
- **2-Tier Intent Classification:** Instant keyword fast-path + grammar LLM classifier for zero-latency chat vs. task routing.
- **Session Persistence:** Save and restore full conversations and execution history across restarts (`/session`).

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Objective                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Two-Tier Intent Gating (Chat vs Task)           │
│  - Tier 1: Zero-latency Keyword Fast-Path                   │
│  - Tier 2: Ollama Grammar-Enforced JSON Classifier          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 ReAct Reasoning & State Loop                │
│  - Observation Offloading (Large payloads written to disk)  │
│  - Grammar Tool Repair (Fallback to logit-masked schema)    │
│  - Dynamic Context Pruning (Smart summary + K=8 turns)      │
│  - Loop Breaker Heuristics (Repetition counter / Backoff)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Parallel & Sequential Tools                 │
│  - Parallel Read Pool (search_web, read_url, read_file)     │
│  - Sequential Actions (execute_python, write_file, etc.)    │
│  - Dynamic Plugin Architecture (Hot-reloadable)             │
│  - macOS Integration (Clipboard, AppleScript, Notes)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   3-Tier Memory & Session                   │
│  1. Scratchpad (Volatile chain-of-thought)                  │
│  2. Semantic Store (Auto-injected into system prompt)       │
│  3. Episodic Ledger (Chronological audit history)           │
│  4. Session Persistence (agent_session.json save/restore)   │
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
