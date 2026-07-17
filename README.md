# AutoAgent 🤖

AutoAgent is a robust, autonomous AI agent framework designed to execute complex tasks using local, small-parameter Large Language Models (LLMs) like `qwen2.5-coder:7b`. It operates through a highly resilient state machine and features native tool-calling capabilities, a three-tier memory system, and strict sandboxed execution.

## 🌟 Key Features

- **Strict State Machine Workflow**: Forces the LLM into a deliberate `EVALUATE → RESEARCH → SYNTHESIS → ACTION` pipeline, effectively mitigating the hallucination and looping issues common in 7B-class models.
- **Failproof Loop Detection**: Advanced heuristics instantly detect when the model falls into an infinite loop or alternating failure pattern (e.g., getting stuck due to missing permissions), automatically injecting a "Critical System Override" to force a change in strategy.
- **Parallel Tool Execution**: Processes multiple tool calls efficiently in a single turn without overwhelming the model's context.
- **3-Tier Memory Architecture**:
  - **Scratchpad**: Short-term reasoning and summarization.
  - **Semantic Memory**: Key-value store for facts and context spanning across runs.
  - **Episodic Memory**: Long-term chronological logs.
- **Sandboxed Environment**: Python execution and bash commands run in isolated subprocesses with strict timeouts and boundaries to prevent system damage.
- **Dynamic Plugin System**: Hot-reloadable plugins in the `plugins/` directory (e.g., PowerPoint creation, PDF reading) extend the agent's capabilities without modifying core logic.

## 🛠️ Architecture

AutoAgent breaks away from the standard `ReAct` loop in favor of a deterministic XML-based state machine. Small models often lose track of their objective in standard ReAct implementations. By forcing a structured transition through well-defined states, AutoAgent drastically improves reliability and reasoning capacity.

- **Chain-of-Thought Mandate**: Every turn requires the LLM to output its reasoning inside `<thought>` tags before acting, ensuring it plans its actions logically.
- **Context Pruning**: Prevents "in-context poisoning" by stripping out redundant failure loops from the model's immediate context window.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) running locally with the `qwen2.5-coder:7b` model pulled.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AutoAgent.git
   cd AutoAgent
   ```

2. Set up the virtual environment:
   ```bash
   python3 -m venv qwen_venv
   source qwen_venv/bin/activate
   pip install requests beautifulsoup4 duckduckgo-search rich
   ```

3. Run the Agent:
   ```bash
   python auto_agent.py
   ```

### Usage
Once the interface loads, you can drop prompts directly into the terminal or place a task inside `inbox.txt` for headless execution. AutoAgent will output its thoughts and observations via a rich streaming terminal UI.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
*Built with ❤️ for autonomous local AI execution.*
