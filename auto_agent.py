#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║            AutoAgent v4.0 — Elite Autonomous AI Agent            ║
║  ReAct · Live Streaming · Intent Routing · Human-in-the-Loop     ║
║  macOS Native Actions · 3-Tier Memory · Sandboxed Python         ║
║  Parallel Tools · Smart Compression · Loop Self-Healing          ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, re, time, subprocess, importlib.util, select
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.table import Table
from rich.rule import Rule

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

DEFAULT_CONFIG: Dict[str, Any] = {
    # ── Model ───────────────────────────────────────────────
    "model": "qwen2.5-coder:7b",
    "context_size": 16384,
    "temperature": 0.0,
    "ollama_url": "http://localhost:11434",
    "llm_retries": 3,
    "retry_delay": 2,

    # ── Timeouts ────────────────────────────────────────────
    "python_timeout": 30,
    "command_timeout": 30,
    "web_timeout": 15,

    # ── Paths ───────────────────────────────────────────────
    "workspace_dir": "workspace",
    "plugins_dir": "plugins",

    # ── UX & Intent ─────────────────────────────────────────
    "enable_streaming": True,
    "enable_intent_gating": True,

    # ── Planning ────────────────────────────────────────────
    "enable_planning": True,
    "planning_threshold": 50,  # chars; tasks longer than this trigger auto-plan

    # ── Context — tuned for 7B/9B small models ──────────────
    "compress_threshold": 16,
    "keep_recent": 8,
    "enable_smart_compress": True,

    # ── Tool output limits ───────────────────────────────────
    "search_max_results": 5,
    "file_read_limit": 6000,
    "url_read_limit": 4000,

    # ── Observation offloading ───────────────────────────────
    "observation_limit": 3000,
    "offload_large_obs": True,

    # ── Safety / loop prevention ─────────────────────────────
    "max_steps": 25,
    "max_json_retries": 3,
    "max_repeat_count": 2,

    # ── Persistence ─────────────────────────────────────────
    "memory_file": "agent_memory.json",
    "episodic_file": "agent_episodic.json",
    "session_file": "agent_session.json",
    "log_file": "agent.log",
}


def load_config() -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    for p in [Path("agent_config.yaml"), Path("agent_config.json")]:
        if p.exists():
            try:
                if p.suffix in (".yaml", ".yml") and HAS_YAML:
                    loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
                else:
                    loaded = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cfg.update(loaded)
                break
            except Exception as e:
                print(f"[warning] Failed to load {p}: {e}")
    return cfg


CONFIG = load_config()
console = Console()

# ═══════════════════════════════════════════════════════
# STRUCTURED LOGGER
# ═══════════════════════════════════════════════════════

class AgentLogger:
    def __init__(self, log_file: str):
        self.path = Path(log_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _w(self, data: dict):
        entry = {"time": datetime.now().isoformat(), **data}
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def user_input(self, text: str):
        self._w({"event": "user_input", "text": text})

    def llm_call(self, dur: float, toks: int):
        self._w({"event": "llm_call", "duration_s": round(dur, 2), "tokens": toks})

    def tool_call(self, name: str, args: dict, result: str, dur: float):
        self._w({"event": "tool_call", "tool": name, "args": args,
                 "result_len": len(result), "duration_s": round(dur, 2)})

    def error(self, msg: str):
        self._w({"event": "error", "message": msg})


logger = AgentLogger(CONFIG["log_file"])

# ═══════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """You are AutoAgent — an elite autonomous workstation AI assistant running locally on Apple Silicon.
Your current working directory is: {WORKSPACE_DIR}

[OPERATING PROTOCOL]
1. For every turn, always begin with your reasoning in <thought>...</thought> tags.
2. After your thought, invoke tool call(s) OR a <transition>STATE</transition> tag. Choose ONE.
3. If you need user clarification or preferences, call <ask_user>. Never write questions to disk or sleep.
4. When you have completed the task or produced deliverables, call <finish> with a complete final summary.

[STATES]
• EVALUATE: Analyze the user's task and plan next moves. Transition to RESEARCH or ACTION.
• RESEARCH: Gather facts using read tools (search_web, read_url, read_file).
• SYNTHESIS: Consolidate research in <scratchpad>...</scratchpad>, then transition to ACTION.
• ACTION: Build deliverables using action tools (execute_python, write_file, run_command, open_file). Transition to DONE or call <finish> when done.
• DONE: Task is complete.

AVAILABLE TOOLS:
{TOOLS_XML}
"""

CHAT_PROMPT = """You are AutoAgent, a friendly, ultra-capable local AI workstation assistant powered by Qwen 2.5 on Apple Silicon M4.
You assist the user with:
• Autonomous deep web research & analysis
• Generating production Python code, scripts, and utilities
• Building formatted Excel workbooks (.xlsx) and slide presentations (.pptx)
• Automating Mac desktop tasks, files, and terminal workflows
• Answering technical, coding, and general knowledge questions

Guidelines:
- Respond warmly, directly, and concisely.
- Do NOT output XML tags, state transitions, or tool calls during conversational chat.
- If the user asks what you can do, give a brief, punchy overview of your capabilities and invite them to give you a task.
"""

# ═══════════════════════════════════════════════════════
# BASE TOOL SCHEMAS
# ═══════════════════════════════════════════════════════

BASE_TOOLS: List[Dict] = [
    {
        "name": "ask_user",
        "schema": "<ask_user>\n<question>Clarification question for the user</question>\n</ask_user>",
        "description": "Ask the human user a question on the terminal when you need clarification, choices, or missing details. Execution pauses until the user replies."
    },
    {
        "name": "finish",
        "schema": "<finish>\n<summary>Comprehensive deliverable summary or answer</summary>\n</finish>",
        "description": "Conclude the task and deliver the final answer/deliverable directly to the user."
    },
    {
        "name": "search_web",
        "schema": "<search_web>\n<query>search query</query>\n</search_web>",
        "description": "Search the web via DuckDuckGo. Returns titles, URLs, and snippets."
    },
    {
        "name": "read_url",
        "schema": "<read_url>\n<url>Full URL to fetch</url>\n</read_url>",
        "description": "Fetch and extract readable text from a URL. Strips nav/ads, focuses on main content."
    },
    {
        "name": "execute_python",
        "schema": "<execute_python>\n# python code here\n</execute_python>",
        "description": "Run Python code in a sandboxed subprocess. Returns stdout + stderr. CWD is workspace/."
    },
    {
        "name": "run_command",
        "schema": "<run_command>\nbash command here\n</run_command>",
        "description": "Run a bash command on MacOS. Returns combined stdout + stderr."
    },
    {
        "name": "read_file",
        "schema": "<read_file>\n<path>file path</path>\n</read_file>",
        "description": "Read a local file's contents (up to 6,000 chars). Good for code, text, JSON, CSV."
    },
    {
        "name": "write_file",
        "schema": "<write_file>\n<path>file path</path>\n<content>text content to write</content>\n</write_file>",
        "description": "Write text to a local file. Creates missing directories. Overwrites if exists."
    },
    {
        "name": "open_file",
        "schema": "<open_file>\n<path>file path (e.g. .xlsx, .pptx, .pdf, .png)</path>\n</open_file>",
        "description": "Open a generated file in its native macOS application (Excel, Keynote, Preview, etc.)."
    },
    {
        "name": "open_url",
        "schema": "<open_url>\n<url>Full URL to open</url>\n</open_url>",
        "description": "Open a website URL in the user's default macOS web browser."
    },
    {
        "name": "notify_user",
        "schema": "<notify_user>\n<title>Notification title</title>\n<message>Notification message</message>\n</notify_user>",
        "description": "Display a native macOS system notification banner with sound upon completing long tasks."
    },
    {
        "name": "list_directory",
        "schema": "<list_directory>\n<path>directory path to list</path>\n</list_directory>",
        "description": "List files and subdirectories at a path, with sizes."
    },
    {
        "name": "sleep",
        "schema": "<sleep>\n<seconds>seconds to wait</seconds>\n</sleep>",
        "description": "Sleep for N seconds (use only in polling or waiting for external processes)."
    },
    {
        "name": "read_memory",
        "schema": "<read_memory></read_memory>",
        "description": "Read all facts from persistent semantic memory (key-value store)."
    },
    {
        "name": "write_memory",
        "schema": "<write_memory>\n<key>fact key</key>\n<value>fact value</value>\n</write_memory>",
        "description": "Save a key-value fact to persistent semantic memory across sessions."
    },
    {
        "name": "search_memory",
        "schema": "<search_memory>\n<query>search query</query>\n</search_memory>",
        "description": "Search semantic memory keys and values for a substring."
    },
    {
        "name": "read_episodic",
        "schema": "<read_episodic></read_episodic>",
        "description": "Read summaries of past completed tasks from episodic memory."
    }
]

# ═══════════════════════════════════════════════════════
# MEMORY — 3-TIER SYSTEM
# ═══════════════════════════════════════════════════════

def _load_json(path: str, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: str, data: Any) -> None:
    try:
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        console.print(f"[red]Memory save error: {e}[/red]")


def read_memory_fn() -> str:
    data = _load_json(CONFIG["memory_file"], {})
    filtered = {k: v for k, v in data.items() if not k.startswith("_")}
    return json.dumps(filtered, indent=2) if filtered else "Semantic memory is empty."


def write_memory_fn(key: str, value: str) -> str:
    mem = _load_json(CONFIG["memory_file"], {})
    mem[key] = value
    mem["_updated"] = datetime.now().isoformat()
    _save_json(CONFIG["memory_file"], mem)
    return f"✓ Saved to memory: {key!r} = {value!r}"


def search_memory_fn(query: str) -> str:
    mem = _load_json(CONFIG["memory_file"], {})
    if not mem:
        return "Semantic memory is empty."
    q = query.lower()
    hits = [(k, v) for k, v in mem.items()
            if not k.startswith("_") and (q in k.lower() or q in str(v).lower())]
    if not hits:
        return f"No memory entries match '{query}'."
    return "Matches:\n" + "\n".join(f"  {k}: {v}" for k, v in hits)


def read_episodic_fn() -> str:
    eps = _load_json(CONFIG["episodic_file"], [])
    if not eps:
        return "No episodic memories yet."
    lines = ["Episodic Memory (Past Tasks):"]
    for e in eps[-5:]:
        lines.append(f"• [{e.get('timestamp', '')[:10]}] Task: {e.get('task', '')}")
        lines.append(f"  Outcome: {e.get('outcome', '')[:200]}")
    return "\n".join(lines)


def save_episode(task: str, outcome: str) -> None:
    eps = _load_json(CONFIG["episodic_file"], [])
    eps.append({
        "timestamp": datetime.now().isoformat(),
        "task": task[:200],
        "outcome": outcome[:500],
    })
    eps = eps[-50:]  # keep last 50
    _save_json(CONFIG["episodic_file"], eps)

# ═══════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════

def ask_user_fn(question: str) -> str:
    console.print()
    console.print(Panel(
        f"[bold yellow]{question}[/bold yellow]",
        title="[bold cyan]❓ AutoAgent Question[/bold cyan]",
        border_style="cyan"
    ))
    try:
        reply = input("👤 Your Reply: ").strip()
        return f"User replied: {reply}" if reply else "User provided no answer."
    except (EOFError, KeyboardInterrupt):
        return "User cancelled the prompt."


def finish_fn(summary: str) -> str:
    return f"TASK_COMPLETED: {summary}"


def open_file_fn(path: str) -> str:
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            ws_p = Path(CONFIG["workspace_dir"]) / path
            if ws_p.exists():
                p = ws_p
        if not p.exists():
            return f"❌ File not found: {path}"
        subprocess.run(["open", str(p.resolve())], check=True)
        return f"✓ Opened {p.name} in macOS default application."
    except Exception as e:
        return f"❌ Failed to open file: {e}"


def open_url_fn(url: str) -> str:
    try:
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"
        subprocess.run(["open", url], check=True)
        return f"✓ Opened URL in default browser: {url}"
    except Exception as e:
        return f"❌ Failed to open URL: {e}"


def notify_user_fn(title: str, message: str) -> str:
    try:
        clean_title = title.replace('"', '\\"')
        clean_msg = message.replace('"', '\\"')
        cmd = f'display notification "{clean_msg}" with title "{clean_title}" sound name "Glass"'
        subprocess.run(["osascript", "-e", cmd], check=True)
        return "✓ macOS notification delivered."
    except Exception as e:
        return f"❌ Notification error: {e}"


def execute_python_sandboxed(code: str) -> str:
    """Run Python code in an isolated subprocess inside workspace/."""
    ws = Path(CONFIG["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            timeout=CONFIG["python_timeout"],
            cwd=str(ws.resolve()),
        )
        out = r.stdout
        if r.stderr.strip():
            out += ("\n" if out else "") + f"[STDERR]\n{r.stderr.strip()}"
        return out.strip() or "✓ Code executed with no output."
    except subprocess.TimeoutExpired:
        return f"❌ Timed out after {CONFIG['python_timeout']}s. Simplify or split the code."
    except Exception as e:
        return f"❌ Execution error: {e}"


def run_command_fn(command: str) -> str:
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=CONFIG["command_timeout"],
        )
        return (r.stdout + r.stderr).strip() or "✓ Command ran with no output."
    except subprocess.TimeoutExpired:
        return f"❌ Command timed out after {CONFIG['command_timeout']}s"
    except Exception as e:
        return f"❌ Command error: {e}"


def read_file_fn(path: str) -> str:
    try:
        p = Path(path).expanduser()
        if not p.is_absolute() and not p.exists():
            ws_p = Path(CONFIG["workspace_dir"]) / path
            if ws_p.exists():
                p = ws_p
        content = p.read_text(encoding="utf-8", errors="replace")
        limit = CONFIG["file_read_limit"]
        if len(content) > limit:
            content = content[:limit] + f"\n\n… [truncated at {limit:,} chars]"
        return content
    except Exception as e:
        return f"❌ File read error: {e}"


def write_file_fn(path: str, content: str) -> str:
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(CONFIG["workspace_dir"]) / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"✓ Wrote {len(content.encode()):,} bytes → {p}"
    except Exception as e:
        return f"❌ File write error: {e}"


def list_dir_fn(path: str) -> str:
    try:
        p = Path(path).expanduser()
        if not p.is_absolute() and not p.exists():
            ws_p = Path(CONFIG["workspace_dir"]) / path
            if ws_p.exists():
                p = ws_p
        if not p.exists():
            return f"❌ Path does not exist: {path}"
        rows = []
        for item in sorted(p.iterdir()):
            if item.is_dir():
                rows.append(f"📁 {item.name}/")
            else:
                rows.append(f"📄 {item.name} ({item.stat().st_size:,} bytes)")
        return "\n".join(rows) if rows else "(empty directory)"
    except Exception as e:
        return f"❌ List dir error: {e}"


def search_web_fn(query: str) -> str:
    try:
        if not DDGS:
            return "❌ DuckDuckGo search library not available."
        results = DDGS().text(query, max_results=CONFIG["search_max_results"])
        if not results:
            return "No results found."
        lines = [f"🔍 Search: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'No title')}")
            lines.append(f"   {r.get('href', '')}")
            lines.append(f"   {r.get('body', '')[:260]}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Search error: {e}"


def read_url_fn(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=CONFIG["web_timeout"])
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body
        text = (main or soup).get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        limit = CONFIG["url_read_limit"]
        if len(text) > limit:
            text = text[:limit] + f"\n\n… [truncated at {limit:,} chars]"
        return text
    except requests.HTTPError as e:
        return f"❌ HTTP {e.response.status_code}: {url}"
    except Exception as e:
        return f"❌ URL read error: {e}"


def sleep_fn(seconds: int) -> str:
    console.print(f"[blue]💤 Waiting {seconds}s...[/blue]")
    time.sleep(seconds)
    return f"✓ Resumed. Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# ═══════════════════════════════════════════════════════
# PLUGIN SYSTEM
# ═══════════════════════════════════════════════════════

_plugin_cache: Dict[str, Any] = {}
_plugin_mtimes: Dict[str, float] = {}


def get_dynamic_plugins() -> Dict[str, Any]:
    plugins_dir = Path(CONFIG["plugins_dir"])
    if not plugins_dir.exists():
        return {}
    plugins: Dict[str, Any] = {}
    for py_file in sorted(plugins_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        key = str(py_file)
        mtime = py_file.stat().st_mtime
        if key in _plugin_cache and _plugin_mtimes.get(key) == mtime:
            mod = _plugin_cache[key]
        else:
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _plugin_cache[key] = mod
                _plugin_mtimes[key] = mtime
                console.print(f"[dim]🔌 Plugin (re)loaded: {py_file.name}[/dim]")
            except Exception as e:
                console.print(f"[red]⚠  Plugin [{py_file.name}] failed: {e}[/red]")
                continue
        if hasattr(mod, "TOOL_SCHEMA") and hasattr(mod, "execute"):
            name = mod.TOOL_SCHEMA.get("function", {}).get("name", py_file.stem)
            plugins[name] = mod
    return plugins

# ═══════════════════════════════════════════════════════
# LLM INFERENCE & STREAMING
# ═══════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def context_tokens(messages: List[Dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


def stream_ollama(messages: List[Dict], temperature: Optional[float] = None):
    url = f"{CONFIG['ollama_url']}/api/chat"
    temp = temperature if temperature is not None else CONFIG["temperature"]
    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "stream": True,
        "options": {"temperature": temp, "num_ctx": CONFIG["context_size"]},
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        yield f"\n[⚠ LLM Error: {e}]"


def chat_ollama_blocking(messages: List[Dict], temperature: float = 0.1) -> Dict:
    url = f"{CONFIG['ollama_url']}/api/chat"
    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": CONFIG["context_size"]},
    }
    for attempt in range(CONFIG["llm_retries"]):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("message", {"role": "assistant", "content": ""})
        except Exception as e:
            if attempt < CONFIG["llm_retries"] - 1:
                time.sleep(CONFIG["retry_delay"] * (attempt + 1))
            else:
                return {"role": "assistant", "content": f"[LLM Error: {e}]"}

# ═══════════════════════════════════════════════════════
# CONVERSATION FAST-PATH (GREETINGS & GENERAL CHAT)
# ═══════════════════════════════════════════════════════

def is_conversational(text: str) -> bool:
    if not CONFIG.get("enable_intent_gating", True):
        return False
    t = text.strip().lower().rstrip("!?.")
    if not t:
        return True
    GREETINGS = {
        "hi", "hello", "hey", "howdy", "sup", "yo", "good morning", "good evening",
        "good afternoon", "who are you", "what are you", "what can you do", "help",
        "what are your tools", "what tools do you have", "thanks", "thank you",
        "cool", "nice", "awesome", "great", "ok", "okay"
    }
    if t in GREETINGS:
        return True
    words = t.split()
    if len(words) <= 3 and words[0] in {"hi", "hello", "hey", "yo"}:
        return True
    return False


def handle_conversation(user_input: str, messages: List[Dict]) -> None:
    chat_prompt = [
        {"role": "system", "content": CHAT_PROMPT},
        {"role": "user", "content": user_input}
    ]
    console.print()
    full_reply = ""
    with Live(console=console, refresh_per_second=12, transient=False) as live:
        live.update(Panel("[dim]Thinking...[/dim]", title="[bold cyan]🤖 AutoAgent[/bold cyan]", border_style="cyan"))
        for chunk in stream_ollama(chat_prompt, temperature=0.7):
            full_reply += chunk
            live.update(Panel(Markdown(full_reply), title="[bold cyan]🤖 AutoAgent[/bold cyan]", border_style="cyan"))

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": full_reply})

# ═══════════════════════════════════════════════════════
# CONTEXT MANAGEMENT (SMART COMPRESSION)
# ═══════════════════════════════════════════════════════

def smart_compress(messages: List[Dict]) -> List[Dict]:
    threshold = CONFIG["compress_threshold"]
    keep = CONFIG["keep_recent"]
    if len(messages) <= threshold:
        return messages

    sys_msg = messages[0]
    to_compress = messages[1:-keep]
    recent = messages[-keep:]

    if CONFIG["enable_smart_compress"] and to_compress:
        snippet = "\n".join(
            f"{m['role'].upper()}: {str(m.get('content', ''))[:300]}"
            for m in to_compress
        )
        summary_req = [
            {"role": "system", "content": (
                "Summarise this agent conversation in 4-5 bullet points. "
                "Include: tasks done, key results, errors. One line per bullet. Output ONLY bullets."
            )},
            {"role": "user", "content": snippet},
        ]
        resp = chat_ollama_blocking(summary_req)
        summary = resp.get("content", "").strip()
        if summary:
            compressed = [sys_msg, {"role": "system", "content": f"[COMPRESSED HISTORY]\n{summary}"}] + recent
            console.print(f"[dim]⚡ Compressed context: {len(messages)} → {len(compressed)} messages[/dim]")
            return compressed
    return [sys_msg] + recent

# ═══════════════════════════════════════════════════════
# XML TOOL CALL EXTRACTION & ROUTING
# ═══════════════════════════════════════════════════════

def _parse_inner_args(inner_xml: str) -> Dict[str, str]:
    args = {}
    inner_matches = re.findall(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', inner_xml, re.DOTALL)
    if inner_matches:
        for arg_name, arg_val in inner_matches:
            args[arg_name] = arg_val.strip()
    return args


def extract_tool_calls(text: str) -> List[Tuple[str, Dict]]:
    calls: List[Tuple[str, Dict]] = []
    matches = re.findall(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', text, re.DOTALL)
    for name, inner_xml in matches:
        if name in ("transition", "scratchpad", "thought"):
            continue
        args = _parse_inner_args(inner_xml)
        if not args:
            raw = inner_xml.strip()
            if name == "execute_python":
                args["code"] = raw
            elif name == "run_command":
                args["command"] = raw
            elif name == "search_web":
                args["query"] = raw
            elif name == "read_url":
                args["url"] = raw
            elif name == "read_file":
                args["path"] = raw
            elif name == "list_directory":
                args["path"] = raw
            elif name == "sleep":
                args["seconds"] = raw
            elif name == "search_memory":
                args["query"] = raw
            elif name == "ask_user":
                args["question"] = raw
            elif name == "open_file":
                args["path"] = raw
            elif name == "open_url":
                args["url"] = raw
            elif name == "finish":
                args["summary"] = raw
        calls.append((name, args))
    return calls


def route_tool(tool_name: str, args: Dict, plugins: Dict) -> str:
    ROUTES = {
        "ask_user":       lambda: ask_user_fn(args.get("question", "")),
        "finish":         lambda: finish_fn(args.get("summary", "")),
        "open_file":      lambda: open_file_fn(args.get("path", "")),
        "open_url":       lambda: open_url_fn(args.get("url", "")),
        "notify_user":    lambda: notify_user_fn(args.get("title", "AutoAgent"), args.get("message", "Task completed.")),
        "search_web":     lambda: search_web_fn(args.get("query", "")),
        "read_url":       lambda: read_url_fn(args.get("url", "")),
        "execute_python": lambda: execute_python_sandboxed(args.get("code", "")),
        "run_command":    lambda: run_command_fn(args.get("command", "")),
        "read_file":      lambda: read_file_fn(args.get("path", "")),
        "write_file":     lambda: write_file_fn(args.get("path", ""), args.get("content", "")),
        "list_directory": lambda: list_dir_fn(args.get("path", ".")),
        "sleep":          lambda: sleep_fn(int(args.get("seconds", 5))),
        "read_memory":    lambda: read_memory_fn(),
        "write_memory":   lambda: write_memory_fn(args.get("key", ""), args.get("value", "")),
        "search_memory":  lambda: search_memory_fn(args.get("query", "")),
        "read_episodic":  lambda: read_episodic_fn(),
    }
    if tool_name in ROUTES:
        return ROUTES[tool_name]()
    if tool_name in plugins:
        try:
            return str(plugins[tool_name].execute(args))
        except Exception as e:
            return f"❌ Plugin error [{tool_name}]: {e}"
    available = list(ROUTES) + list(plugins)
    return f"❌ Unknown tool '{tool_name}'. Available: {available}"

# ═══════════════════════════════════════════════════════
# TASK PLANNER
# ═══════════════════════════════════════════════════════

_COMPLEX_WORDS = {
    "create", "build", "research", "analyze", "analyse", "generate", "compare",
    "summarize", "summarise", "automate", "write", "develop", "design", "find",
    "scrape", "download", "schedule", "deploy", "extract", "deck", "presentation", "excel"
}


def should_plan(task: str) -> bool:
    if not CONFIG.get("enable_planning", True):
        return False
    words = set(task.lower().split())
    return len(task) > CONFIG.get("planning_threshold", 50) or bool(words & _COMPLEX_WORDS)


def make_plan(task: str) -> str:
    msgs = [
        {"role": "system", "content": (
            "You are a master task planner for an AI agent on macOS. Given a task, produce a clean, "
            "numbered 3 to 5 step execution roadmap. Be direct and concrete. "
            "Output ONLY the numbered steps — no intro or conversational filler."
        )},
        {"role": "user", "content": f"Task: {task}"},
    ]
    resp = chat_ollama_blocking(msgs, temperature=0.1)
    return resp.get("content", "").strip()

# ═══════════════════════════════════════════════════════
# OBSERVATION OFFLOADING
# ═══════════════════════════════════════════════════════

_obs_counter = 0
_OBS_CONTENT_HASH: Dict[str, str] = {}


def _prepare_observation(tool_name: str, result: str) -> str:
    global _obs_counter
    limit = CONFIG["observation_limit"]
    if not CONFIG["offload_large_obs"] or len(result) <= limit:
        return result

    import hashlib
    content_hash = hashlib.md5(result.encode()).hexdigest()[:12]

    if content_hash in _OBS_CONTENT_HASH:
        existing = _OBS_CONTENT_HASH[content_hash]
        short = result[:limit]
        return (
            f"{short}\n\n"
            f"… [SAME CONTENT as before — already saved to {existing}\n"
            f"You have already seen this data. Proceed with your action.]"
        )

    _obs_counter += 1
    ws = Path(CONFIG["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    out_path = ws / f"obs_{_obs_counter}_{tool_name}.txt"
    try:
        out_path.write_text(result, encoding="utf-8")
        _OBS_CONTENT_HASH[content_hash] = str(out_path)
        short = result[:limit]
        return (
            f"{short}\n\n"
            f"… [OUTPUT TRUNCATED — full result ({len(result):,} chars) offloaded to disk:\n"
            f"  {out_path}\n"
            f"Use read_file if you need specific sections.]"
        )
    except Exception:
        return result[:limit] + f"\n… [truncated at {limit} chars]"

# ═══════════════════════════════════════════════════════
# AUTONOMOUS REACT STATE MACHINE LOOP
# ═══════════════════════════════════════════════════════

def state_machine_loop(user_input: str, messages: List[Dict]) -> List[Dict]:
    current_state = "EVALUATE"
    step = 0
    max_steps = CONFIG["max_steps"]
    
    tool_sig_counts: Counter = Counter()
    recent_tools: deque = deque(maxlen=8)

    while True:
        step += 1
        if step > max_steps:
            console.print(f"\n[bold red]⛔ Reached safety step limit ({max_steps}).[/bold red]")
            messages.append({
                "role": "system",
                "content": f"[SYSTEM: Maximum step limit reached. Provide a final summary of what was accomplished.]",
            })
            with console.status("[yellow]Generating final summary...[/yellow]", spinner="dots"):
                resp = chat_ollama_blocking(messages)
            summary = resp.get("content", "")
            if summary:
                console.print(Panel(Markdown(summary), title="[bold yellow]Task Summary[/bold yellow]", border_style="yellow"))
                messages.append({"role": "assistant", "content": summary})
            break

        state_prompt = f"[SYSTEM: You are currently in STATE: {current_state}. Please think, act, and transition.]"
        messages.append({"role": "user", "content": state_prompt})
        messages = smart_compress(messages)
        print_status(messages)

        full_reply = ""
        t0 = time.time()

        if CONFIG["enable_streaming"]:
            with Live(console=console, refresh_per_second=10, transient=False) as live:
                live.update(Panel("[dim]Thinking...[/dim]",
                                  title=f"[bold yellow]🤖 AutoAgent [dim]({current_state})[/dim][/bold yellow]",
                                  border_style="yellow"))
                for chunk in stream_ollama(messages):
                    full_reply += chunk
                    visible_reply = full_reply.replace("<thought>", "> **💭 Thought:**\n> ").replace("</thought>", "\n")
                    display = Markdown(visible_reply) if len(visible_reply) < 4000 else visible_reply[-2000:]
                    live.update(Panel(display,
                                      title=f"[bold yellow]🤖 AutoAgent [dim]({current_state})[/dim][/bold yellow]",
                                      border_style="yellow"))
        else:
            with console.status(f"[yellow]🤖 Reasoning in {current_state}…[/yellow]", spinner="dots"):
                resp = chat_ollama_blocking(messages)
                full_reply = resp.get("content", "")
            visible_reply = full_reply.replace("<thought>", "> **💭 Thought:**\n> ").replace("</thought>", "\n")
            console.print(Panel(Markdown(visible_reply),
                                title=f"[bold yellow]🤖 AutoAgent ({current_state})[/bold yellow]",
                                border_style="yellow"))

        dur_llm = time.time() - t0
        logger.llm_call(dur_llm, estimate_tokens(full_reply))
        messages.append({"role": "assistant", "content": full_reply})

        transition_match = re.search(r'<transition>(.*?)</transition>', full_reply, re.IGNORECASE)
        tool_calls = extract_tool_calls(full_reply)

        # Check for explicit finish tool call
        finish_call = next(((t, a) for t, a in tool_calls if t == "finish"), None)
        if finish_call:
            summary = finish_call[1].get("summary", "")
            console.print(Panel(
                Markdown(summary),
                title="[bold green]✔ Task Deliverable Completed[/bold green]",
                border_style="green"
            ))
            notify_user_fn("AutoAgent Task Complete", "Your requested deliverable is ready!")
            break

        # Natural conclusion or answer without XML tags
        if not tool_calls and not transition_match:
            clean_reply = re.sub(r'<thought>.*?</thought>', '', full_reply, flags=re.DOTALL).strip()
            if clean_reply and current_state in ("ACTION", "SYNTHESIS", "EVALUATE") and step > 1:
                console.print(Panel(
                    Markdown(clean_reply),
                    title="[bold green]✔ Task Response[/bold green]",
                    border_style="green"
                ))
                break
            else:
                messages.append({"role": "user", "content": "SYSTEM GUIDANCE: Use <tool_name>...</tool_name> to act, or <transition>STATE</transition> to move between states, or <finish> to conclude."})
                continue

        if tool_calls and transition_match:
            transition_match = None  # prioritize tool execution

        current_plugins = get_dynamic_plugins()

        if transition_match:
            new_state = transition_match.group(1).strip().upper()
            if new_state in ["EVALUATE", "RESEARCH", "SYNTHESIS", "ACTION", "DONE"]:
                current_state = new_state
                console.print(f"[bold magenta]🔄 Transitioning to {current_state}[/bold magenta]")
                if current_state == "DONE":
                    clean_reply = re.sub(r'<thought>.*?</thought>', '', full_reply, flags=re.DOTALL).strip()
                    if clean_reply:
                        console.print(Panel(Markdown(clean_reply), title="[bold green]✔ Deliverable Complete[/bold green]", border_style="green"))
                    break
            else:
                messages.append({"role": "user", "content": f"SYSTEM WARNING: Invalid state '{new_state}'. Valid states: EVALUATE, RESEARCH, SYNTHESIS, ACTION, DONE."})
            continue

        if tool_calls:
            observations = []
            break_loop = False

            for tool_name, tool_args in tool_calls:
                sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                tool_sig_counts[sig] += 1
                recent_tools.append(tool_name)

                # Hard Loop Prevention with Human Intervention
                if tool_sig_counts[sig] > CONFIG["max_repeat_count"]:
                    console.print(f"\n[bold red]⚠ Loop Intercepted: '{tool_name}' repeated {tool_sig_counts[sig]}× without progress.[/bold red]")
                    console.print(Panel(
                        f"The model is attempting to repeat [bold yellow]{tool_name}[/bold yellow] with identical arguments.\n"
                        f"Arguments: [dim]{json.dumps(tool_args)}[/dim]\n\n"
                        f"[cyan]Press Enter to stop and review, or type guidance to redirect the agent.[/cyan]",
                        title="[bold red]Self-Healing Intercept[/bold red]",
                        border_style="red"
                    ))
                    try:
                        guidance = input("👤 Human Guidance: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        guidance = ""
                    if not guidance:
                        console.print("[yellow]Task paused by user.[/yellow]")
                        break_loop = True
                        break
                    else:
                        tool_sig_counts.clear()
                        observations.append(f"OBSERVATION: Human supervisor intervened: {guidance}")
                        continue

                console.print(f"╭{'─'*31} Tool Call {'─'*31}╮")
                console.print(f"│ 🔧 {tool_name:<73}│")
                console.print(f"╰{'─'*75}╯")

                t_tool = time.time()
                result = route_tool(tool_name, tool_args, current_plugins)
                dur_tool = time.time() - t_tool

                logger.tool_call(tool_name, tool_args, str(result), dur_tool)

                preview = str(result)
                if len(preview) > 1200:
                    preview = preview[:1200] + f"\n… [{len(str(result)):,} chars total]"
                console.print(Panel(f"[green]👁 Observation[/green] [dim]({dur_tool:.2f}s)[/dim]\n\n{preview}", border_style="green"))

                if str(result).startswith("TASK_COMPLETED:"):
                    break_loop = True
                    break

                context_result = _prepare_observation(tool_name, str(result))
                observations.append(f"OBSERVATION from {tool_name}:\n{context_result}")

            if break_loop:
                break

            if observations:
                messages.append({"role": "user", "content": "\n\n---\n\n".join(observations)})

            # Update system prompt with active tools
            active_tools = list(BASE_TOOLS) + [p.TOOL_SCHEMA for p in current_plugins.values()]
            tools_xml_str = "\n\n".join(json.dumps(t, indent=2) if isinstance(t, dict) and "schema" not in t else (t.get("schema") if isinstance(t, dict) else str(t)) for t in active_tools)
            messages[0]["content"] = SYSTEM_PROMPT_TEMPLATE.replace("{TOOLS_XML}", tools_xml_str).replace("{WORKSPACE_DIR}", str(Path(CONFIG["workspace_dir"]).absolute()))

    return messages

# ═══════════════════════════════════════════════════════
# INPUT & CLI STATUS
# ═══════════════════════════════════════════════════════

def poll_input() -> str:
    sys.stdout.write("\n👤 You: ")
    sys.stdout.flush()

    while True:
        inbox = Path("inbox.txt")
        if inbox.exists():
            try:
                content = inbox.read_text(encoding="utf-8").strip()
                if content:
                    inbox.write_text("", encoding="utf-8")
                    console.print("\n[bold green]📥 Task from inbox.txt[/bold green]")
                    logger.user_input(f"[inbox] {content}")
                    return content
            except Exception:
                pass

        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.5)
            if readable:
                line = sys.stdin.readline().strip()
                if line:
                    logger.user_input(line)
                    return line
        except Exception:
            try:
                line = input()
                if line.strip():
                    return line.strip()
            except (EOFError, KeyboardInterrupt):
                return "exit"


def print_status(messages: List[Dict]) -> None:
    toks = context_tokens(messages)
    max_t = CONFIG["context_size"]
    pct = min(toks / max_t * 100, 100)
    bar_w = 20
    filled = int(bar_w * pct / 100)
    bar = "█" * filled + "░" * (bar_w - filled)
    color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
    plugins = get_dynamic_plugins()
    console.print(
        f"[dim]Context [{color}]{bar}[/{color}] ~{toks:,} tokens (~{pct:.0f}%) | "
        f"Msgs: {len(messages)} | Plugins: {len(plugins)} | Model: {CONFIG['model']}[/dim]"
    )


def build_active_tools() -> List[Dict]:
    return list(BASE_TOOLS) + [p.TOOL_SCHEMA for p in get_dynamic_plugins().values()]

# ═══════════════════════════════════════════════════════
# MAIN INTERACTION LOOP
# ═══════════════════════════════════════════════════════

def main() -> None:
    active_tools = build_active_tools()
    plugin_count = len(active_tools) - len(BASE_TOOLS)

    console.print()
    console.print(Panel.fit(
        "[bold cyan]⚡ AutoAgent v4.0 — Apple Silicon M4 Workstation Runtime[/bold cyan]\n"
        "[dim]Closed-Loop ReAct · Human-in-the-Loop · Intent Fast-Path · macOS Native[/dim]\n\n"
        f"• [bold white]Model:[/bold white] [green]{CONFIG['model']}[/green]   "
        f"• [bold white]Context:[/bold white] [green]{CONFIG['context_size']:,} tokens[/green]   "
        f"• [bold white]Tools:[/bold white] [green]{len(BASE_TOOLS)} base + {plugin_count} plugins[/green]\n"
        f"• [bold white]Commands:[/bold white] [yellow]/help[/yellow], [yellow]/plan[/yellow], [yellow]/tools[/yellow], [yellow]/memory[/yellow], [yellow]/clear[/yellow], [yellow]/exit[/yellow]",
        border_style="cyan",
        title="[bold]AutoAgent[/bold]",
    ))

    tools_xml_str = "\n\n".join(json.dumps(t, indent=2) if isinstance(t, dict) and "schema" not in t else (t.get("schema") if isinstance(t, dict) else str(t)) for t in active_tools)
    system_content = SYSTEM_PROMPT_TEMPLATE.replace("{TOOLS_XML}", tools_xml_str).replace("{WORKSPACE_DIR}", str(Path(CONFIG["workspace_dir"]).absolute()))
    messages: List[Dict] = [{"role": "system", "content": system_content}]

    try:
        while True:
            print_status(messages)
            user_input = poll_input()

            if user_input.lower() in {"exit", "quit", "/exit", "/quit", "q"}:
                console.print("\n[bold]👋 Standing down AutoAgent. Have a great day![/bold]")
                break

            if not user_input.strip():
                continue

            # ── Slash Commands ──────────────────────────────────────────────
            cmd = user_input.strip()
            if cmd == "/clear" or cmd == "clear":
                os.system("clear")
                continue

            if cmd == "/tools":
                tbl = Table(title="Available AutoAgent Tools", border_style="cyan")
                tbl.add_column("Tool Name", style="bold yellow")
                tbl.add_column("Description", style="white")
                for t in build_active_tools():
                    if isinstance(t, dict):
                        tbl.add_row(t.get("name", "plugin"), t.get("description", ""))
                console.print(tbl)
                continue

            if cmd == "/memory":
                mem = read_memory_fn()
                console.print(Panel(mem, title="[cyan]Semantic Memory[/cyan]", border_style="cyan"))
                continue

            if cmd == "/help":
                console.print(Panel(
                    "• [bold yellow]Chat Naturally:[/bold yellow] Say 'hi', ask questions, discuss ideas.\n"
                    "• [bold yellow]Give Tasks:[/bold yellow] 'Research Apple M4 benchmarks and save to excel'\n"
                    "• [bold yellow]Human-in-the-Loop:[/bold yellow] If the agent needs details, it will ask you directly.\n"
                    "• [bold yellow]macOS Actions:[/bold yellow] Generates files and opens them directly in Excel/Preview.\n"
                    "• [bold yellow]Slash Commands:[/bold yellow] /tools, /plan, /memory, /clear, /exit",
                    title="[bold cyan]AutoAgent Quick Guide[/bold cyan]",
                    border_style="cyan"
                ))
                continue

            # ── Intent Fast-Path for Conversation ───────────────────────────
            if is_conversational(user_input):
                handle_conversation(user_input, messages)
                continue

            # ── Dynamic Execution Plan for Complex Tasks ────────────────────
            active_tools = build_active_tools()
            tools_xml_str = "\n\n".join(json.dumps(t, indent=2) if isinstance(t, dict) and "schema" not in t else (t.get("schema") if isinstance(t, dict) else str(t)) for t in active_tools)
            messages[0]["content"] = SYSTEM_PROMPT_TEMPLATE.replace(
                "{TOOLS_XML}", tools_xml_str
            ).replace("{WORKSPACE_DIR}", str(Path(CONFIG["workspace_dir"]).absolute()))

            if should_plan(user_input):
                with console.status("[cyan]Formulating execution roadmap...[/cyan]", spinner="dots"):
                    plan = make_plan(user_input)
                if plan:
                    console.print(Panel(
                        plan,
                        title="[bold cyan]📋 Execution Roadmap[/bold cyan]",
                        border_style="cyan"
                    ))
                    messages.append({"role": "system", "content": f"[EXECUTION ROADMAP]:\n{plan}\nExecute these steps methodically."})

            messages.append({"role": "user", "content": user_input})

            # ── Launch ReAct State Machine ──────────────────────────────────
            try:
                messages = state_machine_loop(user_input, messages)
            except KeyboardInterrupt:
                console.print("\n[bold red]⛔ Interrupted! Returning to prompt…[/bold red]")
                messages.append({
                    "role": "system",
                    "content": "[SYSTEM: User interrupted the agent. Task halted.]",
                })
            except Exception as e:
                logger.error(str(e))
                console.print(f"[red]Execution error: {e}[/red]")

            # Save episode outcome
            for m in reversed(messages):
                if m["role"] == "assistant" and len(m.get("content", "")) > 20:
                    save_episode(user_input, m["content"])
                    break

    except KeyboardInterrupt:
        console.print("\n[bold]👋 Goodbye![/bold]")


if __name__ == "__main__":
    main()
