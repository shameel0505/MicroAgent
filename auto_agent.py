#!/usr/bin/env python3
"""
AutoAgent v5.0
==============
What's new:
  * Grammar-enforced tool dispatch  (Ollama format= logit mask)
  * Parallel tool execution         (ThreadPoolExecutor for read tools)
  * Session persistence             (save/restore conversations)
  * Ollama health check at startup
  * 5 new tools: append_file, get_clipboard, set_clipboard, run_apple_script, create_note
  * Two-tier intent classifier      (fast keywords + grammar LLM)
  * Tool deduplication within reply
  * Memory auto-injection into system prompt
  * /plan, /session, /history slash commands
  * Step counter (step X of Y) in UI
  * Thread-safe logger
"""

import os, sys, json, re, time, subprocess, importlib.util, select, hashlib, threading
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

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "qwen2.5-coder:7b",
    "context_size": 16384,
    "temperature": 0.0,
    "ollama_url": "http://localhost:11434",
    "llm_retries": 3,
    "retry_delay": 2,
    "python_timeout": 30,
    "command_timeout": 30,
    "web_timeout": 15,
    "intent_timeout": 8,
    "workspace_dir": "workspace",
    "plugins_dir": "plugins",
    "enable_streaming": True,
    "enable_intent_gating": True,
    "intent_use_grammar": True,
    "enable_planning": True,
    "planning_threshold": 50,
    "enable_grammar_repair": True,
    "grammar_repair_timeout": 45,
    "enable_parallel_tools": True,
    "max_parallel_workers": 4,
    "compress_threshold": 20,
    "keep_recent": 8,
    "enable_smart_compress": True,
    "search_max_results": 5,
    "file_read_limit": 6000,
    "url_read_limit": 4000,
    "observation_limit": 3000,
    "offload_large_obs": True,
    "max_steps": 30,
    "max_json_retries": 3,
    "max_repeat_count": 2,
    "memory_file": "agent_memory.json",
    "episodic_file": "agent_episodic.json",
    "session_file": "agent_session.json",
    "log_file": "agent.log",
    "session_keep_messages": 40,
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
# STRUCTURED LOGGER (thread-safe)
# ═══════════════════════════════════════════════════════

class AgentLogger:
    def __init__(self, log_file: str):
        self.path = Path(log_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _w(self, data: dict):
        entry = {"time": datetime.now().isoformat(), **data}
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def user_input(self, text: str): self._w({"event": "user_input", "text": text})
    def llm_call(self, dur: float, toks: int): self._w({"event": "llm_call", "duration_s": round(dur, 2), "tokens": toks})
    def tool_call(self, name: str, args: dict, result: str, dur: float):
        self._w({"event": "tool_call", "tool": name, "args": args, "result_len": len(result), "duration_s": round(dur, 2)})
    def error(self, msg: str): self._w({"event": "error", "message": msg})
    def grammar_repair(self, success: bool, reply_len: int):
        self._w({"event": "grammar_repair", "success": success, "reply_len": reply_len})


logger = AgentLogger(CONFIG["log_file"])

# ═══════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """You are AutoAgent — an elite autonomous workstation AI assistant running locally on Apple Silicon.
Your current working directory is: {WORKSPACE_DIR}
{MEMORY_CONTEXT}

[OPERATING PROTOCOL]
1. Always begin with your reasoning in <thought>...</thought> tags.
2. After your thought, invoke ONE tool call OR a <transition>STATE</transition> tag. Never both.
3. If you need user clarification, call <ask_user>. Never write questions to disk or sleep.
4. When the task is fully complete, call <finish> with a comprehensive summary.
5. Do NOT over-act. If the user asked a simple question, answer it and finish immediately.

[STATES]
EVALUATE  -> Analyse the task. Plan next move. Transition to RESEARCH or ACTION.
RESEARCH  -> Gather facts: search_web, read_url, read_file.
SYNTHESIS -> Consolidate research in <scratchpad>...</scratchpad>, then transition to ACTION.
ACTION    -> Build deliverables: execute_python, write_file, run_command, open_file.
DONE      -> Task complete. Call <finish>.

[FINISH IMMEDIATELY when]
- Task is answerable from your own knowledge
- You have written the requested file(s)
- Research is complete and synthesised
- User asked a conversational or factual question

AVAILABLE TOOLS:
{TOOLS_XML}
"""

CHAT_PROMPT = """You are AutoAgent, a friendly, ultra-capable local AI workstation assistant powered by Qwen 2.5 on Apple Silicon M4.
Guidelines:
- Respond warmly, directly, and concisely.
- Do NOT output XML tags, state transitions, or tool calls during conversational chat.
"""

# ═══════════════════════════════════════════════════════
# GRAMMAR SCHEMAS (Ollama format= logit masks)
# ═══════════════════════════════════════════════════════

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["CHAT", "TASK"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["intent", "confidence"]
}


def _build_repair_schema(tool_names: List[str]) -> Dict:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["call_tool", "transition", "finish", "respond"]},
            "tool_name": {"type": "string", "enum": tool_names},
            "tool_args": {"type": "object"},
            "next_state": {"type": "string", "enum": ["EVALUATE", "RESEARCH", "SYNTHESIS", "ACTION", "DONE"]},
            "finish_summary": {"type": "string"}
        },
        "required": ["action"]
    }

# ═══════════════════════════════════════════════════════
# BASE TOOL SCHEMAS
# ═══════════════════════════════════════════════════════

BASE_TOOLS: List[Dict] = [
    {"name": "ask_user",       "schema": "<ask_user>\n<question>Your question</question>\n</ask_user>",                                         "description": "Ask the user a clarifying question. Execution pauses until they reply."},
    {"name": "finish",         "schema": "<finish>\n<summary>Final answer or deliverable summary</summary>\n</finish>",                         "description": "Conclude the task with a final answer. Call this when done."},
    {"name": "search_web",     "schema": "<search_web>\n<query>search query</query>\n</search_web>",                                            "description": "DuckDuckGo web search. Returns titles, URLs, snippets."},
    {"name": "read_url",       "schema": "<read_url>\n<url>URL to fetch</url>\n</read_url>",                                                    "description": "Fetch and extract readable text from a URL."},
    {"name": "execute_python", "schema": "<execute_python>\n# python code\n</execute_python>",                                                  "description": "Run Python in a sandboxed subprocess. CWD is workspace/."},
    {"name": "run_command",    "schema": "<run_command>\nbash command\n</run_command>",                                                          "description": "Run a bash command on macOS."},
    {"name": "read_file",      "schema": "<read_file>\n<path>file path</path>\n</read_file>",                                                   "description": "Read a local file (up to 6,000 chars)."},
    {"name": "write_file",     "schema": "<write_file>\n<path>path</path>\n<content>content</content>\n</write_file>",                         "description": "Write text to a file. Creates dirs. Overwrites if exists."},
    {"name": "append_file",    "schema": "<append_file>\n<path>path</path>\n<content>content</content>\n</append_file>",                       "description": "Append text to a file without overwriting it."},
    {"name": "open_file",      "schema": "<open_file>\n<path>file path</path>\n</open_file>",                                                   "description": "Open a file in its native macOS app (Excel, Preview, etc.)."},
    {"name": "open_url",       "schema": "<open_url>\n<url>URL</url>\n</open_url>",                                                             "description": "Open a URL in the default macOS browser."},
    {"name": "notify_user",    "schema": "<notify_user>\n<title>title</title>\n<message>message</message>\n</notify_user>",                    "description": "Show a native macOS notification banner."},
    {"name": "list_directory", "schema": "<list_directory>\n<path>directory path</path>\n</list_directory>",                                    "description": "List files and subdirectories with sizes."},
    {"name": "get_clipboard",  "schema": "<get_clipboard></get_clipboard>",                                                                       "description": "Read the current macOS clipboard."},
    {"name": "set_clipboard",  "schema": "<set_clipboard>\n<text>text to copy</text>\n</set_clipboard>",                                       "description": "Write text to the macOS clipboard."},
    {"name": "run_apple_script","schema": "<run_apple_script>\nAppleScript code here\n</run_apple_script>",                                    "description": "Execute arbitrary AppleScript on macOS."},
    {"name": "create_note",    "schema": "<create_note>\n<title>Note title</title>\n<content>Body lines</content>\n</create_note>",            "description": "Create an Apple Note with title and bullets."},
    {"name": "sleep",          "schema": "<sleep>\n<seconds>N</seconds>\n</sleep>",                                                             "description": "Wait N seconds (use only when polling external processes)."},
    {"name": "read_memory",    "schema": "<read_memory></read_memory>",                                                                           "description": "Read all persistent semantic memory."},
    {"name": "write_memory",   "schema": "<write_memory>\n<key>key</key>\n<value>value</value>\n</write_memory>",                             "description": "Save a key-value fact to persistent memory."},
    {"name": "search_memory",  "schema": "<search_memory>\n<query>query</query>\n</search_memory>",                                            "description": "Search semantic memory for a substring."},
    {"name": "read_episodic",  "schema": "<read_episodic></read_episodic>",                                                                       "description": "Read past task summaries from episodic memory."},
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
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
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
    return f"Saved to memory: {key!r} = {value!r}"


def search_memory_fn(query: str) -> str:
    mem = _load_json(CONFIG["memory_file"], {})
    if not mem:
        return "Semantic memory is empty."
    q = query.lower()
    hits = [(k, v) for k, v in mem.items() if not k.startswith("_") and (q in k.lower() or q in str(v).lower())]
    if not hits:
        return f"No memory entries match {query!r}."
    return "Matches:\n" + "\n".join(f"  {k}: {v}" for k, v in hits)


def read_episodic_fn() -> str:
    eps = _load_json(CONFIG["episodic_file"], [])
    if not eps:
        return "No episodic memories yet."
    lines = ["Episodic Memory (Past Tasks):"]
    for e in eps[-5:]:
        lines.append(f"  [{e.get('timestamp', '')[:10]}] {e.get('task', '')} -> {e.get('outcome', '')[:150]}")
    return "\n".join(lines)


def save_episode(task: str, outcome: str) -> None:
    eps = _load_json(CONFIG["episodic_file"], [])
    eps.append({"timestamp": datetime.now().isoformat(), "task": task[:200], "outcome": outcome[:500]})
    _save_json(CONFIG["episodic_file"], eps[-50:])


def get_memory_context() -> str:
    mem = _load_json(CONFIG["memory_file"], {})
    filtered = {k: v for k, v in mem.items() if not k.startswith("_")}
    if not filtered:
        return ""
    lines = ["\n[USER MEMORY from past sessions]"]
    for k, v in list(filtered.items())[:10]:
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════
# SESSION PERSISTENCE
# ═══════════════════════════════════════════════════════

def load_session() -> Optional[List[Dict]]:
    sf = Path(CONFIG["session_file"])
    if not sf.exists():
        return None
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) > 1:
            return data
    except Exception:
        pass
    return None


def save_session(messages: List[Dict]) -> None:
    keep = CONFIG.get("session_keep_messages", 40)
    try:
        Path(CONFIG["session_file"]).write_text(
            json.dumps(messages[-keep:], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass

# ═══════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════

def ask_user_fn(question: str) -> str:
    console.print(Panel(f"[bold yellow]{question}[/bold yellow]",
                        title="[bold cyan]AutoAgent Question[/bold cyan]", border_style="cyan"))
    try:
        reply = input("Your Reply: ").strip()
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
            return f"File not found: {path}"
        subprocess.run(["open", str(p.resolve())], check=True)
        return f"Opened {p.name} in macOS."
    except Exception as e:
        return f"Failed to open file: {e}"


def open_url_fn(url: str) -> str:
    try:
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        subprocess.run(["open", url], check=True)
        return f"Opened URL: {url}"
    except Exception as e:
        return f"Failed to open URL: {e}"


def notify_user_fn(title: str, message: str) -> str:
    try:
        clean_title = title.replace('"', '\\"')
        clean_msg = message.replace('"', '\\"')
        cmd = f'display notification "{clean_msg}" with title "{clean_title}" sound name "Glass"'
        subprocess.run(["osascript", "-e", cmd], check=True)
        return "macOS notification delivered."
    except Exception as e:
        return f"Notification error: {e}"


def execute_python_sandboxed(code: str) -> str:
    ws = Path(CONFIG["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=CONFIG["python_timeout"], cwd=str(ws.resolve()))
        out = r.stdout
        if r.stderr.strip():
            out += ("\n" if out else "") + f"[STDERR]\n{r.stderr.strip()}"
        return out.strip() or "Code executed with no output."
    except subprocess.TimeoutExpired:
        return f"Timed out after {CONFIG['python_timeout']}s."
    except Exception as e:
        return f"Execution error: {e}"


def run_command_fn(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=CONFIG["command_timeout"])
        return (r.stdout + r.stderr).strip() or "Command ran with no output."
    except subprocess.TimeoutExpired:
        return f"Command timed out after {CONFIG['command_timeout']}s"
    except Exception as e:
        return f"Command error: {e}"


def _resolve_read_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    ws = Path(CONFIG["workspace_dir"])
    if p.parts and p.parts[0] == ws.name:
        return p
    if p.exists():
        return p
    if (ws / p).exists():
        return ws / p
    return ws / p


def _resolve_write_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    ws = Path(CONFIG["workspace_dir"])
    if p.parts and p.parts[0] == ws.name:
        return p
    return ws / p


def read_file_fn(path: str) -> str:
    try:
        p = _resolve_read_path(path)
        if not p.exists():
            return f"File does not exist: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        limit = CONFIG["file_read_limit"]
        if len(content) > limit:
            content = content[:limit] + f"\n\n[truncated at {limit:,} chars]"
        return content
    except Exception as e:
        return f"File read error: {e}"


def write_file_fn(path: str, content: str) -> str:
    try:
        p = _resolve_write_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.encode()):,} bytes to {p}"
    except Exception as e:
        return f"File write error: {e}"


def append_file_fn(path: str, content: str) -> str:
    try:
        p = _resolve_write_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content.encode()):,} bytes to {p}"
    except Exception as e:
        return f"Append error: {e}"


def list_dir_fn(path: str) -> str:
    try:
        p = _resolve_read_path(path)
        if not p.exists():
            return f"Path does not exist: {path}"
        rows = [f"{'Dir' if i.is_dir() else 'File'}: {i.name}" + (f" ({i.stat().st_size:,}b)" if i.is_file() else "") for i in sorted(p.iterdir())]
        return "\n".join(rows) if rows else "(empty directory)"
    except Exception as e:
        return f"List dir error: {e}"


def get_clipboard_fn() -> str:
    try:
        r = subprocess.run(["pbpaste"], capture_output=True, text=True)
        return r.stdout or "(empty clipboard)"
    except Exception as e:
        return f"Clipboard read error: {e}"


def set_clipboard_fn(text: str) -> str:
    try:
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
        return f"Copied {len(text):,} chars to clipboard."
    except Exception as e:
        return f"Clipboard write error: {e}"


def run_apple_script_fn(script: str) -> str:
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)
        output = (r.stdout + r.stderr).strip()
        if r.returncode != 0 and not output:
            return f"AppleScript failed (code {r.returncode})."
        return output or "AppleScript executed."
    except subprocess.TimeoutExpired:
        return "AppleScript timed out."
    except Exception as e:
        return f"AppleScript error: {e}"


def create_note_fn(title: str, content: str) -> str:
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    bullets = "".join(f"<li>{l.lstrip('-*• ')}</li>" for l in lines)
    html = f"<h1>{title}</h1><ul>{bullets}</ul>"
    clean_html = html.replace("\\", "\\\\").replace('"', '\\"')
    script = ('tell application "Notes"\ntell default account\n' +
               f'make new note at default folder with properties {{body:"{clean_html}"}}\n' +
               'end tell\nend tell')
    result = run_apple_script_fn(script)
    return f'Apple Note created: "{title}"' if "error" not in result.lower() else result


def search_web_fn(query: str) -> str:
    try:
        if not DDGS:
            return "DuckDuckGo search library not available."
        results = DDGS().text(query, max_results=CONFIG["search_max_results"])
        if not results:
            return "No results found."
        lines = [f"Search: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'No title')}\n   {r.get('href', '')}\n   {r.get('body', '')[:260]}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


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
        return text[:limit] + (f"\n[truncated at {limit:,} chars]" if len(text) > limit else "")
    except requests.HTTPError as e:
        return f"HTTP {e.response.status_code}: {url}"
    except Exception as e:
        return f"URL read error: {e}"


def sleep_fn(seconds: int) -> str:
    console.print(f"[blue]Waiting {seconds}s...[/blue]")
    time.sleep(seconds)
    return f"Resumed at {datetime.now().strftime('%H:%M:%S')}"

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
                console.print(f"[dim]Plugin reloaded: {py_file.name}[/dim]")
            except Exception as e:
                console.print(f"[red]Plugin [{py_file.name}] failed: {e}[/red]")
                continue
        if hasattr(mod, "TOOL_SCHEMA") and hasattr(mod, "execute"):
            name = mod.TOOL_SCHEMA.get("function", {}).get("name", py_file.stem)
            plugins[name] = mod
    return plugins

# ═══════════════════════════════════════════════════════
# LLM INFERENCE
# ═══════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def context_tokens(messages: List[Dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


def check_ollama_health() -> bool:
    try:
        r = requests.get(f"{CONFIG['ollama_url']}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def stream_ollama(messages: List[Dict], temperature: Optional[float] = None):
    url = f"{CONFIG['ollama_url']}/api/chat"
    temp = temperature if temperature is not None else CONFIG["temperature"]
    payload = {"model": CONFIG["model"], "messages": messages, "stream": True,
               "options": {"temperature": temp, "num_ctx": CONFIG["context_size"]}}
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
        yield f"\n[LLM Error: {e}]"


def chat_ollama_blocking(messages: List[Dict], temperature: float = 0.1,
                          format_schema: Optional[Dict] = None, timeout: int = 120) -> Dict:
    url = f"{CONFIG['ollama_url']}/api/chat"
    payload = {"model": CONFIG["model"], "messages": messages, "stream": False,
               "options": {"temperature": temperature, "num_ctx": CONFIG["context_size"]}}
    if format_schema:
        payload["format"] = format_schema
    for attempt in range(CONFIG["llm_retries"]):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("message", {"role": "assistant", "content": ""})
        except Exception as e:
            if attempt < CONFIG["llm_retries"] - 1:
                time.sleep(CONFIG["retry_delay"] * (attempt + 1))
            else:
                return {"role": "assistant", "content": f"[LLM Error: {e}]"}

# ═══════════════════════════════════════════════════════
# GRAMMAR GUARDRAIL — TOOL DISPATCH REPAIR
# ═══════════════════════════════════════════════════════

def repair_tool_dispatch(full_reply: str, available_tool_names: List[str],
                          messages: List[Dict]) -> Optional[Tuple[str, Dict]]:
    """
    When XML extraction yields nothing, use grammar-enforced JSON to recover
    the intended action. This is the #1 reliability improvement over v4.
    """
    if not CONFIG.get("enable_grammar_repair", True):
        return None

    schema = _build_repair_schema(available_tool_names)
    repair_msgs = [
        {"role": "system", "content": (
            "You are a JSON intent extractor. An AI agent produced a response with "
            "no detectable tool call. Analyse it and determine what it intended to do. "
            "Return JSON with action (call_tool/transition/finish/respond), and relevant fields."
        )},
        {"role": "user", "content": (
            f"Agent last response (tail):\n{full_reply[-600:]}\n\n"
            f"Available tools: {available_tool_names}\n"
            "What did the agent intend next?"
        )}
    ]
    try:
        resp = chat_ollama_blocking(repair_msgs, temperature=0.0, format_schema=schema,
                                    timeout=CONFIG.get("grammar_repair_timeout", 45))
        data = json.loads(resp.get("content", "{}"))
        action = data.get("action", "respond")

        if action == "call_tool":
            tool = data.get("tool_name")
            args = data.get("tool_args") or {}
            if tool and tool in available_tool_names:
                logger.grammar_repair(True, len(full_reply))
                console.print(f"[dim yellow]Grammar repair: {tool}({json.dumps(args)[:80]})[/dim yellow]")
                return (tool, args)
        elif action == "transition":
            state = data.get("next_state", "")
            if state in ("EVALUATE", "RESEARCH", "SYNTHESIS", "ACTION", "DONE"):
                logger.grammar_repair(True, len(full_reply))
                return ("__transition__", {"state": state})
        elif action == "finish":
            summary = data.get("finish_summary", "Task complete.")
            logger.grammar_repair(True, len(full_reply))
            return ("finish", {"summary": summary})
    except Exception as e:
        console.print(f"[dim red]Grammar repair error: {e}[/dim red]")

    logger.grammar_repair(False, len(full_reply))
    return None

# ═══════════════════════════════════════════════════════
# INTENT CLASSIFICATION (Two-Tier)
# ═══════════════════════════════════════════════════════

_GREETING_SET = {
    "hi", "hello", "hey", "howdy", "sup", "yo", "good morning", "good evening",
    "good afternoon", "who are you", "what are you", "what can you do", "help",
    "what tools do you have", "thanks", "thank you", "cool", "nice", "awesome",
    "great", "ok", "okay", "bye", "goodbye"
}


def is_conversational(text: str) -> bool:
    if not CONFIG.get("enable_intent_gating", True):
        return False
    t = text.strip().lower().rstrip("!?.")
    if not t:
        return True
    # Tier 1: keyword fast-path (zero latency)
    if t in _GREETING_SET:
        return True
    words = t.split()
    if len(words) <= 2 and words[0] in {"hi", "hello", "hey", "yo"}:
        return True
    # Tier 2: LLM grammar classification (for short ambiguous messages)
    if CONFIG.get("intent_use_grammar", True) and 3 <= len(words) <= 12:
        try:
            resp = chat_ollama_blocking(
                messages=[
                    {"role": "system", "content": (
                        "Classify as CHAT or TASK. "
                        "CHAT = greeting, casual Q&A, small talk, knowledge question with no computer action needed. "
                        "TASK = any request to do something on the computer, create a file, search the web, run code."
                    )},
                    {"role": "user", "content": text}
                ],
                temperature=0.0, format_schema=INTENT_SCHEMA,
                timeout=CONFIG.get("intent_timeout", 8)
            )
            data = json.loads(resp.get("content", "{}"))
            return data.get("intent", "TASK") == "CHAT"
        except Exception:
            pass
    return False

# ═══════════════════════════════════════════════════════
# CONVERSATION FAST-PATH
# ═══════════════════════════════════════════════════════

def handle_conversation(user_input: str, messages: List[Dict]) -> None:
    chat_prompt = [{"role": "system", "content": CHAT_PROMPT}, {"role": "user", "content": user_input}]
    console.print()
    full_reply = ""
    with Live(console=console, refresh_per_second=12, transient=False) as live:
        live.update(Panel("[dim]Thinking...[/dim]", title="[bold cyan]AutoAgent[/bold cyan]", border_style="cyan"))
        for chunk in stream_ollama(chat_prompt, temperature=0.7):
            full_reply += chunk
            live.update(Panel(Markdown(full_reply), title="[bold cyan]AutoAgent[/bold cyan]", border_style="cyan"))
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": full_reply})

# ═══════════════════════════════════════════════════════
# SMART CONTEXT COMPRESSION
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
        snippet = "\n".join(f"{m['role'].upper()}: {str(m.get('content', ''))[:300]}" for m in to_compress)
        resp = chat_ollama_blocking([
            {"role": "system", "content": "Summarise in 4-5 bullet points: tasks done, key results, errors. Output ONLY bullets."},
            {"role": "user", "content": snippet}
        ])
        summary = resp.get("content", "").strip()
        if summary:
            compressed = [sys_msg, {"role": "system", "content": f"[COMPRESSED HISTORY]\n{summary}"}] + recent
            console.print(f"[dim]Context compressed: {len(messages)} -> {len(compressed)} msgs[/dim]")
            return compressed
    return [sys_msg] + recent

# ═══════════════════════════════════════════════════════
# XML TOOL CALL EXTRACTION (with deduplication)
# ═══════════════════════════════════════════════════════

_POSITIONAL_MAP: Dict[str, str] = {
    "execute_python": "code", "run_command": "command", "run_apple_script": "script",
    "search_web": "query", "read_url": "url", "read_file": "path", "list_directory": "path",
    "sleep": "seconds", "search_memory": "query", "ask_user": "question",
    "open_file": "path", "open_url": "url", "set_clipboard": "text", "finish": "summary",
}
_SKIP_TAGS = {"transition", "scratchpad", "thought"}
_NO_ARG_TOOLS = {"read_memory", "read_episodic", "get_clipboard"}


def extract_tool_calls(text: str) -> List[Tuple[str, Dict]]:
    calls: List[Tuple[str, Dict]] = []
    seen_sigs: set = set()
    for name, inner_xml in re.findall(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', text, re.DOTALL):
        if name in _SKIP_TAGS:
            continue
        # Parse nested args
        args = {k: v.strip() for k, v in re.findall(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', inner_xml, re.DOTALL)}
        if not args:
            raw = inner_xml.strip()
            if name in _POSITIONAL_MAP:
                args[_POSITIONAL_MAP[name]] = raw
            elif name in _NO_ARG_TOOLS:
                args = {}
            elif name == "notify_user":
                args = {"title": "AutoAgent", "message": raw}
        # Deduplication
        sig = f"{name}:{json.dumps(args, sort_keys=True)}"
        if sig in seen_sigs:
            console.print(f"[dim yellow]Duplicate tool skipped: {name}[/dim yellow]")
            continue
        seen_sigs.add(sig)
        calls.append((name, args))
    return calls

# ═══════════════════════════════════════════════════════
# TOOL ROUTING
# ═══════════════════════════════════════════════════════

def route_tool(tool_name: str, args: Dict, plugins: Dict) -> str:
    ROUTES = {
        "ask_user":          lambda: ask_user_fn(args.get("question", "")),
        "finish":            lambda: finish_fn(args.get("summary", "")),
        "open_file":         lambda: open_file_fn(args.get("path", "")),
        "open_url":          lambda: open_url_fn(args.get("url", "")),
        "notify_user":       lambda: notify_user_fn(args.get("title", "AutoAgent"), args.get("message", "Done.")),
        "search_web":        lambda: search_web_fn(args.get("query", "")),
        "read_url":          lambda: read_url_fn(args.get("url", "")),
        "execute_python":    lambda: execute_python_sandboxed(args.get("code", "")),
        "run_command":       lambda: run_command_fn(args.get("command", "")),
        "read_file":         lambda: read_file_fn(args.get("path", "")),
        "write_file":        lambda: write_file_fn(args.get("path", ""), args.get("content", "")),
        "append_file":       lambda: append_file_fn(args.get("path", ""), args.get("content", "")),
        "list_directory":    lambda: list_dir_fn(args.get("path", ".")),
        "get_clipboard":     lambda: get_clipboard_fn(),
        "set_clipboard":     lambda: set_clipboard_fn(args.get("text", "")),
        "run_apple_script":  lambda: run_apple_script_fn(args.get("script", "")),
        "create_note":       lambda: create_note_fn(args.get("title", "Note"), args.get("content", "")),
        "sleep":             lambda: sleep_fn(int(args.get("seconds", 5))),
        "read_memory":       lambda: read_memory_fn(),
        "write_memory":      lambda: write_memory_fn(args.get("key", ""), args.get("value", "")),
        "search_memory":     lambda: search_memory_fn(args.get("query", "")),
        "read_episodic":     lambda: read_episodic_fn(),
    }
    if tool_name in ROUTES:
        return ROUTES[tool_name]()
    if tool_name in plugins:
        try:
            return str(plugins[tool_name].execute(args))
        except Exception as e:
            return f"Plugin error [{tool_name}]: {e}"
    return f"Unknown tool '{tool_name}'. Available: {list(ROUTES) + list(plugins)}"

# ═══════════════════════════════════════════════════════
# PARALLEL TOOL EXECUTION
# ═══════════════════════════════════════════════════════

_SEQUENTIAL_TOOLS = {
    "write_file", "append_file", "write_memory", "run_command", "execute_python",
    "open_file", "open_url", "notify_user", "ask_user", "finish", "set_clipboard",
    "run_apple_script", "create_note", "sleep"
}


def _timed_route(name: str, args: Dict, plugins: Dict) -> Tuple[str, str, float]:
    t0 = time.time()
    result = route_tool(name, args, plugins)
    return (name, str(result), time.time() - t0)

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
    resp = chat_ollama_blocking([
        {"role": "system", "content": "You are a master task planner for an AI agent. Given a task, produce a numbered 3-5 step roadmap. Be concrete. Output ONLY numbered steps."},
        {"role": "user", "content": f"Task: {task}"}
    ], temperature=0.1)
    return resp.get("content", "").strip()

# ═══════════════════════════════════════════════════════
# OBSERVATION OFFLOADING
# ═══════════════════════════════════════════════════════

_obs_counter = 0
_OBS_HASH: Dict[str, str] = {}


def _prepare_observation(tool_name: str, result: str) -> str:
    global _obs_counter
    limit = CONFIG["observation_limit"]
    if not CONFIG["offload_large_obs"] or len(result) <= limit:
        return result
    content_hash = hashlib.md5(result.encode()).hexdigest()[:12]
    if content_hash in _OBS_HASH:
        return result[:limit] + f"\n[Same content as {_OBS_HASH[content_hash]} — already processed.]"
    _obs_counter += 1
    ws = Path(CONFIG["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    out_path = ws / f"obs_{_obs_counter}_{tool_name}.txt"
    try:
        out_path.write_text(result, encoding="utf-8")
        _OBS_HASH[content_hash] = str(out_path)
        return result[:limit] + f"\n[Full output ({len(result):,} chars) saved to {out_path}. Use read_file for specific sections.]"
    except Exception:
        return result[:limit] + f"\n[truncated at {limit} chars]"

# ═══════════════════════════════════════════════════════
# REACT STATE MACHINE LOOP
# ═══════════════════════════════════════════════════════

def _format_tools_for_prompt(active_tools: List[Dict]) -> str:
    parts = []
    for t in active_tools:
        if not isinstance(t, dict):
            continue
        if "schema" in t:
            parts.append(f"- {t['name']}: {t['description']}\n  Usage: {t['schema']}")
        elif "function" in t:
            fn = t["function"]
            parts.append(f"- {fn.get('name', '?')}: {fn.get('description', '')}")
    return "\n\n".join(parts)


def _build_system_content(active_tools: List[Dict]) -> str:
    return (SYSTEM_PROMPT_TEMPLATE
            .replace("{TOOLS_XML}", _format_tools_for_prompt(active_tools))
            .replace("{WORKSPACE_DIR}", str(Path(CONFIG["workspace_dir"]).absolute()))
            .replace("{MEMORY_CONTEXT}", get_memory_context()))


def state_machine_loop(user_input: str, messages: List[Dict]) -> List[Dict]:
    current_state = "EVALUATE"
    step = 0
    max_steps = CONFIG["max_steps"]
    tool_sig_counts: Counter = Counter()
    recent_tools: deque = deque(maxlen=8)

    current_plugins = get_dynamic_plugins()
    available_tool_names = [t["name"] for t in BASE_TOOLS] + list(current_plugins.keys())

    while True:
        step += 1
        if step > max_steps:
            console.print(f"\n[bold red]Reached step limit ({max_steps}).[/bold red]")
            messages.append({"role": "system", "content": "[SYSTEM: Step limit reached. Call <finish> immediately.]"})
            resp = chat_ollama_blocking(messages)
            summary = resp.get("content", "")
            if summary:
                console.print(Panel(Markdown(summary), title="[yellow]Summary[/yellow]", border_style="yellow"))
                messages.append({"role": "assistant", "content": summary})
            break

        state_prompt = (f"[SYSTEM: STATE={current_state} | Step {step}/{max_steps}. "
                        "Think, then call ONE tool or transition. If task is done, call <finish> now.]")
        messages.append({"role": "user", "content": state_prompt})
        messages = smart_compress(messages)
        print_status(messages, step, max_steps)

        full_reply = ""
        t0 = time.time()

        if CONFIG["enable_streaming"]:
            with Live(console=console, refresh_per_second=10, transient=False) as live:
                live.update(Panel("[dim]Thinking...[/dim]",
                                  title=f"[bold yellow]AutoAgent [{current_state} step {step}][/bold yellow]",
                                  border_style="yellow"))
                for chunk in stream_ollama(messages):
                    full_reply += chunk
                    visible = full_reply.replace("<thought>", "> Thought: ").replace("</thought>", "\n")
                    display = Markdown(visible) if len(visible) < 4000 else visible[-2000:]
                    live.update(Panel(display, title=f"[bold yellow]AutoAgent [{current_state} step {step}][/bold yellow]", border_style="yellow"))
        else:
            with console.status(f"[yellow]Reasoning ({current_state} step {step})...[/yellow]", spinner="dots"):
                resp = chat_ollama_blocking(messages)
                full_reply = resp.get("content", "")
            visible = full_reply.replace("<thought>", "> Thought: ").replace("</thought>", "\n")
            console.print(Panel(Markdown(visible), title=f"[yellow]AutoAgent [{current_state} step {step}][/yellow]", border_style="yellow"))

        dur_llm = time.time() - t0
        logger.llm_call(dur_llm, estimate_tokens(full_reply))
        messages.append({"role": "assistant", "content": full_reply})

        transition_match = re.search(r'<transition>(.*?)</transition>', full_reply, re.IGNORECASE)
        tool_calls = extract_tool_calls(full_reply)

        # Finish tool check
        finish_call = next(((t, a) for t, a in tool_calls if t == "finish"), None)
        if finish_call:
            summary = finish_call[1].get("summary", "")
            console.print(Panel(Markdown(summary), title="[bold green]Task Complete[/bold green]", border_style="green"))
            notify_user_fn("AutoAgent", "Task complete!")
            break

        # No tool calls and no transition
        if not tool_calls and not transition_match:
            clean_reply = re.sub(r'<thought>.*?</thought>', '', full_reply, flags=re.DOTALL).strip()

            # Try grammar repair
            if CONFIG.get("enable_grammar_repair", True):
                repaired = repair_tool_dispatch(full_reply, available_tool_names, messages)
                if repaired:
                    rname, rargs = repaired
                    if rname == "__transition__":
                        new_state = rargs.get("state", current_state)
                        console.print(f"[magenta]Grammar-repaired transition -> {new_state}[/magenta]")
                        current_state = new_state
                        if current_state == "DONE":
                            break
                        continue
                    else:
                        tool_calls = [(rname, rargs)]
                else:
                    if clean_reply and current_state in ("ACTION", "SYNTHESIS", "EVALUATE") and step > 1:
                        console.print(Panel(Markdown(clean_reply), title="[green]Response[/green]", border_style="green"))
                        break
                    else:
                        messages.append({"role": "user", "content":
                            "SYSTEM: No tool call or transition detected. Use <tool_name>...</tool_name> "
                            "to call a tool, <transition>STATE</transition> to change state, or "
                            "<finish><summary>...</summary></finish> to conclude."})
                        continue

        if tool_calls and transition_match:
            transition_match = None  # tools take priority

        current_plugins = get_dynamic_plugins()
        available_tool_names = [t["name"] for t in BASE_TOOLS] + list(current_plugins.keys())

        if transition_match and not tool_calls:
            new_state = transition_match.group(1).strip().upper()
            if new_state in ["EVALUATE", "RESEARCH", "SYNTHESIS", "ACTION", "DONE"]:
                current_state = new_state
                console.print(f"[magenta]-> {current_state}[/magenta]")
                if current_state == "DONE":
                    clean = re.sub(r'<thought>.*?</thought>', '', full_reply, flags=re.DOTALL).strip()
                    if clean:
                        console.print(Panel(Markdown(clean), title="[green]Done[/green]", border_style="green"))
                    break
            else:
                messages.append({"role": "user", "content": f"SYSTEM: Invalid state '{new_state}'. Valid: EVALUATE, RESEARCH, SYNTHESIS, ACTION, DONE."})
            continue

        if tool_calls:
            observations = []
            break_loop = False

            # Loop prevention
            for tool_name, tool_args in tool_calls:
                sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                tool_sig_counts[sig] += 1
                recent_tools.append(tool_name)

                if tool_sig_counts[sig] > CONFIG["max_repeat_count"]:
                    console.print(f"\n[bold red]Loop: '{tool_name}' repeated {tool_sig_counts[sig]}x with same args.[/bold red]")
                    console.print(Panel(
                        f"Agent repeating [yellow]{tool_name}[/yellow] with no progress.\n"
                        f"Args: {json.dumps(tool_args)}\n\nPress Enter to pause or type guidance.",
                        title="[red]Loop Intercepted[/red]", border_style="red"))
                    try:
                        guidance = input("Human Guidance: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        guidance = ""
                    if not guidance:
                        break_loop = True
                        break
                    tool_sig_counts.clear()
                    observations.append(f"OBSERVATION: Human redirected: {guidance}")
                    continue

            if break_loop:
                break

            # Split into parallel (read-only) and sequential (side-effect) calls
            parallel_calls = [(i, n, a) for i, (n, a) in enumerate(tool_calls) if n not in _SEQUENTIAL_TOOLS]
            sequential_calls = [(i, n, a) for i, (n, a) in enumerate(tool_calls) if n in _SEQUENTIAL_TOOLS]

            results: Dict[int, Tuple[str, str, float]] = {}

            # Run read-only tools in parallel
            if parallel_calls and len(parallel_calls) > 1 and CONFIG.get("enable_parallel_tools"):
                console.print(f"[dim cyan]Running {len(parallel_calls)} tools in parallel[/dim cyan]")
                workers = min(CONFIG.get("max_parallel_workers", 4), len(parallel_calls))
                with ThreadPoolExecutor(max_workers=workers) as exc:
                    futures = {exc.submit(_timed_route, n, a, current_plugins): i for i, n, a in parallel_calls}
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            results[idx] = (tool_calls[idx][0], f"Parallel error: {e}", 0.0)
            else:
                for i, n, a in parallel_calls:
                    results[i] = _timed_route(n, a, current_plugins)

            # Run sequential tools
            for i, n, a in sequential_calls:
                console.print(f"[cyan]Tool: {n}[/cyan]")
                results[i] = _timed_route(n, a, current_plugins)

            # Display results & collect observations
            for i in range(len(tool_calls)):
                if i not in results:
                    continue
                t_name, t_result, t_dur = results[i]
                logger.tool_call(t_name, tool_calls[i][1], t_result, t_dur)
                preview = t_result[:1200] + (f"\n[{len(t_result):,} chars total]" if len(t_result) > 1200 else "")
                console.print(Panel(f"[green]Observation ({t_dur:.2f}s)[/green]\n\n{preview}", border_style="green"))

                if t_result.startswith("TASK_COMPLETED:"):
                    break_loop = True
                    break

                observations.append(f"OBSERVATION from {t_name}:\n{_prepare_observation(t_name, t_result)}")

            if break_loop:
                break

            if observations:
                messages.append({"role": "user", "content": "\n\n---\n\n".join(observations)})

            # Refresh system prompt
            active_tools = build_active_tools()
            messages[0]["content"] = _build_system_content(active_tools)

    return messages

# ═══════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════

def poll_input() -> str:
    sys.stdout.write("\n You: ")
    sys.stdout.flush()
    while True:
        inbox = Path("inbox.txt")
        if inbox.exists():
            try:
                content = inbox.read_text(encoding="utf-8").strip()
                if content:
                    inbox.write_text("", encoding="utf-8")
                    console.print("\n[green]Task from inbox.txt[/green]")
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


def print_status(messages: List[Dict], step: int = 0, max_steps: int = 0) -> None:
    toks = context_tokens(messages)
    max_t = CONFIG["context_size"]
    pct = min(toks / max_t * 100, 100)
    bar_w = 20
    filled = int(bar_w * pct / 100)
    bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
    color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
    step_info = f" | Step {step}/{max_steps}" if step else ""
    console.print(f"[dim]Context [{color}]{bar}[/{color}] ~{toks:,} tok (~{pct:.0f}%) | "
                  f"Msgs: {len(messages)} | Model: {CONFIG['model']}{step_info}[/dim]")


def build_active_tools() -> List[Dict]:
    return list(BASE_TOOLS) + [p.TOOL_SCHEMA for p in get_dynamic_plugins().values()]

# ═══════════════════════════════════════════════════════
# MAIN INTERACTION LOOP
# ═══════════════════════════════════════════════════════

def main() -> None:
    # Health check
    with console.status("[cyan]Connecting to Ollama...[/cyan]", spinner="dots"):
        online = check_ollama_health()
    if not online:
        console.print(Panel(
            f"[red]Ollama is offline at {CONFIG['ollama_url']}[/red]\n\n"
            "Start it with: [bold]ollama serve[/bold]\nThen restart AutoAgent.",
            title="[red]Connection Error[/red]", border_style="red"))
        sys.exit(1)

    active_tools = build_active_tools()
    plugin_count = len(active_tools) - len(BASE_TOOLS)

    console.print()
    console.print(Panel.fit(
        "[bold cyan]AutoAgent v5.0 — Apple Silicon M4 Runtime[/bold cyan]\n"
        "[dim]Grammar Guardrails | Parallel Tools | Session Persistence | Intent AI[/dim]\n\n"
        f"Model: [green]{CONFIG['model']}[/green]  Context: [green]{CONFIG['context_size']:,}[/green] tokens  "
        f"Tools: [green]{len(BASE_TOOLS)} base + {plugin_count} plugins[/green]\n"
        "Commands: [yellow]/help /plan /tools /memory /session /history /clear /exit[/yellow]",
        border_style="cyan", title="AutoAgent"))

    system_content = _build_system_content(active_tools)
    messages: List[Dict] = [{"role": "system", "content": system_content}]

    # Session restore
    saved = load_session()
    if saved and len(saved) > 1:
        console.print(Panel(
            f"[cyan]Saved session: [bold]{len(saved)} messages[/bold].[/cyan] Type [yellow]y[/yellow] to restore.",
            title="[dim]Session Found[/dim]", border_style="dim"))
        try:
            if input("Restore? [y/N]: ").strip().lower() == "y":
                messages = saved
                messages[0]["content"] = system_content
                console.print("[green]Session restored.[/green]")
        except (EOFError, KeyboardInterrupt):
            pass

    try:
        while True:
            print_status(messages)
            user_input = poll_input()

            if user_input.lower() in {"exit", "quit", "/exit", "/quit", "q"}:
                save_session(messages)
                console.print("\n[bold]Session saved. Goodbye![/bold]")
                break

            if not user_input.strip():
                continue

            cmd = user_input.strip()

            if cmd in ("/clear", "clear"):
                os.system("clear")
                continue

            if cmd == "/tools":
                tbl = Table(title="AutoAgent Tools", border_style="cyan")
                tbl.add_column("Tool", style="bold yellow", no_wrap=True)
                tbl.add_column("Description", style="white")
                for t in build_active_tools():
                    if isinstance(t, dict):
                        name = t.get("name") or t.get("function", {}).get("name", "plugin")
                        desc = t.get("description") or t.get("function", {}).get("description", "")
                        tbl.add_row(name, desc)
                console.print(tbl)
                continue

            if cmd == "/memory":
                console.print(Panel(read_memory_fn(), title="[cyan]Semantic Memory[/cyan]", border_style="cyan"))
                continue

            if cmd == "/history":
                console.print(Panel(read_episodic_fn(), title="[cyan]Episodic History[/cyan]", border_style="cyan"))
                continue

            if cmd == "/session":
                tbl = Table(title="Session Messages", border_style="blue")
                tbl.add_column("#", style="dim", width=4)
                tbl.add_column("Role", style="yellow", width=12)
                tbl.add_column("Preview", style="white")
                for i, m in enumerate(messages):
                    tbl.add_row(str(i), m.get("role", "?"), str(m.get("content", ""))[:80].replace("\n", " "))
                console.print(tbl)
                console.print(f"[dim]Total: {len(messages)} msgs | ~{context_tokens(messages):,} tokens[/dim]")
                continue

            if cmd.startswith("/plan"):
                task_for_plan = cmd[5:].strip()
                if not task_for_plan:
                    try:
                        task_for_plan = input("Task to plan: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        continue
                if task_for_plan:
                    with console.status("[cyan]Planning...[/cyan]", spinner="dots"):
                        plan = make_plan(task_for_plan)
                    console.print(Panel(plan, title="[cyan]Execution Roadmap[/cyan]", border_style="cyan"))
                continue

            if cmd == "/help":
                console.print(Panel(
                    "Chat naturally or give tasks. Slash commands:\n"
                    "  /tools    - list all tools\n"
                    "  /plan     - generate task roadmap\n"
                    "  /memory   - view semantic memory\n"
                    "  /history  - view past tasks\n"
                    "  /session  - view current messages\n"
                    "  /clear    - clear screen\n"
                    "  /exit     - save & quit\n\n"
                    "Drop a task in inbox.txt for background pickup.",
                    title="[cyan]AutoAgent v5.0 Help[/cyan]", border_style="cyan"))
                continue

            # Intent classification
            if is_conversational(user_input):
                handle_conversation(user_input, messages)
                save_session(messages)
                continue

            # Refresh tools + system prompt
            active_tools = build_active_tools()
            messages[0]["content"] = _build_system_content(active_tools)

            # Auto-plan for complex tasks
            if should_plan(user_input):
                with console.status("[cyan]Formulating roadmap...[/cyan]", spinner="dots"):
                    plan = make_plan(user_input)
                if plan:
                    console.print(Panel(plan, title="[cyan]Execution Roadmap[/cyan]", border_style="cyan"))
                    messages.append({"role": "system", "content": f"[EXECUTION ROADMAP]:\n{plan}\nExecute methodically."})

            messages.append({"role": "user", "content": user_input})

            try:
                messages = state_machine_loop(user_input, messages)
            except KeyboardInterrupt:
                console.print("\n[red]Interrupted![/red]")
                messages.append({"role": "system", "content": "[SYSTEM: User interrupted. Task halted.]"})
            except Exception as e:
                logger.error(str(e))
                console.print(f"[red]Error: {e}[/red]")

            for m in reversed(messages):
                if m["role"] == "assistant" and len(m.get("content", "")) > 20:
                    save_episode(user_input, m["content"])
                    break

            save_session(messages)

    except KeyboardInterrupt:
        save_session(messages)
        console.print("\n[bold]Session saved. Goodbye![/bold]")


if __name__ == "__main__":
    main()
