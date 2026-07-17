#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║            AutoAgent v3.0 — Elite Autonomous AI Agent            ║
║  ReAct · Live Streaming · 3-Tier Memory · Sandboxed Python       ║
║  Parallel Tools · Smart Compression · Plugin Hot-Reload          ║
║  YAML Config · Structured Logging · Session Persistence          ║
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
from ddgs import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
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

    # ── UX ──────────────────────────────────────────────────
    "enable_streaming": True,

    # ── Planning ────────────────────────────────────────────
    "enable_planning": True,
    "planning_threshold": 80,  # chars; tasks longer than this trigger auto-plan

    # ── Context — tuned for 7B/9B small models ──────────────
    # Small models lose coherence quickly. Compress earlier, keep fewer.
    "compress_threshold": 12,  # messages before compression (was 16)
    "keep_recent": 8,          # messages to keep after compression (was 10)
    "enable_smart_compress": True,

    # ── Tool output limits ───────────────────────────────────
    "search_max_results": 5,   # fewer results → less token noise
    "file_read_limit": 6000,
    "url_read_limit": 4000,    # aggressively short for small context

    # ── Observation offloading ───────────────────────────────
    # If a tool result is larger than this, save to file instead of
    # dumping the full text into the LLM context.
    "observation_limit": 3000,  # chars sent back to LLM
    "offload_large_obs": True,  # save huge results to workspace/obs_N.txt

    # ── Safety / loop prevention ─────────────────────────────
    "max_steps": 30,           # hard cap on ReAct iterations per task
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
                with open(p) as f:
                    user = yaml.safe_load(f) if (HAS_YAML and p.suffix == ".yaml") else json.load(f)
                if isinstance(user, dict):
                    cfg.update(user)
                break
            except Exception:
                pass
    return cfg


CONFIG = load_config()
console = Console()

# ═══════════════════════════════════════════════════════
# STRUCTURED LOGGER
# ═══════════════════════════════════════════════════════

class AgentLogger:
    def __init__(self, log_file: str):
        self.path = Path(log_file)
        self.sid = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _w(self, data: dict):
        try:
            entry = {"ts": datetime.now().isoformat(), "sid": self.sid, **data}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    def user_input(self, text: str):
        self._w({"type": "input", "text": text})

    def llm_call(self, dur: float, toks: int):
        self._w({"type": "llm", "dur_ms": int(dur * 1000), "toks_est": toks})

    def tool_call(self, name: str, args: dict, result: str, dur: float):
        self._w({"type": "tool", "name": name, "args": args,
                 "preview": result[:200], "dur_ms": int(dur * 1000)})

    def error(self, msg: str):
        self._w({"type": "error", "msg": msg})


logger = AgentLogger(CONFIG["log_file"])

# ═══════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════

# Deliberately compact for small (7B/9B) models.
# Long system prompts eat into the effective context window and cause
# small models to lose focus. Keep it tight and example-driven.
SYSTEM_PROMPT_TEMPLATE = """You are AutoAgent — an autonomous AI with full MacOS access.
You operate in a strict State Machine.
Your current working directory is: {WORKSPACE_DIR}

[MANDATORY FORMAT]
For EVERY turn, you MUST output your reasoning inside <thought>...</thought> tags first!
After your thought, you must either output tool call(s) OR a transition tag.

[STATES]
EVALUATE: Analyze the request and plan. Transition to RESEARCH or ACTION.
RESEARCH: Gather info using read tools (search_web, read_url, read_file).
SYNTHESIS: Write a compressed summary in <scratchpad>...</scratchpad> tags, then immediately <transition>ACTION</transition>.
ACTION: Execute tasks to fulfill the goal using write tools (write_file, execute_python, run_command). Transition to DONE when finished.

[RULES]
1. Use XML tool tags for ALL tool calls.
2. If info is missing, RESEARCH. If ready to build/modify, ACTION.
3. NEVER call a tool and transition in the same turn. Choose ONE.
4. You may execute multiple independent tools in the same turn.

AVAILABLE TOOLS:
{TOOLS_XML}
"""


# ═══════════════════════════════════════════════════════
# BASE TOOL SCHEMAS
# ═══════════════════════════════════════════════════════

BASE_TOOLS: List[Dict] = [
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
        "schema": "<read_file>\n<path>absolute or relative file path</path>\n</read_file>",
        "description": "Read a local file's contents (up to 8 000 chars). Good for code, text, JSON, CSV."
    },
    {
        "name": "write_file",
        "schema": "<write_file>\n<path>file path</path>\n<content>text content to write</content>\n</write_file>",
        "description": "Write text to a local file. Creates missing directories. Overwrites if exists."
    },
    {
        "name": "list_directory",
        "schema": "<list_directory>\n<path>directory path to list</path>\n</list_directory>",
        "description": "List files and subdirectories at a path, with sizes."
    },
    {
        "name": "sleep",
        "schema": "<sleep>\n<seconds>seconds to wait</seconds>\n</sleep>",
        "description": "Sleep for N seconds (use in polling/monitoring loops)."
    },
    {
        "name": "read_memory",
        "schema": "<read_memory></read_memory>",
        "description": "Read all facts from persistent semantic memory (key-value store)."
    },
    {
        "name": "write_memory",
        "schema": "<write_memory>\n<key>Memory key/label</key>\n<value>Value to remember</value>\n</write_memory>",
        "description": "Save a fact to persistent semantic memory. Survives across sessions."
    },
    {
        "name": "search_memory",
        "schema": "<search_memory>\n<query>Text to search</query>\n</search_memory>",
        "description": "Search semantic memory for entries matching a text query."
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


# Semantic memory (key-value, persistent)
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


# Episodic memory (session summaries)
def read_episodic_fn() -> str:
    eps = _load_json(CONFIG["episodic_file"], [])
    if not eps:
        return "No episodic memories yet."
    return "\n".join(f"[{e.get('ts', '?')}] {e.get('summary', '')}" for e in eps[-30:])


def save_episode(task: str, outcome: str) -> None:
    eps = _load_json(CONFIG["episodic_file"], [])
    eps.append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "summary": f"Task: {task[:80]} → {outcome[:220]}"
    })
    _save_json(CONFIG["episodic_file"], eps[-100:])

# ═══════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════

def think_fn(reasoning: str) -> str:
    # For small models: thinking is cheap, but we must force forward action.
    # Return empty string so this step adds nothing to the context window —
    # the model's THOUGHT is already captured in its reply text.
    return ""


_OBS_CONTENT_HASH: Dict[str, str] = {}  # content_hash → obs_filepath


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
        content = p.read_text(encoding="utf-8", errors="replace")
        limit = CONFIG["file_read_limit"]
        if len(content) > limit:
            return content[:limit] + f"\n\n… [truncated — {len(content):,} total chars, showing first {limit:,}]"
        return content
    except Exception as e:
        return f"❌ File read error: {e}"


def write_file_fn(path: str, content: str) -> str:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"✓ Wrote {len(content.encode()):,} bytes → {p}"
    except Exception as e:
        return f"❌ File write error: {e}"


def list_dir_fn(path: str) -> str:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"❌ Path does not exist: {path}"
        rows = []
        for item in sorted(p.iterdir()):
            if item.is_dir():
                rows.append(f"📁  {item.name}/")
            else:
                sz = item.stat().st_size
                label = f"{sz}B" if sz < 1024 else f"{sz // 1024}KB" if sz < 1024 ** 2 else f"{sz // 1024 ** 2}MB"
                rows.append(f"📄  {item.name}  ({label})")
        return "\n".join(rows) if rows else "Empty directory."
    except Exception as e:
        return f"❌ List error: {e}"


def search_web_fn(query: str) -> str:
    try:
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
    console.print(f"[blue]💤 Sleeping {seconds}s...[/blue]")
    time.sleep(seconds)
    return f"✓ Woke up. Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# ═══════════════════════════════════════════════════════
# PLUGIN SYSTEM  (mtime-cached hot-reload)
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
# LLM COMMUNICATION
# ═══════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def context_tokens(messages: List[Dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


def stream_ollama(messages: List[Dict]):
    """Yield text chunks from the Ollama streaming API."""
    url = f"{CONFIG['ollama_url']}/api/chat"
    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "stream": True,
        "options": {"temperature": CONFIG["temperature"], "num_ctx": CONFIG["context_size"]},
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


def chat_ollama_blocking(messages: List[Dict]) -> Dict:
    """Blocking (non-streaming) call — for planning and summarisation."""
    url = f"{CONFIG['ollama_url']}/api/chat"
    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": CONFIG["context_size"]},
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
# CONTEXT MANAGEMENT  (smart compression)
# ═══════════════════════════════════════════════════════

def smart_compress(messages: List[Dict]) -> List[Dict]:
    """
    Context compression tuned for small (7B/9B) models.
    Small models degrade significantly with long histories — compress
    earlier and more aggressively than you would with large models.
    """
    threshold = CONFIG["compress_threshold"]
    keep = CONFIG["keep_recent"]
    if len(messages) <= threshold:
        return messages

    sys_msg = messages[0]
    to_compress = messages[1:-keep]
    recent = messages[-keep:]

    if CONFIG["enable_smart_compress"] and to_compress:
        # Keep snippet short — the summary call itself burns context
        snippet = "\n".join(
            f"{m['role'].upper()}: {str(m.get('content', ''))[:300]}"
            for m in to_compress
        )
        # Use a minimal summarisation prompt (smaller = faster + less wasted tokens)
        summary_req = [
            {"role": "system", "content": (
                "Summarise this agent conversation in 4-5 bullet points. "
                "Include: tasks done, key results, errors. One line per bullet. Output ONLY bullets."
            )},
            {"role": "user", "content": snippet},
        ]
        with console.status("[dim]📦 Compressing context...[/dim]", spinner="dots"):
            summary_resp = chat_ollama_blocking(summary_req)
        summary = summary_resp.get("content", "[compression unavailable]")
        # Use a USER role for the summary — Qwen attends better to user turns
        compressed = [
            sys_msg,
            {"role": "user", "content": f"[HISTORY SUMMARY]\n{summary}"},
            {"role": "assistant", "content": "Understood. Continuing from where we left off."},
        ]
        compressed.extend(recent)
        console.print(f"[dim]📦 Context: {len(messages)} → {len(compressed)} messages[/dim]")
        return compressed

    # Fallback: simple drop
    return [sys_msg, {"role": "user", "content": "[older messages truncated]"},
            {"role": "assistant", "content": "OK, continuing."}] + recent

# ═══════════════════════════════════════════════════════
# TOOL EXTRACTION  (multi-block, auto-repair)
# ═══════════════════════════════════════════════════════

def _parse_inner_args(inner_xml: str) -> Dict[str, str]:
    args = {}
    inner_matches = re.findall(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', inner_xml, re.DOTALL)
    if inner_matches:
        for arg_name, arg_val in inner_matches:
            args[arg_name] = arg_val.strip()
    return args


def extract_tool_calls(text: str) -> List[Tuple[str, Dict]]:
    """Extract all valid XML tool calls from LLM response. Returns list of (name, args)."""
    calls: List[Tuple[str, Dict]] = []
    
    # Match <tool_name>...</tool_name>
    matches = re.findall(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', text, re.DOTALL)
    for name, inner_xml in matches:
        if name in ("transition", "scratchpad", "thought"):
            continue # Handled by state machine
            
        args = _parse_inner_args(inner_xml)
        if not args:
            # Fallback for single-arg tools where the inner content is the raw value
            if name == "execute_python":
                args["code"] = inner_xml.strip()
            elif name == "run_command":
                args["command"] = inner_xml.strip()
            elif name == "search_web":
                args["query"] = inner_xml.strip()
            elif name == "read_url":
                args["url"] = inner_xml.strip()
            elif name == "read_file":
                args["path"] = inner_xml.strip()
            elif name == "list_directory":
                args["path"] = inner_xml.strip()
            elif name == "sleep":
                args["seconds"] = inner_xml.strip()
            elif name == "search_memory":
                args["query"] = inner_xml.strip()
                
        calls.append((name, args))
        
    return calls

# ═══════════════════════════════════════════════════════
# TOOL ROUTER
# ═══════════════════════════════════════════════════════

def route_tool(tool_name: str, args: Dict, plugins: Dict) -> str:
    """Dispatch a single tool call and return the observation string."""
    ROUTES = {
        "think":          lambda: think_fn(args.get("reasoning", "")),
        "search_web":     lambda: search_web_fn(args.get("query", "")),
        "read_url":       lambda: read_url_fn(args.get("url", "")),
        "execute_python": lambda: execute_python_sandboxed(args.get("code", "")),
        "run_command":    lambda: run_command_fn(args.get("command", "")),
        "read_file":      lambda: read_file_fn(args.get("path", "")),
        "write_file":     lambda: write_file_fn(args.get("path", ""), args.get("content", "")),
        "list_directory": lambda: list_dir_fn(args.get("path", ".")),
        "sleep":          lambda: sleep_fn(int(args.get("seconds", 10))),
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
    "scrape", "download", "monitor", "schedule", "deploy", "fetch", "extract",
}


def should_plan(task: str) -> bool:
    if not CONFIG["enable_planning"]:
        return False
    words = set(task.lower().split())
    return len(task) > CONFIG["planning_threshold"] or bool(words & _COMPLEX_WORDS)


def make_plan(task: str) -> str:
    msgs = [
        {"role": "system", "content": (
            "You are a task planner for an AI agent. Given a task, produce a numbered "
            "step-by-step execution plan with concrete actions (max 8 steps). "
            "Output ONLY the plan — no intro, no conclusion."
        )},
        {"role": "user", "content": f"Task: {task}"},
    ]
    resp = chat_ollama_blocking(msgs)
    return resp.get("content", "")

# ═══════════════════════════════════════════════════════
# REACT LOOP  (core agent execution)
# ═══════════════════════════════════════════════════════

# ── Observation counter for offloading ──────────────────────────────
_obs_counter = 0


def _prepare_observation(tool_name: str, result: str) -> str:
    """
    For small models: if the raw result is too long, save it to a file
    and return a short pointer instead.  This keeps the context window
    clean and prevents the model from losing track of its task.

    Content-addressed: if the same content is returned by two different
    calls (e.g. read_file on the same offloaded obs file), the SAME path
    is reused — preventing the model from bypassing loop detection by
    getting a new obs_N filename each time.
    """
    global _obs_counter
    limit = CONFIG["observation_limit"]
    if not CONFIG["offload_large_obs"] or len(result) <= limit:
        return result

    import hashlib
    content_hash = hashlib.md5(result.encode()).hexdigest()[:12]

    # Reuse existing file if content is identical
    if content_hash in _OBS_CONTENT_HASH:
        existing = _OBS_CONTENT_HASH[content_hash]
        short = result[:limit]
        return (
            f"{short}\n\n"
            f"… [SAME CONTENT as before — already saved to {existing}\n"
            f"You have already read this. Do NOT read it again.\n"
            f"Use the information above to take action.]"
        )

    # New content: save to workspace
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
            f"… [OUTPUT TRUNCATED — full result ({len(result):,} chars) saved to:\n"
            f"  {out_path}\n"
            f"Use read_file to access specific sections if needed.]"
        )
    except Exception:
        return result[:limit] + f"\n… [truncated at {limit} chars]"


def state_machine_loop(user_input: str, messages: List[Dict]) -> List[Dict]:
    """Execute one complete user turn through the State Machine."""
    
    current_state = "EVALUATE"
    step = 0
    max_steps = CONFIG["max_steps"]
    
    # ── Loop detection ─────────────────────────────────────────────────────
    tool_sig_counts: Counter = Counter()
    recent_tools: deque = deque(maxlen=8)
    
    while True:
        step += 1
        if step > max_steps:
            console.print(f"[bold red]⛔ Max steps ({max_steps}) reached. Stopping.[/bold red]")
            messages.append({
                "role": "system",
                "content": f"[SYSTEM: Maximum step limit ({max_steps}) reached. Summarise what you accomplished and what remains.]",
            })
            with console.status("[yellow]Generating summary...[/yellow]", spinner="dots"):
                resp = chat_ollama_blocking(messages)
            summary = resp.get("content", "")
            if summary:
                console.print(Panel(Markdown(summary), title="[yellow]Task Summary[/yellow]", border_style="yellow"))
                messages.append({"role": "assistant", "content": summary})
            break
            
        # Append state reminder explicitly
        state_prompt = f"[SYSTEM: You are currently in STATE: {current_state}. Please think, act, and transition.]"
        messages.append({"role": "user", "content": state_prompt})
        
        print_status(messages)
        
        full_reply = ""
        t0 = time.time()
        
        if CONFIG["enable_streaming"]:
            with Live(console=console, refresh_per_second=10, transient=False) as live:
                live.update(Panel("[dim]…[/dim]",
                                  title=f"[bold yellow]🤖 Agent [dim]({current_state})[/dim][/bold yellow]",
                                  border_style="yellow"))
                for chunk in stream_ollama(messages):
                    full_reply += chunk
                    # Convert <thought> to visible markdown blockquotes
                    visible_reply = full_reply.replace("<thought>", "> **💭 Thought:**\n> ").replace("</thought>", "\n")
                    display = Markdown(visible_reply) if len(visible_reply) < 4000 else visible_reply[-2000:]
                    live.update(Panel(display,
                                      title=f"[bold yellow]🤖 Agent [dim]({current_state})[/dim][/bold yellow]",
                                      border_style="yellow"))
        else:
            with console.status(f"[yellow]🤖 Thinking in {current_state}…[/yellow]", spinner="dots"):
                resp = chat_ollama_blocking(messages)
                full_reply = resp.get("content", "")
            visible_reply = full_reply.replace("<thought>", "> **💭 Thought:**\n> ").replace("</thought>", "\n")
            console.print(Panel(Markdown(visible_reply),
                                title=f"[bold yellow]🤖 Agent ({current_state})[/bold yellow]",
                                border_style="yellow"))
                                
        dur_llm = time.time() - t0
        logger.llm_call(dur_llm, estimate_tokens(full_reply))
        messages.append({"role": "assistant", "content": full_reply})
        
        # Check for state transitions
        transition_match = re.search(r'<transition>(.*?)</transition>', full_reply, re.IGNORECASE)
        
        # Extract and execute tools
        tool_calls = extract_tool_calls(full_reply)
        
        if not tool_calls and not transition_match:
            # Model hallucinated or gave pure text without transition
            messages.append({"role": "user", "content": "SYSTEM WARNING: No XML tool calls or <transition> tags found. You MUST use XML tags to act or transition."})
            continue

        if tool_calls and transition_match:
            # Model tried to do both. We prioritize the tool call because it needs
            # the result before it can actually transition. We silently ignore the transition.
            transition_match = None
            
        current_plugins = get_dynamic_plugins()
        
        if transition_match:
            new_state = transition_match.group(1).strip().upper()
            if new_state in ["EVALUATE", "RESEARCH", "SYNTHESIS", "ACTION", "DONE"]:
                current_state = new_state
                console.print(f"[bold magenta]🔄 Transitioning to {current_state}[/bold magenta]")
                if current_state == "DONE":
                    break
            else:
                messages.append({"role": "user", "content": f"SYSTEM WARNING: Invalid state '{new_state}'. Valid states are EVALUATE, RESEARCH, SYNTHESIS, ACTION, DONE."})
            continue
        
        if tool_calls:
            observations = []
            
            for tool_name, tool_args in tool_calls:
                # Loop Detection
                sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                if tool_name != "think":
                    tool_sig_counts[sig] += 1
                recent_tools.append(tool_name)
                
                if tool_sig_counts[sig] > CONFIG["max_repeat_count"]:
                    console.print(f"[bold red]🔄 Loop: '{tool_name}' called {tool_sig_counts[sig]}× total![/bold red]")
                    observations.append(f"OBSERVATION from {tool_name}:\n[CRITICAL SYSTEM OVERRIDE] Tool {tool_name} with these arguments is BLOCKED because you are in an infinite loop. DO NOT apologize. DO NOT use this tool again. Transition to a different state or use a different tool IMMEDIATELY.")
                    continue
                
                if len(recent_tools) >= 6:
                    last6 = list(recent_tools)[-6:]
                    if last6[0] == last6[2] == last6[4] and last6[1] == last6[3] == last6[5] and last6[0] != last6[1]:
                        console.print(f"[bold red]🔄 Alternating loop detected: {last6[0]}↔{last6[1]} repeating![/bold red]")
                        observations.append(f"OBSERVATION from {tool_name}:\n[CRITICAL SYSTEM OVERRIDE] You are stuck alternating between '{last6[0]}' and '{last6[1]}'. DO NOT apologize. Change your approach completely.")
                        continue
                        
                # Enforce state constraints
                if current_state == "RESEARCH" and tool_name in ["write_file"]:
                    observations.append(f"OBSERVATION from {tool_name}:\nSYSTEM WARNING: Tool '{tool_name}' is forbidden in RESEARCH state.")
                    continue
                if current_state == "ACTION" and tool_name in ["search_web", "read_url"]:
                    observations.append(f"OBSERVATION from {tool_name}:\nSYSTEM WARNING: Tool '{tool_name}' is forbidden in ACTION state.")
                    continue
                if current_state == "SYNTHESIS":
                    observations.append(f"OBSERVATION from {tool_name}:\nSYSTEM WARNING: You cannot call ANY tools in the SYNTHESIS state.")
                    continue
                    
                console.print(f"╭{'─'*31} Tool Call {'─'*31}╮")
                console.print(f"│ 🔧 {tool_name:<73}│")
                console.print(f"╰{'─'*75}╯")
                
                t_tool = time.time()
                result = route_tool(tool_name, tool_args, current_plugins)
                dur_tool = time.time() - t_tool
                
                logger.tool_call(tool_name, tool_args, str(result), dur_tool)
                
                preview = str(result)
                if len(preview) > 1500:
                    preview = preview[:1500] + f"\n… [{len(str(result)):,} chars total]"
                console.print(Panel(f"[green]👁 Observation[/green] [dim]({dur_tool:.2f}s)[/dim]\n\n{preview}", border_style="green"))
                
                # Offload observations natively to avoid blowing up context
                context_result = _prepare_observation(tool_name, str(result))
                observations.append(f"OBSERVATION from {tool_name}:\n{context_result}")
                
            if observations:
                messages.append({"role": "user", "content": "\n\n---\n\n".join(observations)})
            
            # Keep system prompt in sync with any newly loaded plugins
            active_tools = list(BASE_TOOLS) + [p.TOOL_SCHEMA for p in current_plugins.values()]
            tools_xml_str = "\n\n".join(json.dumps(t, indent=2) if isinstance(t, dict) and "schema" not in t else (t.get("schema") if isinstance(t, dict) else str(t)) for t in active_tools)
            messages[0]["content"] = SYSTEM_PROMPT_TEMPLATE.replace("{TOOLS_XML}", tools_xml_str).replace("{WORKSPACE_DIR}", str(Path(CONFIG["workspace_dir"]).absolute()))
            
    return messages


# ═══════════════════════════════════════════════════════
# ASYNC INPUT  (stdin + inbox.txt polling)
# ═══════════════════════════════════════════════════════

def poll_input() -> str:
    """Non-blocking poll: whichever arrives first — stdin or inbox.txt."""
    sys.stdout.write("\n👤 You: ")
    sys.stdout.flush()

    while True:
        # Check inbox
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

        # Non-blocking stdin (1-second window)
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 1.0)
            if readable:
                line = sys.stdin.readline().strip()
                if line:
                    logger.user_input(line)
                    return line
        except Exception:
            # Fallback for non-tty / Windows
            try:
                line = input()
                if line.strip():
                    return line.strip()
            except (EOFError, KeyboardInterrupt):
                return "exit"

# ═══════════════════════════════════════════════════════
# CONTEXT STATUS BAR
# ═══════════════════════════════════════════════════════

def print_status(messages: List[Dict]) -> None:
    toks = context_tokens(messages)
    max_t = CONFIG["context_size"]
    pct = min(toks / max_t * 100, 100)
    bar_w = 22
    filled = int(bar_w * pct / 100)
    bar = "█" * filled + "░" * (bar_w - filled)
    color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
    plugins = get_dynamic_plugins()
    console.print(
        f"[dim]Context [{color}]{bar}[/{color}] ~{toks:,} tokens (~{pct:.0f}%) | "
        f"Msgs: {len(messages)} | Plugins: {len(plugins)} | Model: {CONFIG['model']}[/dim]"
    )

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def build_active_tools() -> List[Dict]:
    return list(BASE_TOOLS) + [p.TOOL_SCHEMA for p in get_dynamic_plugins().values()]


def main() -> None:
    active_tools = build_active_tools()
    plugin_count = len(active_tools) - len(BASE_TOOLS)

    console.print(Panel.fit(
        "[bold magenta]🤖  AutoAgent v3.0[/bold magenta]\n"
        "[cyan]ReAct · Live Streaming · 3-Tier Memory · Sandboxed Python[/cyan]\n"
        "[cyan]Parallel Tools · Smart Compression · Plugin Hot-Reload[/cyan]\n"
        f"[dim]Model: {CONFIG['model']} | Context: {CONFIG['context_size']:,} tokens | "
        f"Base tools: {len(BASE_TOOLS)} | Plugins: {plugin_count}[/dim]",
        border_style="magenta",
        title="[bold]AutoAgent[/bold]",
    ))

    tools_xml_str = "\n\n".join(json.dumps(t, indent=2) if isinstance(t, dict) and "schema" not in t else (t.get("schema") if isinstance(t, dict) else str(t)) for t in active_tools)
    system_content = SYSTEM_PROMPT_TEMPLATE.replace("{TOOLS_XML}", tools_xml_str).replace("{WORKSPACE_DIR}", str(Path(CONFIG["workspace_dir"]).absolute()))
    messages: List[Dict] = [{"role": "system", "content": system_content}]

    # Session restore is DISABLED.
    # Restoring a failed/stuck session injects a confused context that
    # causes the model to repeat the same failed actions on every restart.
    # Each conversation starts fresh. The semantic memory (agent_memory.json)
    # and episodic memory (agent_episodic.json) still persist facts across runs.
    #
    # To re-enable: uncomment the block below.
    #
    # if session_path.exists():
    #     try:
    #         saved = json.loads(session_path.read_text())
    #         saved_msgs = saved.get("messages", [])
    #         if len(saved_msgs) > 1:
    #             messages = [messages[0]] + saved_msgs[1:]
    #             console.print(f"[dim]📂 Session restored ...[/dim]")
    #     except Exception:
    #         pass

    console.print(
        "[dim]Type [bold]exit[/bold] to quit. "
        "Drop tasks into [bold]inbox.txt[/bold] for headless mode.[/dim]"
    )
    console.rule(style="dim")

    try:
        while True:
            print_status(messages)
            user_input = poll_input()

            if user_input.lower() in {"exit", "quit", "q"}:
                console.print("\n[bold]👋 Goodbye![/bold]")
                break

            if not user_input.strip():
                continue

            # Refresh tool list (plugins may have changed between turns)
            active_tools = build_active_tools()
            tools_xml_str = "\n\n".join(json.dumps(t, indent=2) if isinstance(t, dict) and "schema" not in t else (t.get("schema") if isinstance(t, dict) else str(t)) for t in active_tools)
            messages[0]["content"] = SYSTEM_PROMPT.replace(
                "{TOOLS_XML}", tools_xml_str
            )
            messages.append({"role": "user", "content": user_input})

            try:
                messages = state_machine_loop(user_input, messages)
            except KeyboardInterrupt:
                console.print("\n[bold red]⛔ Interrupted! Returning to prompt…[/bold red]")
                messages.append({
                    "role": "system",
                    "content": "[SYSTEM: User interrupted the agent. Task aborted.]",
                })
            except Exception as e:
                logger.error(str(e))
                console.print(f"[red]Unexpected error: {e}[/red]")

            # Save episode summary from last assistant turn
            for m in reversed(messages):
                if m["role"] == "assistant" and len(m.get("content", "")) > 20:
                    save_episode(user_input, m["content"])
                    break

            console.rule(style="dim")

            # Persist session
            try:
                session_path.write_text(
                    json.dumps(
                        {"messages": messages, "ts": datetime.now().isoformat()},
                        indent=2, default=str,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

    except KeyboardInterrupt:
        console.print("\n[bold]Interrupted. Exiting.[/bold]")


if __name__ == "__main__":
    main()
