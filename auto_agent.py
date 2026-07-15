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
SYSTEM_PROMPT = """You are AutoAgent — an autonomous AI with full MacOS access. EXECUTE tasks; never tell the user to do them manually.

TOOL FORMAT — output a ```json block:
```json
{"tool": "tool_name", "arguments": {"key": "value"}}
```
JSON RULES (critical):
- Double quotes only. No Python triple-quotes inside JSON.
- Escape code newlines as \\n, internal quotes as \\".
- Wait for OBSERVATION before the next tool call.

REASONING: Think step-by-step. State THOUGHT → ACTION → read OBSERVATION → repeat until done.
On error: re-read the error carefully, try a different approach, never give up.

MEMORY:
- read_memory / write_memory / search_memory → persistent key-value facts
- read_episodic → past task summaries (check first on complex tasks)

DYNAMIC TOOLS: write plugins/<name>.py with TOOL_SCHEMA dict + execute(args) fn → instantly available.

AVAILABLE TOOLS:
{TOOLS_JSON}
"""


# ═══════════════════════════════════════════════════════
# BASE TOOL SCHEMAS
# ═══════════════════════════════════════════════════════

BASE_TOOLS: List[Dict] = [
    {"type": "function", "function": {
        "name": "think",
        "description": "Reason through a problem or make a plan before acting. No side effects.",
        "parameters": {"type": "object",
                       "properties": {"reasoning": {"type": "string", "description": "Your internal analysis or step-by-step plan."}},
                       "required": ["reasoning"]}}},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web via DuckDuckGo. Returns titles, URLs, and snippets.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Search query"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_url",
        "description": "Fetch and extract readable text from a URL. Strips nav/ads, focuses on main content.",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string", "description": "Full URL to fetch"}},
                       "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "execute_python",
        "description": "Run Python code in a sandboxed subprocess. Returns stdout + stderr. CWD is workspace/.",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Python code (escape newlines as \\n in JSON)"}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a bash command on MacOS. Returns combined stdout + stderr.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string", "description": "Bash command to run"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a local file's contents (up to 8 000 chars). Good for code, text, JSON, CSV.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Absolute or relative file path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write text to a local file. Creates missing directories. Overwrites if exists.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "File path"},
                                      "content": {"type": "string", "description": "Text content to write"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List files and subdirectories at a path, with sizes.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Directory path to list"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "sleep",
        "description": "Sleep for N seconds (use in polling/monitoring loops).",
        "parameters": {"type": "object",
                       "properties": {"seconds": {"type": "integer", "description": "Seconds to wait"}},
                       "required": ["seconds"]}}},
    {"type": "function", "function": {
        "name": "read_memory",
        "description": "Read all facts from persistent semantic memory (key-value store).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "write_memory",
        "description": "Save a fact to persistent semantic memory. Survives across sessions.",
        "parameters": {"type": "object",
                       "properties": {"key": {"type": "string", "description": "Memory key/label"},
                                      "value": {"type": "string", "description": "Value to remember"}},
                       "required": ["key", "value"]}}},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Search semantic memory for entries matching a text query.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Text to search across memory keys and values"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_episodic",
        "description": "Read summaries of past completed tasks from episodic memory.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
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

def _repair_json(raw: str) -> Optional[Dict]:
    """Multi-pass JSON repair for common small-model output mistakes."""
    candidates = [raw]

    # Pass 1: escape bare newlines inside strings
    candidates.append(re.sub(r'(?<!\\)\n', r'\\n', raw))

    # Pass 2: replace Python literals with JSON equivalents
    py_fixed = re.sub(r'(?<!["\w])True(?!["\w])', 'true', raw)
    py_fixed = re.sub(r'(?<!["\w])False(?!["\w])', 'false', py_fixed)
    py_fixed = re.sub(r'(?<!["\w])None(?!["\w])', 'null', py_fixed)
    candidates.append(py_fixed)

    # Pass 3: single-quote → double-quote (only on key names and simple values)
    sq_fixed = re.sub(r"'([^']*?)'", r'"\1"', raw)
    candidates.append(sq_fixed)

    # Pass 4: strip trailing commas before } or ]
    tc_fixed = re.sub(r',\s*([}\]])', r'\1', raw)
    candidates.append(tc_fixed)

    # Pass 5: combined passes
    combined = re.sub(r'(?<!\\)\n', r'\\n', tc_fixed)
    combined = re.sub(r'(?<!["\w])True(?!["\w])', 'true', combined)
    combined = re.sub(r'(?<!["\w])False(?!["\w])', 'false', combined)
    combined = re.sub(r'(?<!["\w])None(?!["\w])', 'null', combined)
    candidates.append(combined)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, Exception):
            continue
    return None


def extract_tool_calls(text: str) -> List[Tuple[str, Dict]]:
    """Extract all valid tool calls from LLM response. Returns list of (name, args)."""
    calls: List[Tuple[str, Dict]] = []

    # Primary: ```json { ... } ``` or ``` { ... } ```
    blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)

    # Fallback 1: bare JSON with "tool" key (model forgot the fences)
    if not blocks:
        blocks = [m.group(1) for m in re.finditer(r'(\{\s*"tool"\s*:.*?\})', text, re.DOTALL)]

    # Fallback 2: bare JSON with "name" + "arguments" keys (older format)
    if not blocks:
        blocks = [m.group(1) for m in re.finditer(r'(\{\s*"name"\s*:.*?"arguments".*?\})', text, re.DOTALL)]

    for raw in blocks:
        data = _repair_json(raw.strip())
        if data is None:
            continue
        tool_name = data.get("tool") or data.get("name")
        args = data.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        if tool_name:
            calls.append((tool_name, args))

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


def react_loop(user_input: str, messages: List[Dict]) -> List[Dict]:
    """Execute one complete user turn through the ReAct loop."""

    # Optional: auto-generate plan for complex tasks
    if should_plan(user_input):
        console.print("[dim]🗺  Generating execution plan...[/dim]")
        plan = make_plan(user_input)
        if plan:
            console.print(Panel(plan, title="[bold cyan]🗺  Execution Plan[/bold cyan]",
                                border_style="cyan"))
            # Inject plan as a user message (not system) — Qwen follows user turns better
            messages.append({
                "role": "user",
                "content": f"Before you start, here is a suggested plan:\n{plan}\nNow execute it step by step.",
            })

    json_error_count = 0
    last_tool_sig: Optional[str] = None
    repeat_count = 0
    step = 0
    max_steps = CONFIG["max_steps"]

    # ── Loop detection ─────────────────────────────────────────────────────
    # Track how many times each exact (tool, args) pair has been called.
    # This catches BOTH consecutive loops (A→A) AND alternating loops (A→B→A→B).
    tool_sig_counts: Counter = Counter()
    # Track last N tool names for pattern detection
    recent_tools: deque = deque(maxlen=8)

    # ── Progress enforcer ────────────────────────────────────────────
    # Small models get stuck in research loops. If no concrete output
    # (execute_python / write_file / run_command) is produced within
    # MAX_RESEARCH_STEPS consecutive steps, we force action.
    CONCRETE_TOOLS = {"execute_python", "write_file", "run_command"}
    MAX_RESEARCH_STEPS = 4   # steps without concrete output before warning
    steps_without_concrete = 0


    while True:
        # ── Hard step limit (prevents infinite loops in small models) ──
        step += 1
        if step > max_steps:
            console.print(f"[bold red]⛔ Max steps ({max_steps}) reached. Stopping.[/bold red]")
            messages.append({
                "role": "system",
                "content": f"[SYSTEM: Maximum step limit ({max_steps}) reached. Summarise what you accomplished and what remains.]",
            })
            # One final LLM call for summary
            with console.status("[yellow]Generating summary...[/yellow]", spinner="dots"):
                resp = chat_ollama_blocking(messages)
            summary = resp.get("content", "")
            if summary:
                console.print(Panel(Markdown(summary),
                                    title="[yellow]Task Summary[/yellow]",
                                    border_style="yellow"))
                messages.append({"role": "assistant", "content": summary})
            break

        # ── Compress if needed ──────────────────────────
        messages = smart_compress(messages)

        # ── Call LLM (streaming or blocking) ───────────
        t0 = time.time()
        full_reply = ""

        if CONFIG["enable_streaming"]:
            with Live(console=console, refresh_per_second=10, transient=False) as live:
                live.update(Panel("[dim]…[/dim]",
                                  title=f"[bold yellow]🤖 Agent  [dim](step {step}/{max_steps})[/dim][/bold yellow]",
                                  border_style="yellow"))
                for chunk in stream_ollama(messages):
                    full_reply += chunk
                    display = Markdown(full_reply) if len(full_reply) < 4000 else full_reply[-2000:]
                    live.update(Panel(display,
                                      title=f"[bold yellow]🤖 Agent  [dim](step {step}/{max_steps})[/dim][/bold yellow]",
                                      border_style="yellow"))
        else:
            with console.status(f"[yellow]🤖 Thinking… (step {step}/{max_steps})[/yellow]", spinner="dots"):
                resp = chat_ollama_blocking(messages)
                full_reply = resp.get("content", "")
            console.print(Panel(Markdown(full_reply),
                                title=f"[bold yellow]🤖 Agent (step {step})[/bold yellow]",
                                border_style="yellow"))

        dur_llm = time.time() - t0
        logger.llm_call(dur_llm, estimate_tokens(full_reply))
        messages.append({"role": "assistant", "content": full_reply})

        # ── Extract tool calls ──────────────────────────
        tool_calls = extract_tool_calls(full_reply)

        if not tool_calls:
            # Check if the model tried but produced bad JSON
            attempted = bool(re.search(r'"tool"\s*:|```json', full_reply))
            if attempted:
                json_error_count += 1
                if json_error_count >= CONFIG["max_json_retries"]:
                    console.print("[red]⛔ JSON parse failed 3×. Returning to prompt.[/red]")
                    messages.append({
                        "role": "system",
                        "content": "[SYSTEM: You failed to write valid JSON 3 times. Stop tool calls. Give your best answer in plain text.]",
                    })
                    break
                # ── Small-model-specific error guidance ──
                # Give a concrete mini-example to re-anchor the model
                errmsg = (
                    f"SYSTEM ERROR: Your JSON could not be parsed (attempt {json_error_count}/{CONFIG['max_json_retries']}).\n"
                    "Common mistakes: single quotes, unescaped newlines, trailing commas, Python True/False/None.\n"
                    "Correct format example:\n"
                    '```json\n{"tool": "run_command", "arguments": {"command": "ls"}}\n```\n'
                    "Try again with correct JSON."
                )
                messages.append({"role": "user", "content": errmsg})
                continue
            # Clean finish — model gave a conversational response
            break

        json_error_count = 0  # reset on clean parse

        # ── Execute tools — SEQUENTIAL ONLY for small models ─────────
        # Parallel execution confuses 7B/9B models: observations arrive
        # out-of-order relative to reasoning. Always execute one at a time.
        current_plugins = get_dynamic_plugins()

        # Execute only the FIRST valid tool call per step.
        # If the model emitted multiple, we'll process the rest next iteration.
        tool_name, tool_args = tool_calls[0]
        if len(tool_calls) > 1:
            console.print(
                f"[dim]ℹ  Model requested {len(tool_calls)} tools; executing first ({tool_name}), "
                "rest will follow in subsequent steps.[/dim]"
            )

        # ── Comprehensive loop detection ──────────────────────────────────
        sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

        # 1. Global per-signature count — excludes 'think' (it's always unique reasoning)
        if tool_name != "think":
            tool_sig_counts[sig] += 1
        recent_tools.append(tool_name)

        if tool_sig_counts[sig] > CONFIG["max_repeat_count"]:
            console.print(f"[bold red]🔄 Loop: '{tool_name}' called {tool_sig_counts[sig]}× total![/bold red]")
            
            if tool_name in CONCRETE_TOOLS:
                msg = (
                    f"SYSTEM WARNING: You have called '{tool_name}' with the same broken arguments "
                    f"{tool_sig_counts[sig]} times. Your code is failing. "
                    "STOP trying the exact same code. You must completely change your approach, "
                    "fix the syntax error, or ask the user for help."
                )
            else:
                msg = (
                    f"SYSTEM WARNING: You have called '{tool_name}' with the same arguments "
                    f"{tool_sig_counts[sig]} times. You already have the result. Stop re-reading. "
                    "Your next action MUST use execute_python or write_file to make a real change."
                )

            messages.append({"role": "user", "content": msg})
            last_tool_sig = None
            repeat_count = 0
            continue
        # 2. Alternating pattern detection: e.g. [A,B,A,B,A,B] in last 6 calls
        if len(recent_tools) >= 6:
            last6 = list(recent_tools)[-6:]
            # Pattern [X,Y,X,Y,X,Y] — two tools alternating 3 times
            if (last6[0] == last6[2] == last6[4] and
                    last6[1] == last6[3] == last6[5] and
                    last6[0] != last6[1]):
                console.print(
                    f"[bold red]🔄 Alternating loop detected: "
                    f"{last6[0]}↔{last6[1]} repeating![/bold red]"
                )
                
                if last6[0] in CONCRETE_TOOLS or last6[1] in CONCRETE_TOOLS:
                    msg = (
                        f"SYSTEM WARNING: You are stuck alternating between '{last6[0]}' and "
                        f"'{last6[1]}'. Your actions are failing. STOP repeating the same steps. "
                        "Change your approach completely or ask for help."
                    )
                else:
                    msg = (
                        f"SYSTEM WARNING: You are stuck alternating between '{last6[0]}' and "
                        f"'{last6[1]}' without making progress. You already know the file contents. "
                        "STOP reading. Your ONLY valid next action is to write improved code with "
                        "execute_python or write_file. Take concrete action NOW."
                    )
                
                messages.append({"role": "user", "content": msg})
                last_tool_sig = None
                repeat_count = 0
                continue

        # 3. Consecutive repeat guard (original fast path)
        if sig == last_tool_sig:
            repeat_count += 1
            if repeat_count >= CONFIG["max_repeat_count"]:
                if tool_name in CONCRETE_TOOLS:
                    msg = (
                        f"SYSTEM: '{tool_name}' failed or returned the same result twice in a row. "
                        "Your code is broken. Do NOT try the exact same code again. Fix it."
                    )
                else:
                    msg = (
                        f"SYSTEM: '{tool_name}' returned the same result twice in a row. "
                        "Use the information you already have and move forward."
                    )
                
                messages.append({"role": "user", "content": msg})
                repeat_count = 0
                last_tool_sig = None
                continue
        else:
            repeat_count = 0
            last_tool_sig = sig

        # Hot-reload plugins (agent may have just written one)
        current_plugins = get_dynamic_plugins()

        console.print(Panel(
            f"[bold cyan]🔧 {tool_name}[/bold cyan]\n"
            f"[dim]{json.dumps(tool_args, indent=2)[:500]}[/dim]",
            title=f"Tool Call  [dim](step {step})[/dim]", border_style="cyan",
        ))

        t_tool = time.time()
        result = route_tool(tool_name, tool_args, current_plugins)
        dur_tool = time.time() - t_tool
        logger.tool_call(tool_name, tool_args, result, dur_tool)

        # Display full result in terminal
        preview = str(result)
        if len(preview) > 1500:
            preview = preview[:1500] + f"\n… [{len(str(result)):,} chars total]"
        console.print(Panel(
            f"[green]👁  Observation[/green] [dim]({dur_tool:.2f}s)[/dim]\n\n{preview}",
            border_style="green",
        ))

        # Offload large observations to file — keep context clean for small model
        context_result = _prepare_observation(tool_name, str(result))

        # Update progress counter
        if tool_name in CONCRETE_TOOLS:
            steps_without_concrete = 0  # reset on real action
        else:
            steps_without_concrete += 1

        # ── Progress enforcer: force concrete output after too much research ──
        if steps_without_concrete >= MAX_RESEARCH_STEPS:
            console.print(
                f"[bold red]⏰ No concrete output in {steps_without_concrete} steps. "
                f"Forcing action.[/bold red]"
            )
            steps_without_concrete = 0
            messages.append({
                "role": "user",
                "content": (
                    f"URGENT — ACTION REQUIRED: You have completed {MAX_RESEARCH_STEPS} steps "
                    "of research without producing any output file or running any code.\n"
                    "You already have enough information. STOP researching.\n"
                    "Write Python code RIGHT NOW using execute_python to create or modify the "
                    "target file with whatever content you have gathered so far.\n"
                    "Do NOT search the web again. Do NOT read any files. Just write the code."
                ),
            })
            continue

        # Refresh plugin list (catches newly created plugins)
        current_plugins = get_dynamic_plugins()

        # Feed observation back — clear, simple format for small models
        messages.append({
            "role": "user",
            "content": f"OBSERVATION from {tool_name}:\n{context_result}\n\nWhat is your next step?"
        })

        # Keep system prompt in sync with any newly loaded plugins
        active_tools = list(BASE_TOOLS) + [p.TOOL_SCHEMA for p in current_plugins.values()]
        messages[0]["content"] = SYSTEM_PROMPT.replace(
            "{TOOLS_JSON}", json.dumps(active_tools, indent=2)
        )

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

    system_content = SYSTEM_PROMPT.replace("{TOOLS_JSON}", json.dumps(active_tools, indent=2))
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
            messages[0]["content"] = SYSTEM_PROMPT.replace(
                "{TOOLS_JSON}", json.dumps(active_tools, indent=2)
            )
            messages.append({"role": "user", "content": user_input})

            try:
                messages = react_loop(user_input, messages)
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
