"""Minimal OpenAI-compatible client (chat + tool calling + vision).

Works with DeepSeek, Qwen (DashScope compatible mode), OpenAI, Moonshot,
local vLLM/Ollama ... anything exposing /chat/completions.
"""
import base64
import io
import json
import os
import re

import requests

from .core import cfg_get, log


class LLMUnavailable(RuntimeError):
    pass


def env_key(name):
    """Read an API key from the environment; on Windows also fall back to the
    user environment in the registry (keys set with `setx` are not visible to
    terminals that were already open)."""
    if not name:
        return ""
    val = os.environ.get(name, "")
    if not val and os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                val = str(winreg.QueryValueEx(k, name)[0])
        except OSError:
            val = ""
    return val.strip()


def _client_cfg(cfg, section):
    if not cfg_get(cfg, f"{section}.enabled", False):
        raise LLMUnavailable(f"{section} disabled in config")
    key = env_key(cfg_get(cfg, f"{section}.api_key_env", ""))
    if not key:
        raise LLMUnavailable(f"{section}: env var {cfg_get(cfg, f'{section}.api_key_env')} not set")
    return cfg_get(cfg, f"{section}.base_url").rstrip("/"), cfg_get(cfg, f"{section}.model"), key


def chat(cfg, messages, tools=None, section="llm", temperature=0.2, timeout=180, json_mode=False):
    base, model, key = _client_cfg(cfg, section)
    body = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(f"{base}/chat/completions", json=body, timeout=timeout,
                      headers={"Authorization": f"Bearer {key}"})
    if r.status_code >= 400:
        raise RuntimeError(f"{section} HTTP {r.status_code}: {r.text[:500]}")
    return r.json()["choices"][0]["message"]


def image_part(path_or_pil, max_side=1024):
    from PIL import Image
    im = Image.open(path_or_pil) if isinstance(path_or_pil, str) else path_or_pil
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def parse_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            text = text[a:b + 1]
    return json.loads(text)


def vision_json(cfg, prompt, images, system=None):
    """Ask the VLM about one or more images and get a JSON object back."""
    content = [{"type": "text", "text": prompt}] + [image_part(i) for i in images]
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": content})
    msg = chat(cfg, msgs, section="vlm", temperature=0.0)
    try:
        return parse_json(msg.get("content") or "")
    except Exception:
        log("VLM returned non-JSON:", (msg.get("content") or "")[:300])
        raise
