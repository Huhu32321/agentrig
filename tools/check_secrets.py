"""Fail if any file that would be committed contains an API key.

    python tools/check_secrets.py            # scans the repository (git-tracked + untracked, not ignored)

Checks the literal values of LLM_API_KEY / VLM_API_KEY / TRIPO_API_KEY from the environment
(if set) and generic key shapes (sk-..., Bearer tokens). Key values are never printed.
Exit code 1 on a hit - use it as a pre-commit hook:  echo "python tools/check_secrets.py" > .git/hooks/pre-commit
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = [re.compile(rb"sk-[A-Za-z0-9._\-]{20,}"), re.compile(rb"Bearer\s+[A-Za-z0-9._\-]{24,}")]
BINARY = (".png", ".jpg", ".jpeg", ".glb", ".fbx", ".blend", ".mp4", ".pdf", ".uasset", ".umap", ".npy")


def files():
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.splitlines()
        return [os.path.join(ROOT, f) for f in out]
    except Exception:
        skip = {".git", ".venv", "output", "__pycache__", "bench_data"}
        res = []
        for r, ds, fs in os.walk(ROOT):
            ds[:] = [d for d in ds if d not in skip]
            res += [os.path.join(r, f) for f in fs if f != "config.yaml"]
        return res


def env_values():
    vals = []
    for name in ("LLM_API_KEY", "VLM_API_KEY", "TRIPO_API_KEY"):
        v = os.environ.get(name, "")
        if not v and os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                    v = str(winreg.QueryValueEx(k, name)[0])
            except OSError:
                v = ""
        if len(v.strip()) >= 8:
            vals.append((name, v.strip().encode()))
    return vals


def main():
    keys, hits = env_values(), []
    for p in files():
        if not os.path.isfile(p) or os.path.getsize(p) > 50 * 1024 * 1024:
            continue
        data = open(p, "rb").read()
        rel = os.path.relpath(p, ROOT)
        for name, v in keys:
            if v in data:
                hits.append(f"{rel}: contains the value of {name}")
        if not p.lower().endswith(BINARY) and any(pt.search(data) for pt in PATTERNS):
            hits.append(f"{rel}: looks like an API key")
    if hits:
        print("API key found - do not commit:")
        print("\n".join("  " + h for h in hits))
        sys.exit(1)
    print(f"no secrets found ({len(keys)} env key value(s) checked)")


if __name__ == "__main__":
    main()
