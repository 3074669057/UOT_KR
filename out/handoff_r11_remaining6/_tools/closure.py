"""Read-only static local-import closure extractor (no import of target modules).

Uses `ast` only: parses each target .py file, walks Import/ImportFrom nodes at any
depth (including function bodies, where this repo does its lazy imports), resolves
names to repo-local files, and emits caller -> module -> path -> sha256 rows.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(r"D:\trae\tool\a\cross")

# sys.path roots used by the frozen scripts (they insert these at runtime).
ROOTS = [
    REPO / "scripts" / "multi_bridge",   # holdout, dev_candidate, decoder_audit, baseline_mechanism, diag...
    REPO / "src",                        # cross.*
    REPO / "scripts",
    REPO / "tools" / "cross_aml",
    REPO,
]

STDLIB_HINT = {
    "json", "sys", "os", "re", "math", "time", "typing", "pathlib", "hashlib", "ast",
    "argparse", "subprocess", "shutil", "collections", "dataclasses", "itertools",
    "functools", "random", "tempfile", "csv", "warnings", "copy", "datetime", "io",
    "traceback", "contextlib", "decimal", "statistics", "glob", "textwrap", "logging",
    "concurrent", "multiprocessing", "pickle", "gzip", "zipfile", "base64", "uuid",
    "inspect", "importlib", "abc", "enum", "types", "operator", "string", "struct",
    "threading", "queue", "signal", "socket", "unicodedata", "platform", "secrets",
    "__future__", "weakref", "numbers", "heapq", "bisect", "array", "errno", "sys",
}
EXTERNAL = {
    "numpy", "pandas", "scipy", "torch", "ot", "POT", "sklearn", "matplotlib",
    "jsonschema", "web3", "requests", "yaml", "tqdm", "networkx", "seaborn",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(mod: str, level: int, cur_file: Path) -> Path | None:
    """Resolve a dotted module name to a local .py file, or None."""
    if mod in STDLIB_HINT or mod.split(".")[0] in STDLIB_HINT:
        return None
    if mod.split(".")[0] in EXTERNAL:
        return None
    parts = [p for p in mod.split(".") if p]
    cands: list[Path] = []
    if level and level > 0:
        base = cur_file.parent
        for _ in range(level - 1):
            base = base.parent
        cands.append(base.joinpath(*parts))
    else:
        for r in ROOTS:
            cands.append(r.joinpath(*parts))
    for c in cands:
        f = c.with_suffix(".py")
        if f.is_file():
            return f
        if c.is_dir() and (c / "__init__.py").is_file():
            return c / "__init__.py"
    return None


def imports_of(path: Path) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return []
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, 0))
        elif isinstance(node, ast.ImportFrom):
            out.append((node.module or "", node.level or 0))
    return out


def closure(seeds: list[Path], max_depth: int = 8) -> dict:
    seen: dict[str, dict] = {}
    frontier = [(p, 0, "SEED") for p in seeds]
    edges: list[dict] = []
    unresolved: set[str] = set()
    while frontier:
        cur, depth, caller = frontier.pop()
        key = str(cur)
        if key in seen:
            continue
        seen[key] = {"path": cur, "depth": depth, "sha256": sha256(cur)}
        if depth >= max_depth:
            continue
        for mod, level in imports_of(cur):
            tgt = resolve(mod, level, cur)
            if tgt is None:
                if mod and mod.split(".")[0] not in STDLIB_HINT and mod.split(".")[0] not in EXTERNAL:
                    unresolved.add(mod)
                continue
            edges.append({
                "caller": str(cur.relative_to(REPO)),
                "module": mod,
                "level": level,
                "resolved": str(tgt.relative_to(REPO)),
                "sha256": sha256(tgt),
            })
            frontier.append((tgt, depth + 1, str(cur)))
    return {"files": seen, "edges": edges, "unresolved": sorted(unresolved)}


if __name__ == "__main__":
    seed_paths = [Path(a) if Path(a).is_absolute() else REPO / a for a in sys.argv[1:]]
    res = closure(seed_paths)
    out = {
        "seed_files": [str(p.relative_to(REPO)) for p in seed_paths],
        "n_local_files": len(res["files"]),
        "files": [
            {"path": str(v["path"].relative_to(REPO)), "depth": v["depth"], "sha256": v["sha256"]}
            for v in sorted(res["files"].values(), key=lambda x: str(x["path"]))
        ],
        "edges": res["edges"],
        "unresolved_local_candidates": res["unresolved"],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
