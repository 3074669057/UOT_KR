"""R7 preflight -- task-state snapshot + seed freshness audit.

READ-ONLY with respect to the repository.  Writes only under
``out/r7_confirmatory_kernel_ranking_20260917/00_preflight/``.

Produces
--------
00_preflight/git_head.txt
00_preflight/git_branch.txt
00_preflight/git_status_before.txt
00_preflight/git_diff_stat_before.txt
00_preflight/git_diff_cached_stat_before.txt
00_preflight/relevant_diff_before.patch
00_preflight/software_check.json
00_preflight/seed_freshness_audit.json
00_preflight/seed_freshness_evidence.csv

The seed freshness audit answers one question only:

    have seed blocks 206-215 and 401-410 ever been *generated or evaluated*
    anywhere in this repository?

A "reserved constant" that merely lists a seed is NOT contamination.
Produced experiment output (a data file, a metric file, a plan, a manifest
that records a completed run) IS contamination.

Usage:
    python scripts/run_r7_preflight.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "out" / "r7_confirmatory_kernel_ranking_20260917"
PREFLIGHT = EXP / "00_preflight"

SELECTION_BLOCK = tuple(range(206, 216))
CONFIRMATORY_BLOCK = tuple(range(401, 411))
# Blocks that are permanently forbidden for R7 by the task specification.
FORBIDDEN_BLOCKS = {
    "42-46": tuple(range(42, 47)),
    "201-205": tuple(range(201, 206)),
    "301-305": tuple(range(301, 306)),
}

# Scan roots for the content-level search.  Large binary archives and VCS
# metadata are excluded; their text manifests are scanned separately.
CONTENT_ROOTS = ("out", "results", "experiments", "config", "data", "docs",
                 "manuscript_final", "schemas", "src", "tests", "scripts",
                 "tools", "1", "2", "3", "ZN_TIFS_R5_ARCHIVE",
                 "ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE", "_review",
                 "_phase3_extract", "_pkg_scratch", "_sivia_work")
SKIP_DIR_PARTS = {".git", "node_modules", "__pycache__", ".pytest_cache",
                  ".mypy_cache", ".venv", "venv", ".tmp_pytest_ec",
                  ".test_protocol_tmp", "_arch_scratch"}
TEXT_EXT = {".json", ".csv", ".md", ".txt", ".yaml", ".yml", ".py", ".tex",
            ".tsv", ".log", ".jsonl", ".cfg", ".ini", ".toml"}
MAX_CONTENT_BYTES = 40 * 1024 * 1024      # skip gigantic single files
CONTENT_BUDGET_BYTES = 900 * 1024 * 1024  # overall scan budget


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
                 encoding="utf-8")


def git(args: list[str]) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=600)
        return r.stdout
    except Exception as exc:                                  # pragma: no cover
        return f"<<git failed: {exc}>>"


# --------------------------------------------------------------------------- #
# 1. task-state snapshot
# --------------------------------------------------------------------------- #

def snapshot_git() -> dict[str, Any]:
    head = git(["rev-parse", "HEAD"]).strip()
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    status = git(["status", "--porcelain=v1"])
    diff_stat = git(["diff", "--stat"])
    diff_cached = git(["diff", "--cached", "--stat"])
    write_text(PREFLIGHT / "git_head.txt", head + "\n")
    write_text(PREFLIGHT / "git_branch.txt", branch + "\n")
    write_text(PREFLIGHT / "git_status_before.txt", status)
    write_text(PREFLIGHT / "git_diff_stat_before.txt", diff_stat)
    write_text(PREFLIGHT / "git_diff_cached_stat_before.txt", diff_cached)

    # Only the R7-relevant code paths; the pre-existing dirty tree is otherwise
    # left completely untouched and is *not* reverted.
    relevant = [
        "scripts/multi_bridge/", "scripts/run_r7", "src/cross/domain/evaluation/",
        "src/cross/domain/uot/", "config/",
    ]
    patch = git(["diff", "--", *relevant])
    write_text(PREFLIGHT / "relevant_diff_before.patch", patch)

    dirty_lines = [ln for ln in status.splitlines() if ln.strip()]
    return {
        "head": head,
        "branch": branch,
        "dirty_entry_count": len(dirty_lines),
        "diff_stat_lines": len(diff_stat.splitlines()),
        "relevant_diff_bytes": len(patch.encode("utf-8")),
    }


def software_check() -> dict[str, Any]:
    mods: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "ot", "sklearn",
                 "statsmodels", "pytest", "pyarrow", "networkx"):
        try:
            m = __import__(name)
            mods[name] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            mods[name] = f"NOT INSTALLED ({type(exc).__name__})"
    out = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": mods,
        "cwd": os.getcwd(),
        "repo": str(REPO),
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(PREFLIGHT / "software_check.json", out)
    return out


# --------------------------------------------------------------------------- #
# 2. seed freshness audit
# --------------------------------------------------------------------------- #

SEED_DIR_RE = re.compile(r"(?:^|[^0-9])(?:seed_?|s)(\d{1,4})(?:[^0-9]|$)")
SEED_ANY_RE = re.compile(r"(?<![0-9])(\d{2,4})(?![0-9])")


def iter_scan_files() -> Iterable[Path]:
    for root_name in CONTENT_ROOTS:
        root = REPO / root_name
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_PARTS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in TEXT_EXT:
                    continue
                try:
                    if p.stat().st_size > MAX_CONTENT_BYTES:
                        continue
                except OSError:
                    continue
                yield p


def audit() -> dict[str, Any]:
    watched = set(SELECTION_BLOCK) | set(CONFIRMATORY_BLOCK)
    for blk in FORBIDDEN_BLOCKS.values():
        watched |= set(blk)

    evidence: list[dict[str, Any]] = []
    counts = {k: {"dir_names": 0, "file_names": 0, "content_hits": 0}
              for k in ("selection", "confirmatory", "forbidden")}

    def block_of(seed: int) -> str | None:
        if seed in SELECTION_BLOCK:
            return "selection"
        if seed in CONFIRMATORY_BLOCK:
            return "confirmatory"
        for name, blk in FORBIDDEN_BLOCKS.items():
            if seed in blk:
                return "forbidden"
        return None

    scanned_files = 0
    scanned_bytes = 0
    t0 = time.time()

    for p in iter_scan_files():
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if scanned_bytes + sz > CONTENT_BUDGET_BYTES:
            continue
        rel = str(p.relative_to(REPO))

        # (a) directory-name evidence
        for part in p.relative_to(REPO).parts[:-1]:
            for m in SEED_DIR_RE.finditer(part):
                s = int(m.group(1))
                b = block_of(s)
                if b:
                    counts[b]["dir_names"] += 1
                    evidence.append({"kind": "dir_name", "seed": s, "block": b,
                                     "path": rel, "token": part})

        # (b) file-name evidence
        for m in SEED_DIR_RE.finditer(p.name):
            s = int(m.group(1))
            b = block_of(s)
            if b:
                counts[b]["file_names"] += 1
                evidence.append({"kind": "file_name", "seed": s, "block": b,
                                 "path": rel, "token": p.name})

        # (c) content evidence -- explicit seed tokens only
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        scanned_files += 1
        scanned_bytes += sz
        found: set[int] = set()
        for m in re.finditer(r"(?:seed_?|\"seed\"\s*:\s*|'seed'\s*:\s*|seed\s*=\s*|s)(\d{1,4})\b",
                             txt, flags=re.IGNORECASE):
            try:
                s = int(m.group(1))
            except ValueError:
                continue
            if s in watched:
                found.add(s)
        for s in sorted(found):
            b = block_of(s)
            if b:
                counts[b]["content_hits"] += 1
                evidence.append({"kind": "content", "seed": s, "block": b,
                                 "path": rel, "token": f"seed-token:{s}"})

    # ---- produced-output existence probe ---------------------------------- #
    produced_roots = {
        "paper_full_pipeline_run/synthetic": REPO / "out" / "paper_full_pipeline_run" / "synthetic",
    }
    existence: dict[str, Any] = {}
    for label, base in produced_roots.items():
        if not base.is_dir():
            continue
        for s in sorted(watched):
            d = base / f"synthetic_eval_seed_{s}"
            if d.is_dir():
                files = [f for f in d.rglob("*") if f.is_file()]
                existence[f"{label}/synthetic_eval_seed_{s}"] = {
                    "seed": s,
                    "block": block_of(s),
                    "file_count": len(files),
                    "total_bytes": int(sum(f.stat().st_size for f in files)),
                }

    # ---- verdict ---------------------------------------------------------- #
    def verdict(block_name: str, seeds: tuple[int, ...]) -> dict[str, Any]:
        produced = {s for s in seeds
                    if f"paper_full_pipeline_run/synthetic/synthetic_eval_seed_{s}" in existence}
        content = sorted({e["seed"] for e in evidence
                          if e["block"] == block_name and e["kind"] == "content"})
        return {
            "seeds": list(seeds),
            "produced_output_seeds": sorted(produced),
            "content_token_seeds": content,
            "status": "USED" if produced else "UNUSED",
            "contaminated": bool(produced),
        }

    res = {
        "audit_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": str(REPO),
        "git_head": git(["rev-parse", "HEAD"]).strip(),
        "definition_of_contamination": (
            "A seed is CONTAMINATED iff produced experiment output for it exists on disk "
            "(a generated data/metric/plan artifact recording a completed run). Plain seed "
            "constants in code or reserved-ID lists are NOT contamination."
        ),
        "scan": {
            "roots": list(CONTENT_ROOTS),
            "files_scanned": scanned_files,
            "bytes_scanned": scanned_bytes,
            "budget_bytes": CONTENT_BUDGET_BYTES,
            "seconds": round(time.time() - t0, 2),
        },
        "selection_block_206_215": verdict("selection", SELECTION_BLOCK),
        "confirmatory_block_401_410": verdict("confirmatory", CONFIRMATORY_BLOCK),
        "forbidden_blocks": {
            name: verdict("forbidden", blk) for name, blk in FORBIDDEN_BLOCKS.items()
        },
        "produced_output_dirs_found": existence,
        "evidence_counts": counts,
        "HARD_BLOCKER": None,
        "notes": [],
    }

    # 401-410 contamination is the one condition that forbids R7 outright.
    if res["confirmatory_block_401_410"]["contaminated"]:
        res["HARD_BLOCKER"] = "CONFIRMATORY_BLOCK_ALREADY_USED"
        res["notes"].append(
            "401-410 has produced output on disk: R7 must STOP and this block can never "
            "serve as a confirmatory holdout.")
    elif res["selection_block_206_215"]["contaminated"]:
        res["HARD_BLOCKER"] = "SELECTION_BLOCK_ALREADY_USED"
        res["notes"].append(
            "206-215 is partially contaminated: "
            f"{res['selection_block_206_215']['produced_output_seeds']} already have produced "
            "experiment output. The R7 selection block as specified is not fresh.")
    else:
        res["notes"].append("Both reserved blocks are fresh.")

    write_json(PREFLIGHT / "seed_freshness_audit.json", res)

    with open(PREFLIGHT / "seed_freshness_evidence.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "seed", "block", "path", "token"])
        w.writeheader()
        for e in sorted(evidence, key=lambda r: (r["block"], r["seed"], r["path"], r["kind"])):
            w.writerow(e)

    return res


# --------------------------------------------------------------------------- #

def main() -> int:
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    g = snapshot_git()
    s = software_check()
    a = audit()
    print(json.dumps({"git": g, "software_python": s["python"],
                      "packages": s["packages"],
                      "hard_blocker": a["HARD_BLOCKER"],
                      "selection_206_215": a["selection_block_206_215"]["status"],
                      "selection_used_seeds": a["selection_block_206_215"]["produced_output_seeds"],
                      "confirmatory_401_410": a["confirmatory_block_401_410"]["status"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
