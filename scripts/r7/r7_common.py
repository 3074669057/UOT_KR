"""R7 shared machinery: paths, frozen constants, seed guards, hashing, metrics.

Nothing in this module writes to, or reads data for, a forbidden seed.  The seed guards
are evaluated at import time for every constant block and at call time for every entry
point that can reach data.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
MB = REPO / "scripts" / "multi_bridge"
for _p in (str(SRC), str(MB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXP = REPO / "out" / "r7_confirmatory_kernel_ranking_20260917"
DIR_PREFLIGHT = EXP / "00_preflight"
DIR_SELECTION = EXP / "selection"
DIR_DEGREE = DIR_SELECTION / "degree_calibration"
DIR_GENERATOR = DIR_SELECTION / "generator"
DIR_RULES = DIR_SELECTION / "rule_search"
DIR_CONFIG = EXP / "config"
DIR_CONFIRMATORY = EXP / "confirmatory"
DIR_RAW = DIR_CONFIRMATORY / "raw"
DIR_ANALYSIS = EXP / "analysis"
DIR_DIAGNOSTICS = EXP / "diagnostics"
DIR_FIGURES = EXP / "figures"
DIR_PAPER = EXP / "paper"
DIR_LOGS = EXP / "logs"
DIR_PROVENANCE = EXP / "provenance"

LOG_PATH = DIR_LOGS / "r7.log"

# Frozen upstream artefacts (READ ONLY -- never written by R7)
FROZEN_FEATURE_STATS = (REPO / "out" / "multi_bridge_expansion"
                        / "faithful_flow_structural_three_bridges" / "feature_stats")
V4_ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v4"
V5_ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v5"

# --------------------------------------------------------------------------- #
# Frozen experiment constants
# --------------------------------------------------------------------------- #
EXPERIMENT_ID = "r7_confirmatory_kernel_ranking_20260917"
BRIDGES = ("Celer", "Multi", "Poly")

# Seed blocks.  AMENDED 2026-09-17: the originally reserved selection block 206-215 was
# found partially contaminated (212-215 already carry produced output from
# out/paper_full_pipeline_run/synthetic/).  See BLOCKER_REPORT.md and
# 00_preflight/seed_freshness_audit.json.  The amendment keeps the fresh part of the
# originally reserved block (206-211) and fills the remainder with the next fresh seeds.
SELECTION_SEEDS = (206, 207, 208, 209, 210, 211, 112, 113, 114, 115)
SELECTION_SEEDS_AS_SPECIFIED = (206, 207, 208, 209, 210, 211, 212, 213, 214, 215)

# Confirmatory block.  AMENDED 2026-09-17: the originally frozen block 401-410 was SPENT
# by a first confirmatory execution that crashed on a plumbing defect after its O_EXCL
# touch ledger was written and after Celer/401 had been generated, but before any unit
# result was written or observed.  Per the specification a touched holdout is never
# re-run; a fresh, never-touched block is used instead with the protocol unchanged.
# See confirmatory/INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION.md.
CONFIRMATORY_SEEDS = (411, 412, 413, 414, 415, 416, 417, 418, 419, 420)
RETIRED_CONFIRMATORY_BLOCKS = (
    {"block": [401, 410],
     "reason": "INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION (plumbing defect, no result produced)",
     "ledger": "confirmatory/CONFIRMATORY_TOUCH_ONCE.json",
     "reused": False},
)

FORBIDDEN_SEEDS = frozenset(
    set(range(42, 47)) | set(range(201, 206)) | set(range(301, 306))
    | set(range(212, 216))          # contaminated 2026-09-17, never usable as selection
    | set(range(401, 411))          # SPENT confirmatory block, never reusable
)

# Frozen cost / solver configuration -- NOT re-opened in R7 (specification section 12).
EPSILON = 0.05                     # UOT entropic regularisation (reg)
LAMBDA = 0.5                       # UOT marginal relaxation (reg_m)
COST_WEIGHTS_ABSOLUTE = {"time": 0.25, "route": 0.15, "risk": 0.15,
                         "evidence": 0.05, "novelty": 0.05}
COST_RENORM_DENOM = 0.65
SOLVER_NUM_ITER_MAX = 20000
SOLVER_STOP_THR = 1e-11
SOLVER_MARGINAL_TOL = 1e-7         # Gate D
SUPPORT_THRESHOLD = 1e-9           # hard support filter
K_MAX = 8                          # degree cap (specification section 7)
DEGREE_RANGE = (2, 8)

# Template-family design
N_FAMILIES = 24
INSTANCES_PER_FAMILY = 2
N_TEMPLATES_PER_CELL = N_FAMILIES * INSTANCES_PER_FAMILY      # 48, unchanged scale
AMOUNT_QUARTILES = 4
DELAY_STRATA = 6                   # sextiles; original design used tertiles

# Threshold-MM / Dual-Softmax calibration grids (locked in the candidate space, before
# any 206-215 method F1 is observed).
THRESHOLD_MM_Q_GRID = tuple(round(0.02 * k, 2) for k in range(1, 50))   # 0.02 .. 0.98
# Refined discretisation of the SAME equal-budget criterion.  The coarse grid's lower
# bound (q = 0.02) turned out to bind: at q = 0.02 the threshold rule already predicts
# ~80.6 edges per family against a UOT_KR target of ~17.4, because a mutual-top-k(k=3)
# decoder emits far fewer edges than any 2%-quantile cost cut.  The criterion
# ("minimise |budget(cutoff) - budget(UOT_KR)|") and the target are UNCHANGED; only the
# discretisation of the continuous threshold is refined so the minimum is actually
# attained.  This direction makes the H2 comparator STRONGER, i.e. conservative.
THRESHOLD_MM_Q_GRID_REFINED = (
    0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015,
) + THRESHOLD_MM_Q_GRID
# Dual-Softmax confidence grid.  The scale is structural: D = row_softmax(logK) *
# col_softmax(logK) over a ~289 x 289 matrix, so a confident mutual pair sits near
# 0.02-0.03 and not near 1.  Measured on frozen selection cells over
# mutual-nearest-neighbour pairs only (label-free): pooled D spans 2.9e-05 .. 0.151 with
# median 0.0236, p10 0.0137, p90 0.0277.  The grid below spans that range.  It was fixed
# from the SCORE SCALE, before any rule-selection F1 was computed.
DUAL_SOFTMAX_TAU_GRID = (0.0, 0.005, 0.010, 0.015, 0.020, 0.024, 0.030, 0.050)
DUAL_SOFTMAX_ACCEPT = ("mutual_nn",)

# Statistics
N_BOOT = 4000
RNG_BOOTSTRAP = 20240101
N_PERM = 20000
RNG_PERMUTATION = 20240102
ALPHA = 0.05
TIE_TOL = 1e-12
GATE_C_FLOOR = -0.005

METHODS = ("UOT_KR", "RAW_UOT_PLAN", "CONDITIONAL_UOT", "SUPPORT_PLUS_K",
           "HUNGARIAN_1TO1", "THRESHOLD_MM", "DUAL_SOFTMAX")
SECONDARY_METHODS = ("ORACLE_1TO1_CEILING",)      # diagnostic, never a baseline


# --------------------------------------------------------------------------- #
# Seed guards -- evaluated before ANY data access
# --------------------------------------------------------------------------- #
class SeedGuardViolation(SystemExit):
    """Raised when a forbidden seed reaches a data-touching entry point."""


def assert_seeds_allowed(seeds: Iterable[int], context: str = "") -> None:
    """Hard guard.  Only the two frozen R7 blocks may ever reach data code."""
    seeds = [int(s) for s in seeds]
    bad_forbidden = sorted(set(seeds) & FORBIDDEN_SEEDS)
    if bad_forbidden:
        raise SeedGuardViolation(
            f"R7 SEED GUARD: forbidden seed(s) {bad_forbidden} requested "
            f"{('in ' + context) if context else ''}. Seeds 42-46 (historical stress), "
            f"201-205 (R5/R6 dev), 301-305 (round-1 frozen holdout) and 212-215 "
            f"(contaminated) are permanently excluded from R7.")
    unknown = sorted(set(seeds) - set(SELECTION_SEEDS) - set(CONFIRMATORY_SEEDS))
    if unknown:
        raise SeedGuardViolation(
            f"R7 SEED GUARD: seed(s) {unknown} are not in any R7 block "
            f"{('in ' + context) if context else ''}. Allowed: selection "
            f"{list(SELECTION_SEEDS)} or confirmatory {list(CONFIRMATORY_SEEDS)}.")


def assert_selection_seeds(seeds: Iterable[int], context: str = "") -> None:
    assert_seeds_allowed(seeds, context)
    off = sorted(set(int(s) for s in seeds) - set(SELECTION_SEEDS))
    if off:
        raise SeedGuardViolation(
            f"R7 SELECTION GUARD: {off} outside the selection block "
            f"{list(SELECTION_SEEDS)} {('in ' + context) if context else ''}.")


def assert_confirmatory_seeds(seeds: Iterable[int], context: str = "") -> None:
    assert_seeds_allowed(seeds, context)
    off = sorted(set(int(s) for s in seeds) - set(CONFIRMATORY_SEEDS))
    if off:
        raise SeedGuardViolation(
            f"R7 CONFIRMATORY GUARD: {off} outside the confirmatory block "
            f"{list(CONFIRMATORY_SEEDS)} {('in ' + context) if context else ''}.")


def assert_bridges(bridges: Iterable[str]) -> None:
    off = sorted(set(bridges) - set(BRIDGES))
    if off:
        raise SeedGuardViolation(f"R7 BRIDGE GUARD: unknown bridge(s) {off}.")


# --------------------------------------------------------------------------- #
# Hashing / IO
# --------------------------------------------------------------------------- #
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return sha256_bytes(a.tobytes())


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_obj(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_code(paths: Iterable[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(Path(x) for x in paths):
        if p.is_file():
            out[str(p.relative_to(REPO))] = sha256_file(p)
    return out


def resolve_frozen_path(rel: str) -> Path:
    """Resolve a frozen-manifest key.

    Code lives under the repository root (``scripts/``, ``src/``); R7 artefacts live under
    the experiment output directory.  The manifest stores both under one key space, so
    resolution tries the repository root first and then the experiment root.
    """
    p = REPO / rel
    if p.is_file():
        return p
    return EXP / rel


def frozen_hash(rel: str) -> str | None:
    p = resolve_frozen_path(rel)
    return sha256_file(p) if p.is_file() else None


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                 encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def log(msg: str, also_print: bool = True) -> None:
    line = f"[{utc_now()}] {msg}"
    if also_print:
        print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def git(args: list[str]) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=600)
        return r.stdout
    except Exception as exc:                                   # pragma: no cover
        return f"<<git failed: {exc}>>"


def environment_record() -> dict[str, Any]:
    mods: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "ot", "sklearn"):
        try:
            m = __import__(name)
            mods[name] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            mods[name] = f"NOT INSTALLED ({type(exc).__name__})"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "packages": mods,
        "git_head": git(["rev-parse", "HEAD"]).strip(),
        "git_branch": git(["rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "utc": utc_now(),
    }


# --------------------------------------------------------------------------- #
# Template identity helpers (shared with the frozen evaluator)
# --------------------------------------------------------------------------- #
def tpl_of(flow_id: str) -> str:
    """Template key of a flow id -- frozen definition from baseline_mechanism.common."""
    s = str(flow_id)
    if "__synth" in s:
        return s.split("__synth")[0]
    if "__" in s:
        return s.rsplit("__", 1)[0]
    return s


def family_of_template(template_id: str) -> str:
    """R7 family key of a template id.

    R7 template ids are ``<anchor>__r7fam<FF>__inst<II>``; the family key is the
    grid slot ``r7fam<FF>`` (00..23), shared by every instance and every anchor that
    belongs to that family.  Frozen (non-R7) template ids pass through unchanged.
    """
    s = str(template_id)
    if "__r7fam" in s:
        return "r7fam" + s.split("__r7fam", 1)[1].split("__inst", 1)[0]
    return s


def instance_of_template(template_id: str) -> str:
    s = str(template_id)
    if "__inst" in s:
        return s.split("__inst", 1)[1]
    return "00"


def family_index_of_template(template_id: str) -> int:
    fam = family_of_template(template_id)
    if fam.startswith("r7fam"):
        return int(fam[5:])
    return -1
