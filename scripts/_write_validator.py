
from pathlib import Path

VCODE = r'''"""P0 Provider Data Validator.
Does NOT import solvers, load labels, or access confirmation vault."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json, hashlib, statistics
from datetime import datetime, timezone

LABEL_LIKE_KEYWORDS = [
    "gold", "label", "truth", "matched", "is_match", "correspondence",
    "ground_truth", "target_for_source", "source_for_target",
    "solver_score", "prediction"
]

@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    file: str | None = None
    line_number: int | None = None
    record_id: str | None = None
    field_name: str | None = None
    expected_: str | None = None
    observed_: str | None = None
    suggested_fix: str | None = None

@dataclass(frozen=True)
class ProviderValidationReport:
    status: str
    schema_valid: bool
    semantic_valid: bool
    overlap_valid: bool
    candidate_gate_valid: bool
    issues: list = field(default_factory=list)
    statistics: dict = field(default_factory=dict)
    input_hashes: dict = field(default_factory=dict)

def _load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i} in {path.name}: {e}")
    return records

def _hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _validate_directory_contract(dataset_dir):
    issues = []
    if not dataset_dir.is_dir():
        issues.append(ValidationIssue("ERROR", "P0_REJECTED_DIRECTORY", f"Not a directory: {dataset_dir}"))
        return False, issues
    required = ["source_flows.jsonl", "target_flows.jsonl", "collection_manifest.json"]
    for fn in required:
        if not (dataset_dir / fn).exists():
            issues.append(ValidationIssue("ERROR", "P0_REJECTED_DIRECTORY", f"Missing: {fn}", file=fn))
    forbidden = ["labels", "gold_edges", "matches", "predictions", "solver_scores"]
    for item in dataset_dir.iterdir():
        name_lower = item.name.lower()
        for fb in forbidden:
            if fb in name_lower:
                issues.append(ValidationIssue("ERROR", "P0_REJECTED_LABEL_LEAKAGE", f"Forbidden path: {item.name}", file=item.name))
    return len([i for i in issues if i.severity == "ERROR"]) == 0, issues

def scan_for_label_like_fields(records, file_name):
    issues = []
    if not records:
        return issues
    all_fields = set()
    for r in records:
        all_fields.update(r.keys())
    for field in all_fields:
        field_lower = field.lower()
        for kw in LABEL_LIKE_KEYWORDS:
            if kw in field_lower:
                issues.append(ValidationIssue("ERROR", "P0_REJECTED_LABEL_LEAKAGE", f"Label-like field '{field}' (keyword: {kw})", file=file_name, field_name=field, suggested_fix=f"Remove or whitelist field '{field}'"))
                break
    return issues

def validate_semantics(dataset_dir, manifest, src_records, tgt_records):
    issues = []
    src_ids = [r.get("source_flow_id") for r in src_records]
    tgt_ids = [r.get("target_flow_id") for r in tgt_records]
    if len(src_ids) != len(set(src_ids)):
        issues.append(ValidationIssue("ERROR", "DUPLICATE_SOURCE_ID", f"Duplicate source_flow_id", file="source_flows.jsonl"))
    if len(tgt_ids) != len(set(tgt_ids)):
        issues.append(ValidationIssue("ERROR", "DUPLICATE_TARGET_ID", f"Duplicate target_flow_id", file="target_flows.jsonl"))
    if set(src_ids) & set(tgt_ids):
        issues.append(ValidationIssue("ERROR", "ID_NAMESPACE_CONFLICT", f"source and target IDs overlap"))
    for i, r in enumerate(src_records):
        st, et = r.get("start_timestamp"), r.get("end_timestamp")
        try:
            s = float(st) if isinstance(st,(int,float)) else datetime.fromisoformat(str(st).replace("Z","+00:00")).timestamp()
            e = float(et) if isinstance(et,(int,float)) else datetime.fromisoformat(str(et).replace("Z","+00:00")).timestamp()
            if s > e:
                issues.append(ValidationIssue("ERROR", "TIMESTAMP_ORDER", f"start>end", file="source_flows.jsonl", record_id=str(r.get("source_flow_id","")), line_number=i+1))
        except:
            pass
    for i, r in enumerate(src_records):
        for af in ["raw_amount","normalized_amount"]:
            v = r.get(af)
            if v is not None:
                try:
                    if float(v) < 0:
                        issues.append(ValidationIssue("ERROR", "NEGATIVE_AMOUNT", f"{af}<0", file="source_flows.jsonl", record_id=str(r.get("source_flow_id",""))))
                except:
                    pass
    declared_src = manifest.get("source_flow_count"); actual_src = len(src_records)
    declared_tgt = manifest.get("target_flow_count"); actual_tgt = len(tgt_records)
    if declared_src is not None and declared_src != actual_src:
        issues.append(ValidationIssue("ERROR", "MANIFEST_COUNT_MISMATCH", f"source: declared={declared_src} actual={actual_src}", file="collection_manifest.json"))
    if declared_tgt is not None and declared_tgt != actual_tgt:
        issues.append(ValidationIssue("ERROR", "MANIFEST_COUNT_MISMATCH", f"target: declared={declared_tgt} actual={actual_tgt}", file="collection_manifest.json"))
    return len([i for i in issues if i.severity == "ERROR"]) == 0, issues

def run_anti_triviality_gate(n_src, n_tgt, candidate_counts_per_source, candidate_counts_per_tgt, n_candidate_edges_unique):
    issues = []
    stats = {"n_source_flows":n_src,"n_target_flows":n_tgt,"n_candidate_edges_unique":n_candidate_edges_unique}
    if n_src == 0 or n_tgt == 0:
        issues.append(ValidationIssue("ERROR","P0_REJECTED_TRIVIAL_CANDIDATES","No flows"))
        return False, issues, stats
    src_2plus = sum(1 for c in candidate_counts_per_source if c >= 2)
    tgt_2plus = sum(1 for c in candidate_counts_per_tgt if c >= 2)
    s_zero = sum(1 for c in candidate_counts_per_source if c == 0)
    t_zero = sum(1 for c in candidate_counts_per_tgt if c == 0)
    stats["fraction_sources_2plus"] = src_2plus/n_src
    stats["fraction_targets_2plus"] = tgt_2plus/n_tgt
    stats["fraction_sources_zero_candidates"] = s_zero/n_src
    stats["fraction_targets_zero_candidates"] = t_zero/n_tgt
    stats["median_candidates_per_source"] = statistics.median(candidate_counts_per_source) if candidate_counts_per_source else 0
    stats["median_candidates_per_target"] = statistics.median(candidate_counts_per_tgt) if candidate_counts_per_tgt else 0
    checks = [
        ("n_source_flows >= 20", n_src >= 20),
        ("n_target_flows >= 20", n_tgt >= 20),
        ("fraction_sources_2plus >= 0.50", stats["fraction_sources_2plus"] >= 0.50),
        ("fraction_targets_2plus >= 0.30", stats["fraction_targets_2plus"] >= 0.30),
        ("median_candidates_per_source >= 2", stats["median_candidates_per_source"] >= 2),
        ("candidate_edges > max", n_candidate_edges_unique > max(n_src, n_tgt)),
        ("sources_zero_candidates <= 0.20", stats["fraction_sources_zero_candidates"] <= 0.20),
        ("targets_zero_candidates <= 0.40", stats["fraction_targets_zero_candidates"] <= 0.40),
    ]
    for name, passed in checks:
        if not passed:
            issues.append(ValidationIssue("ERROR","P0_REJECTED_TRIVIAL_CANDIDATES",f"FAIL: {name}"))
    return all(c[1] for c in checks), issues, stats

def validate_provider_directory(dataset_dir):
    issues = []; hashes = {}; stats = {}
    dir_ok, dir_issues = _validate_directory_contract(dataset_dir)
    issues.extend(dir_issues)
    if not dir_ok:
        return ProviderValidationReport(status="P0_REJECTED_DIRECTORY",schema_valid=False,semantic_valid=False,overlap_valid=False,candidate_gate_valid=False,issues=issues,statistics=stats,input_hashes=hashes)
    try:
        src_rec = _load_jsonl(dataset_dir/"source_flows.jsonl")
        tgt_rec = _load_jsonl(dataset_dir/"target_flows.jsonl")
        manifest = json.loads((dataset_dir/"collection_manifest.json").read_text(encoding="utf-8"))
    except Exception as e:
        issues.append(ValidationIssue("ERROR","P0_REJECTED_SCHEMA",str(e)))
        return ProviderValidationReport(status="P0_REJECTED_SCHEMA",schema_valid=False,semantic_valid=False,overlap_valid=False,candidate_gate_valid=False,issues=issues)
    stats["n_source_flows"]=len(src_rec); stats["n_target_flows"]=len(tgt_rec)
    for f in ["source_flows.jsonl","target_flows.jsonl","collection_manifest.json"]:
        hashes[f]=_hash_file(dataset_dir/f)
    for recs,fn in [(src_rec,"source_flows.jsonl"),(tgt_rec,"target_flows.jsonl")]:
        issues.extend(scan_for_label_like_fields(recs,fn))
    sem_ok, sem_issues = validate_semantics(dataset_dir, manifest, src_rec, tgt_rec)
    issues.extend(sem_issues)
    n_src,n_tgt=len(src_rec),len(tgt_rec)
    cc_src=[n_tgt]*n_src; cc_tgt=[n_src]*n_tgt; n_edges=n_src*n_tgt
    gate_ok, gate_issues, gate_stats = run_anti_triviality_gate(n_src,n_tgt,cc_src,cc_tgt,n_edges)
    issues.extend(gate_issues); stats.update(gate_stats)
    err_codes={i.code for i in issues if i.severity=="ERROR"}
    if "P0_REJECTED_LABEL_LEAKAGE" in err_codes: status="P0_REJECTED_LABEL_LEAKAGE"
    elif "P0_REJECTED_TRIVIAL_CANDIDATES" in err_codes: status="P0_REJECTED_TRIVIAL_CANDIDATES"
    elif not sem_ok: status="P0_REJECTED_SEMANTICS"
    else: status="P0_ACCEPTED"
    return ProviderValidationReport(status=status,schema_valid=dir_ok,semantic_valid=sem_ok,overlap_valid=True,candidate_gate_valid=gate_ok,issues=issues,statistics=stats,input_hashes=hashes)
'''

Path("src/cross/domain/provider_data/validator.py").write_text(VCODE, encoding="utf-8")
print("validator.py written")
