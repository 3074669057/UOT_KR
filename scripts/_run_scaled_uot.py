import sys, json, time
from pathlib import Path
REPO = Path(r"<REPO>")
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from cross.application.standalone_flow_uot import run_standalone_flow_uot

AGG_DIR = REPO / "out/bsc_open_independent_v1/stage5_6_flow_aggregation"
OUT_DIR = AGG_DIR / "rc_uot_results" / "w3h_scaled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SRC_FLOWS = AGG_DIR / "src_flows_aggregated.csv"
DST_FLOWS = AGG_DIR / "dst_flows_aggregated.csv"
FLOW_LABELS = AGG_DIR / "flow_labels_aggregated.csv"

COST_WEIGHTS = {"amount": 0.35, "time": 0.25, "route": 0.15, "risk": 0.15, "graph": 0.05, "evidence": 0.05}

print("=== Revised RC-UOT: scaled params for entity-level ===")
print(f"  max_matrix_cells=15M, top_k=4000, max_delay=86400s (24h)")
t0 = time.time()

result = run_standalone_flow_uot(
    out_dir=OUT_DIR,
    src_flows_csv=SRC_FLOWS,
    dst_flows_csv=DST_FLOWS,
    flow_labels_csv=FLOW_LABELS,
    uot_reg=0.05,
    uot_reg_m=0.5,
    uot_decode_threshold=0.05,
    uot_cost_weights=COST_WEIGHTS,
    uot_backend="numpy",
    uot_max_delay_sec=86400.0,
    uot_causal_violation_penalty=5.0,
    uot_lambda_risk=0.25,
    uot_causal_infeasible_delay_sec=None,
    uot_export_matrix=False,
    uot_export_cost_components=False,
    uot_allow_unmatched=True,
    uot_use_graph_embedding=False,
    graph_ranker_checkpoint=None,
    uot_ablation="none",
    run_flow_baselines=False,
    flow_label_min_confidence=0.0,
    uot_flow_dst_top_k=4000,
    uot_flow_max_matrix_cells=15_000_000,
    uot_pool_strategy="default",
)

elapsed = time.time() - t0
print(f"Elapsed: {elapsed:.1f}s")

# Check results
metrics_path = OUT_DIR / "eval" / "uot_evaluation_metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        m = json.load(f)
    print(f"F1: {m.get('flow_pair_f1')}")
    print(f"Precision: {m.get('flow_pair_precision')}")
    print(f"Recall: {m.get('flow_pair_recall')}")
    print(f"Mass recall: {m.get('flow_mass_recall')}")
    print(f"Top1: {m.get('top1_flow_correspondence_accuracy')}")
    print(f"Edges: {m.get('num_predicted_edges')}")
    print(f"TP: {m.get('num_true_positive_edges')}")

# Check diagnostics  
import pandas as pd
diag_path = OUT_DIR / "uot" / "uot_diagnostics.json"
if diag_path.exists():
    with open(diag_path) as f:
        diag = json.load(f)
    meta = diag.get("transport_graph_meta", {})
    print(f"\nCandidate recall: {meta.get('candidate_dst_recall')}")
    print(f"Matrix cells: {meta.get('matrix_cells')}")
    print(f"Dst in subgraph: {meta.get('n_bnb_active')}")
    print(f"Truth dst in subgraph: {meta.get('truth_dst_in_subgraph')} / {meta.get('truth_dst_unique_in_segments')}")

plan_path = OUT_DIR / "uot" / "uot_transport_plan.csv"
if plan_path.exists():
    plan = pd.read_csv(plan_path)
    print(f"\nTransport plan: {len(plan)} edges, mass={plan['transport_mass'].sum():.4f}")
    print(f"Unique src: {plan['src_flow_id'].nunique()}, dst: {plan['dst_flow_id'].nunique()}")
