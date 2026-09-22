"""Gate 3.5: Split independence audit ? build independence groups and check cross-split contamination."""
import sys, json, csv, hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.application.experiments.run_admissible_decoding import _truth_from_labels
from cross.shared.normalize import norm_addr

# Load
uot_prod = _REPO / "out" / "uot_delay_fixed_production"
eth_flows = load_flow_segments(uot_prod / "uot" / "uot_flow_segments_eth.csv")
bnb_flows = load_flow_segments(uot_prod / "uot" / "uot_flow_segments_bnb.csv")
label_csv = _REPO / "out" / "baseline_compare" / "labels" / "gt_tx_pairs.csv"
ld = pd.read_csv(label_csv)
ld = ld.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})
truth = _truth_from_labels(ld)

# Build lookup
tx_to_src_flow = {}
for i, sf in enumerate(eth_flows):
    for txh in sf.get("tx_hashes") or []:
        tx_to_src_flow[norm_addr(str(txh))] = i

tx_to_dst_flow = {}
for j, tf in enumerate(bnb_flows):
    for txh in tf.get("tx_hashes") or []:
        tx_to_dst_flow.setdefault(norm_addr(str(txh)), set()).add(j)

# Build gold connected components (connected by shared source or target flow)
# Also connect through anchor pairs
src_to_components = defaultdict(set)
dst_to_components = defaultdict(set)

component_id = 0
pair_to_component = {}

for st, dt in truth.items():
    si = tx_to_src_flow.get(st, -1)
    djs = tx_to_dst_flow.get(dt, set())
    if si < 0 or not djs:
        continue
    for dj in djs:
        pair_to_component[(si, dj)] = -1  # placeholder

# Union-find on (source_flow, dst_flow) pairs
parent = {}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry:
        parent[rx] = ry

# Initialize
all_flows = set()
for (si, dj) in pair_to_component:
    all_flows.add(("s", si))
    all_flows.add(("d", dj))
for f in all_flows:
    parent[f] = f

# Connect flows through anchor pairs
for (si, dj) in pair_to_component:
    union(("s", si), ("d", dj))

# Also connect destination flows that share source flow
src_to_dsts = defaultdict(set)
for (si, dj) in pair_to_component:
    src_to_dsts[si].add(dj)
for si, djs in src_to_dsts.items():
    dj_list = list(djs)
    for i in range(len(dj_list)):
        for j in range(i+1, len(dj_list)):
            union(("d", dj_list[i]), ("d", dj_list[j]))

# Assign component IDs
comp_to_id = {}
next_id = 0
for f in all_flows:
    root = find(f)
    if root not in comp_to_id:
        comp_to_id[root] = next_id
        next_id += 1

# Map (src_flow, dst_flow) pair to component
pair_comp = {}
for (si, dj) in pair_to_component:
    root = find(("s", si))
    pair_comp[(si, dj)] = comp_to_id[root]

# Map anchor pairs to components
anchor_to_comp = {}
for st, dt in truth.items():
    si = tx_to_src_flow.get(st, -1)
    djs = tx_to_dst_flow.get(dt, set())
    if si < 0 or not djs:
        continue
    for dj in djs:
        if (si, dj) in pair_comp:
            anchor_to_comp[(st, dt)] = pair_comp[(si, dj)]
            break

n_components = len(comp_to_id)
n_anchors = len(anchor_to_comp)
print(f"Independence groups: {n_components}")
print(f"Anchors with component: {n_anchors}")

# Build independence groups: each component => set of target_flow indices
comp_to_target_flows = defaultdict(set)
for (si, dj), cid in pair_comp.items():
    comp_to_target_flows[cid].add(dj)

# Check split assignment
n_total = max(len(eth_flows), len(bnb_flows))
rng = np.random.RandomState(42)
split_ids = []
for i in range(n_total):
    h = hashlib.sha256(f"flow_{i}_seed_42".encode()).hexdigest()
    split_ids.append(int(h[:8], 16) % 100)

dev_set = set(i for i, s in enumerate(split_ids) if s < 60)
val_set = set(i for i, s in enumerate(split_ids) if 60 <= s < 80)
pilot_set = set(i for i, s in enumerate(split_ids) if s >= 80)

# Check cross-split: does any component have flows in more than one split?
cross_split = defaultdict(set)
for cid, tfs in comp_to_target_flows.items():
    splits = set()
    for tf in tfs:
        if tf in dev_set: splits.add("dev")
        if tf in val_set: splits.add("val")
        if tf in pilot_set: splits.add("pilot")
    if len(splits) > 1:
        cross_split[cid] = splits

print(f"\nCross-split components: {len(cross_split)}")
for cid, splits in list(cross_split.items())[:10]:
    print(f"  Component {cid}: in {splits}")

# Find clean confirmation: components entirely in val_set and never in dev or pilot
clean_val_comps = []
for cid, tfs in comp_to_target_flows.items():
    tf_set = set(tfs)
    if tf_set.issubset(val_set) and not tf_set.intersection(dev_set) and not tf_set.intersection(pilot_set):
        clean_val_comps.append(cid)

print(f"\nClean confirmation components: {len(clean_val_comps)}")
clean_val_flows = set()
clean_val_anchors = 0
for cid in clean_val_comps:
    for tf in comp_to_target_flows[cid]:
        clean_val_flows.add(tf)
for (st, dt), cid in anchor_to_comp.items():
    if cid in clean_val_comps:
        clean_val_anchors += 1

print(f"Clean confirmation flows: {len(clean_val_flows)}")
print(f"Clean confirmation anchor pairs: {clean_val_anchors}")

# Check: are ANY val flows used in dev or pilot?
val_in_dev = val_set & dev_set
val_in_pilot = val_set & pilot_set
print(f"\nVal-dev overlap: {len(val_in_dev)}")
print(f"Val-pilot overlap: {len(val_in_pilot)}")

# Write independence group manifest
manifest_path = _REPO / "out" / "rc_uot_v2_study" / "protocol" / "independence_group_manifest.jsonl"
with open(manifest_path, "w", encoding="utf-8") as f:
    for cid in sorted(comp_to_target_flows.keys()):
        flows = sorted(comp_to_target_flows[cid])
        splits_in = set()
        for tf in flows:
            if tf in dev_set: splits_in.add("dev")
            if tf in val_set: splits_in.add("val")
            if tf in pilot_set: splits_in.add("pilot")
        f.write(json.dumps({
            "independence_group_id": cid,
            "n_target_flows": len(flows),
            "target_flow_indices": flows,
            "splits": sorted(splits_in),
            "cross_split": len(splits_in) > 1,
        }) + "\n")

# Write split independence audit
audit_path = _REPO / "out" / "rc_uot_v2_study" / "audit" / "split_independence_audit.json"
audit = {
    "n_independence_groups": n_components,
    "n_anchor_pairs_with_component": n_anchors,
    "cross_split_components": len(cross_split),
    "cross_split_rate": len(cross_split) / max(n_components, 1),
    "clean_confirmation_components": len(clean_val_comps),
    "clean_confirmation_flows": len(clean_val_flows),
    "clean_confirmation_anchors": clean_val_anchors,
    "sealed_confirmation_eligible": len(clean_val_comps) >= 50,
    "original_val_size": len(val_set),
    "clean_val_fraction": len(clean_val_flows) / max(len(val_set), 1),
    "decision_ceiling": "Case D" if len(clean_val_comps) < 50 else "Case A/B/C",
}
with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
print(f"\nAudit written. Eligible: {audit['sealed_confirmation_eligible']}")

# Write overlap examples
if cross_split:
    overlap_path = _REPO / "out" / "rc_uot_v2_study" / "protocol" / "split_overlap_examples.csv"
    with open(overlap_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["component_id", "splits", "n_target_flows"])
        for cid, splits in sorted(cross_split.items()):
            w.writerow([cid, ",".join(sorted(splits)), len(comp_to_target_flows[cid])])
