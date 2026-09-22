import json, hashlib, sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from cross.shared.normalize import norm_addr

OUT_DIR = REPO / 'out' / 'bsc_open_independent_v1' / 'stage5_6_flow_aggregation'
RAW_RELAY = REPO / 'out' / 'bsc_open_independent_v1' / 'stage5_5_candidate_completion' / 'raw_relay_logs_dev_v2.json'
ETH_CSV = REPO / 'data' / 'in' / 'Celer_ETH_cun.csv'
LABEL_CSV = REPO / 'data' / 'label' / 'celer_label.csv'
SPLIT_CSV = REPO / 'out' / 'bsc_open_independent_v1' / 'stage2_split' / 'chronological_split_assignments.csv'

OUT_DIR.mkdir(parents=True, exist_ok=True)


def _utc():
    return datetime.now(timezone.utc).isoformat()

def _safe_float(v, default=0.0):
    try:
        s = str(v).strip()
        return float(int(s, 16)) if s.startswith('0x') else float(s)
    except:
        return default

def decode_relay_events(raw_path):
    with open(raw_path) as f:
        logs = json.load(f)
    events = []
    for log in logs:
        tx_hash = log['transactionHash'].lower()
        ts = int(log['blockTimestamp'], 16)
        d = log['data'][2:]
        # 7 params x 32 bytes: transferId|sender|receiver|token|amount|?|?
        receiver = '0x' + d[152:192]
        token = '0x' + d[216:256]
        amount = int('0x' + d[256:320], 16)
        events.append({
            'tx_hash': tx_hash,
            'receiver': norm_addr(receiver),
            'token': norm_addr(token),
            'amount': amount,
            'timestamp': ts,
        })
    return pd.DataFrame(events)

def load_eth_source_txs(dev_src_hashes):
    eth = pd.read_csv(ETH_CSV, dtype=str)
    eth['hl'] = eth['hash'].str.lower()
    eth = eth[eth['hl'].isin(dev_src_hashes)].copy()
    eth['sender'] = eth['from'].apply(norm_addr)
    eth['to_addr'] = eth['to'].apply(norm_addr)
    eth['token'] = eth['contractAddress'].apply(norm_addr)
    eth['amount'] = eth['value'].apply(_safe_float)
    eth['timestamp'] = eth['timeStamp'].astype(int)
    return eth[['hl', 'sender', 'to_addr', 'token', 'amount', 'timestamp']]

def load_dev_anchor_pairs():
    labels = pd.read_csv(LABEL_CSV)
    labels['sh'] = labels['srcTxhash'].str.lower()
    labels['dh'] = labels['dstTxhash'].str.lower()
    split = pd.read_csv(SPLIT_CSV)
    dev_h = set(split[split['split'] == 'development']['source_tx_hash'].str.lower())
    return labels[labels['sh'].isin(dev_h)].copy()


def aggregate_by_window(df, group_col, time_col, window_sec):
    df = df.copy()
    df['_window'] = (df[time_col] // window_sec) * window_sec

    # Determine the tx hash column
    if 'tx_hash' in df.columns:
        tx_col = 'tx_hash'
    elif 'hl' in df.columns:
        tx_col = 'hl'
    else:
        tx_col = df.columns[0]

    segments = []
    tx_to_seg = {}

    for keys, grp in df.groupby([group_col, '_window'], sort=False):
        group_val, win = keys if isinstance(keys, tuple) else (keys, 0)
        txs = grp[tx_col].tolist()
        seg_id = f'{group_col}_{str(group_val)[:12]}_{int(win)}_{len(segments):06d}'

        amounts = grp['amount'].tolist()
        timestamps = grp[time_col].tolist()

        # Collect addresses from all available columns
        all_addrs = set()
        for col in ['sender', 'receiver', 'token', 'to_addr']:
            if col in grp.columns:
                for v in grp[col]:
                    s = str(v).strip()
                    if s and s not in ('nan', 'None', ''):
                        all_addrs.add(s)

        seg = {
            'flow_id': seg_id,
            'tx_hashes': txs,
            'tx_count': len(txs),
            'amount_usd': float(sum(amounts)),
            'raw_amount_sum': float(sum(amounts)),
            'start_time': int(min(timestamps)),
            'end_time': int(max(timestamps)),
            'address_set': '|'.join(sorted(all_addrs)) if all_addrs else '',
            'token_symbol': str(grp['token'].iloc[0])[:40] if 'token' in grp.columns else 'unknown',
            'asset_group': str(grp['token'].iloc[0])[:40] if 'token' in grp.columns else 'unknown',
            '_txs': txs,
            '_group_val': str(group_val),
        }
        segments.append(seg)
        for tx in txs:
            tx_to_seg[tx.lower()] = seg_id

    return segments, tx_to_seg


def main():
    print('Loading dev anchor pairs...')
    dev_labels = load_dev_anchor_pairs()
    print(f'  Dev anchor pairs: {len(dev_labels)}')
    dev_src_hashes = set(dev_labels['sh'])

    print('Loading ETH source txs (dev only)...')
    eth_df = load_eth_source_txs(dev_src_hashes)
    print(f'  Dev ETH txs: {len(eth_df)}')

    print('Decoding BSC relay events...')
    bsc_df = decode_relay_events(RAW_RELAY)
    print(f'  BSC relay events: {len(bsc_df)}')

    # ===== BSC: aggregate by receiver + token + 3h =====
    print('')
    print('=== BSC Aggregation: receiver + token + 3h ===')
    dst_segs, bsc_tx_to_seg = aggregate_by_window(bsc_df, 'receiver', 'timestamp', 3 * 3600)
    tx_p_dst = [s['tx_count'] for s in dst_segs]
    print(f'  Segments: {len(dst_segs)}')
    print(f'  TX/seg: min={min(tx_p_dst)}, max={max(tx_p_dst)}, median={np.median(tx_p_dst):.0f}, mean={np.mean(tx_p_dst):.1f}')
    over20_d = sum(1 for s in dst_segs if s['tx_count'] > 20)
    print(f'  Segments >20 txs: {over20_d}')

    # ===== ETH: aggregate by sender + token + 3h =====
    print('')
    print('=== ETH Aggregation: sender + token + 3h ===')
    src_segs_3h, eth_tx_to_seg_3h = aggregate_by_window(eth_df, 'sender', 'timestamp', 3 * 3600)
    tx_p_3h = [s['tx_count'] for s in src_segs_3h]
    over20_3h = sum(1 for s in src_segs_3h if s['tx_count'] > 20)
    print(f'  3h Segments: {len(src_segs_3h)}')
    print(f'  3h TX/seg: min={min(tx_p_3h)}, max={max(tx_p_3h)}, median={np.median(tx_p_3h):.0f}')
    print(f'  3h Segments >20 txs: {over20_3h}')

    src_segs = src_segs_3h
    eth_tx_to_seg = eth_tx_to_seg_3h
    window_used = '3h'
    fallback = False

    if over20_3h > 0:
        fallback = True
        print('  Falling back to 1h...')
        src_segs_1h, eth_tx_to_seg_1h = aggregate_by_window(eth_df, 'sender', 'timestamp', 1 * 3600)
        tx_p_1h = [s['tx_count'] for s in src_segs_1h]
        over20_1h = sum(1 for s in src_segs_1h if s['tx_count'] > 20)
        print(f'  1h Segments: {len(src_segs_1h)}')
        print(f'  1h TX/seg: min={min(tx_p_1h)}, max={max(tx_p_1h)}, median={np.median(tx_p_1h):.0f}')
        print(f'  1h Segments >20 txs: {over20_1h}')
        window_used = '1h'

        if over20_1h > 0:
            print(f'  Still >20 for {over20_1h} segments, using chain+token fallback...')
            window_used = '1h_chain_token_fallback'
            eth_df_c = eth_df.copy()
            eth_df_c['_ct'] = 'ETH_' + eth_df_c['token'].astype(str)
            src_segs_fb, eth_tx_to_seg_fb = aggregate_by_window(eth_df_c, '_ct', 'timestamp', 1 * 3600)
            src_segs = src_segs_fb
            eth_tx_to_seg = eth_tx_to_seg_fb
            print(f'  Fallback Segments: {len(src_segs)}')
        else:
            src_segs = src_segs_1h
            eth_tx_to_seg = eth_tx_to_seg_1h


    # ===== GT Anchor Pair Mapping =====
    print('')
    print('=== GT Anchor Pair Mapping ===')
    label_rows = []
    mapped = 0
    unmapped_src = 0
    unmapped_dst = 0
    seg_pair_counts = defaultdict(int)

    for _, row in dev_labels.iterrows():
        sh = row['sh']
        dh = row['dh']
        src_seg = eth_tx_to_seg.get(sh)
        dst_seg = bsc_tx_to_seg.get(dh)
        if src_seg and dst_seg:
            mapped += 1
            label_rows.append({
                'src_flow_id': src_seg,
                'dst_flow_id': dst_seg,
                'src_tx_hash': sh,
                'dst_tx_hash': dh,
                'src_chain': 'ETH',
                'dst_chain': 'BNB',
            })
            seg_pair_counts[(src_seg, dst_seg)] += 1
        else:
            if not src_seg:
                unmapped_src += 1
            if not dst_seg:
                unmapped_dst += 1

    multi_pairs = [(ss, ds) for (ss, ds), cnt in seg_pair_counts.items() if cnt > 1]
    mapping_rate = mapped / max(len(dev_labels), 1)

    print(f'  Total anchors: {len(dev_labels)}')
    print(f'  Mapped: {mapped}')
    print(f'  Unmapped (src missing): {unmapped_src}')
    print(f'  Unmapped (dst missing): {unmapped_dst}')
    print(f'  Mapping rate: {100*mapping_rate:.2f}%')
    print(f'  Multi-pair segments: {len(multi_pairs)}')

    if mapping_rate < 0.995:
        print('  FATAL: Mapping rate below 99.5%!')
        missing_src = dev_labels[~dev_labels['sh'].isin(set(eth_tx_to_seg.keys()))]
        missing_dst = dev_labels[~dev_labels['dh'].isin(set(bsc_tx_to_seg.keys()))]
        print(f'  Missing src txs: {len(missing_src)}, Missing dst txs: {len(missing_dst)}')
        if len(missing_dst) > 0 and len(missing_dst) <= 20:
            print('  Missing dst tx hashes:')
            for h in missing_dst['dh'].head(10):
                print(f'    {h}')
        sys.exit(1)


    # ===== Build Output DataFrames =====
    print('')
    print('=== Building Output Files ===')

    def build_flow_df(segments, chain, route_type_default):
        rows = []
        for seg in segments:
            txs = seg['tx_hashes']
            rows.append({
                'flow_id': seg['flow_id'],
                'chain': chain,
                'tx_hashes': '|'.join(txs),
                'address_set': seg['address_set'],
                'amount_usd': seg['amount_usd'],
                'raw_amount_sum': seg['raw_amount_sum'],
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'token_symbol': seg['token_symbol'],
                'route_type': route_type_default,
                'route_id': seg['_group_val'][:20],
                'asset_group': seg['asset_group'],
                'aml_score': 0.0,
                'aml_risk_score_raw': 0.0,
                'evidence_quality_score': 0.5,
                'evidence_level': 2,
                'evidence_levels': 'aggregated',
                'tx_count': seg['tx_count'],
                'price_snapshot_ok': True,
                'graph_embedding': None,
            })
        return pd.DataFrame(rows)

    src_flow_df = build_flow_df(src_segs, 'ETH', 'celer_bridge_agg')
    dst_flow_df = build_flow_df(dst_segs, 'BNB', 'celer_relay_agg')
    label_df = pd.DataFrame(label_rows)

    src_path = OUT_DIR / 'src_flows_agg.csv'
    dst_path = OUT_DIR / 'dst_flows_agg.csv'
    lbl_path = OUT_DIR / 'flow_labels_agg.csv'

    src_flow_df.to_csv(src_path, index=False)
    dst_flow_df.to_csv(dst_path, index=False)
    label_df.to_csv(lbl_path, index=False)

    print(f'  Src flows: {len(src_flow_df)} rows -> {src_path}')
    print(f'  Dst flows: {len(dst_flow_df)} rows -> {dst_path}')
    print(f'  Labels: {len(label_df)} rows -> {lbl_path}')
    # ===== Segment-to-TX Mapping =====
    mapping = {
        'src_segment_to_tx': {s['flow_id']: s['_txs'] for s in src_segs},
        'dst_segment_to_tx': {s['flow_id']: s['_txs'] for s in dst_segs},
    }
    map_path = OUT_DIR / 'segment_tx_mapping.json'
    with open(map_path, 'w') as f:
        json.dump(mapping, f, indent=2, default=str)

    # ===== GT Traceability Check =====
    check = {
        'total_anchors': int(len(dev_labels)),
        'mapped': int(mapped),
        'unmapped_src_missing': int(unmapped_src),
        'unmapped_dst_missing': int(unmapped_dst),
        'mapping_rate': round(mapping_rate, 6),
        'multi_pair_segment_count': len(multi_pairs),
        'passed': mapping_rate >= 0.995,
        'src_segments': len(src_segs),
        'dst_segments': len(dst_segs),
        'aggregation_window': window_used,
        'src_aggregation': 'sender+token+3h' if window_used == '3h' else ('sender+token+1h' if window_used == '1h' else 'chain+token+1h'),
        'dst_aggregation': 'receiver+token+3h',
        'max_tx_per_src_segment': int(max(tx_p_3h)),
        'max_tx_per_dst_segment': int(max(tx_p_dst)),
        'fallback_applied': fallback,
    }
    check_path = OUT_DIR / 'gt_traceability_check.json'
    with open(check_path, 'w') as f:
        json.dump(check, f, indent=2)

    # ===== Manifest =====
    manifest = {
        'generated_at': _utc(),
        'script': 'build_aggregated_flow_segments.py',
        'src_aggregation': check['src_aggregation'],
        'dst_aggregation': check['dst_aggregation'],
        'n_src_flows': check['src_segments'],
        'n_dst_flows': check['dst_segments'],
        'n_labels': check['total_anchors'],
        'mapping_rate': check['mapping_rate'],
        'gt_traceability_passed': check['passed'],
        'files': {
            'src_flows': str(src_path),
            'dst_flows': str(dst_path),
            'flow_labels': str(lbl_path),
            'segment_tx_mapping': str(map_path),
            'gt_traceability_check': str(check_path),
        },
    }
    manifest_path = OUT_DIR / 'aggregation_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'')
    print(f'Done. Manifest: {manifest_path}')
    return manifest


if __name__ == '__main__':
    main()
