import json, sys, tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / 'src'))

from cross.domain.provider_data.validator import (
    validate_provider_directory, scan_for_label_like_fields,
    run_anti_triviality_gate, validate_semantics,
)

def _make_valid_dir(tmp):
    d = tmp / 'valid_p0'; d.mkdir()
    src = [{'source_flow_id':f'src_{i:03d}','source_chain':'ETH','transaction_ids':[f'0xtx_{i}'],'constituent_transaction_ids':[f'0xct_{i}'],'start_timestamp':'2026-06-01T00:00:00Z','end_timestamp':'2026-06-01T01:00:00Z','asset':'USDC','raw_amount':'1000.00','normalized_amount':'1000.00','source_address':'0xaddr','bridge_or_route_id':'cBridge','raw_evidence_reference':'ev','data_collection_timestamp':'2026-06-24T00:00:00Z','data_source_version':'v1.0'} for i in range(25)]
    tgt = [{'target_flow_id':f'tgt_{j:03d}','target_chain':'BNB','transaction_ids':[f'0xtx_t{j}'],'constituent_transaction_ids':[f'0xct_t{j}'],'start_timestamp':'2026-06-01T01:01:00Z','end_timestamp':'2026-06-01T02:00:00Z','asset':'USDC','raw_amount':'998.00','normalized_amount':'998.00','target_address':'0xaddr','bridge_or_route_id':'cBridge','raw_evidence_reference':'ev','data_collection_timestamp':'2026-06-24T00:00:00Z','data_source_version':'v1.0'} for j in range(22)]
    manifest = {'dataset_id':'test','provider_id_hash':'sha256:00','hash_algorithm':'SHA-256','collection_start':'2026-06-01T00:00:00Z','collection_end':'2026-06-02T00:00:00Z','source_chains':['ETH'],'target_chains':['BNB'],'bridge_protocols':['cBridge'],'assets':['USDC'],'collection_method':'test','candidate_universe_scope':'all_pairs','raw_record_count':47,'source_flow_count':25,'target_flow_count':22,'data_source_version':'v1.0','schema_version':'v4.3a'}
    with open(d/'source_flows.jsonl','w') as f:
        for r in src: f.write(json.dumps(r)+'\n')
    with open(d/'target_flows.jsonl','w') as f:
        for r in tgt: f.write(json.dumps(r)+'\n')
    (d/'collection_manifest.json').write_text(json.dumps(manifest))
    return d

def test_valid():
    with tempfile.TemporaryDirectory() as td:
        r = validate_provider_directory(_make_valid_dir(Path(td)))
        assert r.status == 'P0_ACCEPTED', r.status

def test_missing_src():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);(d/'target_flows.jsonl').write_text('{}\n');(d/'collection_manifest.json').write_text('{}')
        assert 'REJECTED' in validate_provider_directory(d).status

def test_missing_tgt():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);(d/'source_flows.jsonl').write_text('{}\n');(d/'collection_manifest.json').write_text('{}')
        assert 'REJECTED' in validate_provider_directory(d).status

def test_labels_dir():
    with tempfile.TemporaryDirectory() as td:
        d=_make_valid_dir(Path(td));(d/'labels').mkdir()
        assert validate_provider_directory(d).status == 'P0_REJECTED_LABEL_LEAKAGE'

def test_label_like_field():
    with tempfile.TemporaryDirectory() as td:
        d=_make_valid_dir(Path(td))
        src = json.loads((d/'source_flows.jsonl').read_text().split('\n')[0])
        src['ground_truth']='matched'
        issues = scan_for_label_like_fields([src],'test.jsonl')
        assert len(issues)>0

def test_dup_id():
    with tempfile.TemporaryDirectory() as td:
        d=_make_valid_dir(Path(td))
        s=json.loads((d/'source_flows.jsonl').read_text().split('\n')[0])
        with open(d/'source_flows.jsonl','w') as f:f.write(json.dumps(s)+'\n'+json.dumps(s)+'\n')
        assert 'REJECTED' in validate_provider_directory(d).status

def test_valid_gate():
    ok,_,_ = run_anti_triviality_gate(20,20,[2]*20,[2]*20,400)
    assert ok

def test_rev_ts():
    with tempfile.TemporaryDirectory() as td:
        d=_make_valid_dir(Path(td))
        s=json.loads((d/'source_flows.jsonl').read_text().split('\n')[0])
        s['start_timestamp']='2026-06-02T00:00:00Z';s['end_timestamp']='2026-06-01T00:00:00Z'
        with open(d/'source_flows.jsonl','w') as f:f.write(json.dumps(s)+'\n')
        _,issues = validate_semantics(d,{'source_chains':['ETH'],'target_chains':['BNB']},[s],[])
        assert any('TIMESTAMP' in i.code for i in issues)

def test_neg_amount():
    with tempfile.TemporaryDirectory() as td:
        d=_make_valid_dir(Path(td))
        s=json.loads((d/'source_flows.jsonl').read_text().split('\n')[0]);s['raw_amount']='-100'
        with open(d/'source_flows.jsonl','w') as f:f.write(json.dumps(s)+'\n')
        _,issues = validate_semantics(d,{'source_chains':['ETH'],'target_chains':['BNB']},[s],[])
        assert any('NEGATIVE' in i.code for i in issues)

def test_count_mismatch():
    with tempfile.TemporaryDirectory() as td:
        d=_make_valid_dir(Path(td))
        m=json.loads((d/'collection_manifest.json').read_text());m['source_flow_count']=999
        (d/'collection_manifest.json').write_text(json.dumps(m))
        assert 'REJECTED' in validate_provider_directory(d).status

def test_trivial():
    ok,_,_ = run_anti_triviality_gate(5,5,[1]*5,[1]*5,5)
    assert not ok

def test_prematched():
    ok,_,_ = run_anti_triviality_gate(20,20,[1]*20,[1]*20,20)
    assert not ok

def test_zero_fraction():
    _,_,stats = run_anti_triviality_gate(30,25,[10]*15+[0]*15,[5]*10+[0]*15,500)
    assert stats['fraction_sources_zero_candidates']==0.50

def test_full_sha256():
    h=json.loads((_REPO/'out/rc_uot_v2_2/provider_contract/candidate_generation_policy.json').read_text())['policy_hash_sha256']
    assert len(h)==64

def test_no_solver_import():
    txt=(_REPO/'src/cross/domain/provider_data/validator.py').read_text()
    for fb in ['from cross.domain.uot','import ot ','sinkhorn','from ot ']:
        assert fb not in txt

def test_no_labels_access():
    txt=(_REPO/'src/cross/domain/provider_data/validator.py').read_text()
    for fb in ['gold_label','label_loader','load_labels','annotation_file']:
        assert fb not in txt.lower()

tests = [
    test_valid,test_missing_src,test_missing_tgt,test_labels_dir,
    test_label_like_field,test_dup_id,test_valid_gate,test_rev_ts,
    test_neg_amount,test_count_mismatch,test_trivial,test_prematched,
    test_zero_fraction,test_full_sha256,test_no_solver_import,test_no_labels_access,
]
p=f=0
for t in tests:
    try:
        t();print(f'PASS: {t.__name__}');p+=1
    except Exception as e:
        print(f'FAIL: {t.__name__} - {e}');f+=1
print(f'{p}/{len(tests)} passed, {f} failed')
sys.exit(0 if f==0 else 1)
