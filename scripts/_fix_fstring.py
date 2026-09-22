path = r'scripts/run_m1_causal_masked_analysis.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
old = '    print(f"  {len(ctx[\"eth_flows\"])} src, {len(ctx[\"bnb_flows\"])} dst, {len(ctx[\"truth\"])} gold pairs")'
new = '    n_src_one = len(ctx["eth_flows"]); n_dst_one = len(ctx["bnb_flows"]); n_gt_one = len(ctx["truth"])\n    print(f"  {n_src_one} src, {n_dst_one} dst, {n_gt_one} gold pairs")'
content = content.replace(old, new)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed')
