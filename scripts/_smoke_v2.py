import numpy as np
import sys
sys.path.insert(0, 'src')
from cross.domain.uot.rc_uot_v2 import solve_rc_uot_v2

np.random.seed(42)
n_src, n_dst = 5, 8
C = np.random.rand(n_src, n_dst) * 2.0
a = np.random.rand(n_src)
b = np.random.rand(n_dst)
causal_mask = np.ones((n_src, n_dst), dtype=bool)
causal_mask[0, 0] = False
causal_mask[4, 7] = False
q_s = np.random.rand(n_src)
q_t = np.random.rand(n_dst)

for variant in ['rc_uot_v2_partial', 'rc_uot_v2_reliability', 'rc_uot_v2_sparse']:
    result = solve_rc_uot_v2(C, a, b, causal_mask, variant=variant, q_s=q_s, q_t=q_t)
    m = result.meta
    print(f"{variant}: real_mass={m['transport_mass_real']:.3f}, "
          f"src_dustbin={m['source_dustbin_fraction']:.3f}, "
          f"dst_dustbin={m['target_dustbin_fraction']:.3f}, "
          f"rt={m['runtime_sec']:.3f}s")
    assert result.P_real.shape == (n_src, n_dst)
    assert result.P_real[0, 0] == 0.0
    assert result.P_real[4, 7] == 0.0
    assert (result.P_real >= -1e-12).all()
    assert m['causal_violation_rate'] == 0.0
print('All smoke tests passed')
