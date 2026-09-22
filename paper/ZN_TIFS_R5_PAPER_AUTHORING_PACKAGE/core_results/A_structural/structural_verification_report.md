# Independent verification report

- **Celer**: split 0.9500 ± 0.0316; merge 0.9667 ± 0.0432 (per-seed split [0.895833, 0.958333, 0.958333, 0.979167, 0.958333], merge [0.895833, 0.958333, 1.0, 1.0, 0.979167])
- **Multi**: split 0.9708 ± 0.0349; merge 0.9792 ± 0.0255 (per-seed split [0.916667, 0.958333, 0.979167, 1.0, 1.0], merge [0.9375, 0.979167, 0.979167, 1.0, 1.0])
- **Poly**: split 1.0000 ± 0.0000; merge 1.0000 ± 0.0000 (per-seed split [1.0, 1.0, 1.0, 1.0, 1.0], merge [1.0, 1.0, 1.0, 1.0, 1.0])

```json
{
  "Celer_n_seeds": 5,
  "Celer_artifact_vs_recalc_consistent": true,
  "Celer_templates_48": true,
  "Celer_decoded_has_multi_dst": true,
  "Celer_decoded_has_multi_src": true,
  "Multi_n_seeds": 5,
  "Multi_artifact_vs_recalc_consistent": true,
  "Multi_templates_48": true,
  "Multi_decoded_has_multi_dst": true,
  "Multi_decoded_has_multi_src": true,
  "Poly_n_seeds": 5,
  "Poly_artifact_vs_recalc_consistent": true,
  "Poly_templates_48": true,
  "Poly_decoded_has_multi_dst": true,
  "Poly_decoded_has_multi_src": true,
  "celer_regression": {
    "frozen": {
      "split": 0.9458333333333332,
      "merge": 0.9666666666666668
    },
    "new": {
      "split_mean": 0.95,
      "merge_mean": 0.9666666666666668
    },
    "split_delta": 0.0042,
    "merge_delta": 0.0,
    "pass": true
  },
  "baselines_structure_level": {
    "all_zero": true,
    "n_rows": 30,
    "per_bridge_zero": {
      "Celer": true,
      "Multi": true,
      "Poly": true
    }
  },
  "template_structure_spotcheck": {
    "Celer": {
      "split_1src_to_2dst_ok": true,
      "merge_2src_to_1dst_ok": true,
      "n_split_templates": 48,
      "n_merge_templates": 48
    },
    "Multi": {
      "split_1src_to_2dst_ok": true,
      "merge_2src_to_1dst_ok": true,
      "n_split_templates": 48,
      "n_merge_templates": 48
    },
    "Poly": {
      "split_1src_to_2dst_ok": true,
      "merge_2src_to_1dst_ok": true,
      "n_split_templates": 48,
      "n_merge_templates": 48
    }
  }
}
```