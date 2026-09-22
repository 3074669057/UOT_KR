# Paper main results paragraph (fixed-delay RC-UOT-Q)

The final model uses **`tx_if_available_else_flow_representative`** delay policy, replacing the legacy flow-boundary delay. On all **7,296** Celer anchor pairs, RC-UOT is evaluated as a **ranked flow-correspondence model**—not a raw tx-pair oracle. The raw tx-level argmax projection reaches pair F1 **0.589** and top-3 recall **0.658**, but is reported only as a **compatibility projection**, not as a forensic admissibility claim (tx-level CVR **0.338**).

The formal **RC-UOT-Q high-coverage decoding** is **positive_delay_top3_rescue**: pair F1 **0.590**, top-3 recall **0.658**, tx-level CVR **0.005**, coverage **0.996**. The formal **forensic high-confidence subset** is **joint_time_admissible_filter**: precision **0.889**, recall **0.589**, pair F1 **0.708**, tx-level CVR **0**, coverage **0.845**.

Permuted-label controls collapse to near-zero pair F1, confirming label-structured recovery.

Old headline metrics (pair F1 ≈ 0.321, legacy delay / pre-admissible pipeline) are retained for **diagnostic comparison only**, not for paper headline claims.
