"""R7 -- degree-calibrated UOT-KR confirmatory kernel ranking.

Package layout
--------------
``r7_common``     paths, frozen constants, seed guards, hashing, metrics
``r7_degree``     Stage 0A: empirical degree calibration from the frozen v4/v5 audits
``r7_generator``  Stage 0B: extended faithful generator (empirical split/merge degrees,
                  24 template families), reusing the real paper generator's code path
``r7_methods``    every decoder: UOT-KR rule families, RAW / CONDITIONAL / SUPPORT+K,
                  THRESHOLD_MM, HUNGARIAN_1TO1, DUAL_SOFTMAX, ORACLE_1TO1_CEILING
``r7_pipeline``   one cell = generate -> cost -> one UOT solve -> all decoders -> metrics
"""

__all__ = ["r7_common", "r7_degree", "r7_generator", "r7_methods", "r7_pipeline"]
