"""M1 Ablation: unified solver interface, fixed decoder, evaluation infrastructure."""
from cross.domain.uot.m1_ablation.transport_solver import (
    ALL_SOLVERS, FACTORIAL_SOLVERS, SolverName, TransportResult,
    solve_thresholded_cost, solve_greedy_nn, solve_hungarian,
    solve_balanced_ot, solve_rc_uot, get_solver,
)
from cross.domain.uot.m1_ablation.fixed_decoder import (
    decode_with_fixed_rc_uot_q, DecodeResult, DecodeConfig,
)
from cross.domain.uot.m1_ablation.evaluation import (
    compute_metrics, compute_stratified_metrics, BootstrapCI,
    StructureLabel, assign_structure_label, MetricsResult, StratifiedResult,
)
from cross.domain.uot.m1_ablation.calibration import (
    calibrate_threshold, CalibrationResult,
)
from cross.domain.uot.m1_ablation.precision_coverage import (
    compute_precision_coverage_curve, PrecisionCoveragePoint,
    PrecisionCoverageResult, save_precision_coverage_curve,
)
from cross.domain.uot.m1_ablation.data_adapter import (
    discover_data, print_data_summary, M1Dataset,
)
