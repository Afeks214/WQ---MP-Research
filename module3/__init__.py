from .profile_fingerprint_engine import compute_profile_fingerprint_tensor, compute_profile_regime_tensor
from .schema import ContextIdx, FeatureIdx, FingerprintIdx, RegimeCode, StructIdx
from .structural_context_builder import build_context_tensor
from .structural_kernels import (
    compute_poc_distance,
    compute_profile_kurtosis,
    compute_profile_skew,
    compute_value_area_width,
)
from .structural_prefix_sums import (
    build_prefix_count,
    build_prefix_sum,
    rolling_count_from_prefix,
    rolling_mean_from_prefix,
    rolling_sum_from_prefix,
)
from .structural_validation import (
    benchmark_naive_vs_prefix,
    deterministic_digest_sha256_module3,
    run_forensic_validation,
    validate_context_causality,
    validate_fingerprint_stability,
    validate_output_contract,
    validate_prefix_sum_parity,
    validate_shared_tensor_contract,
    validate_window_alignment,
)
from .structural_window_engine import run_structural_window_engine
from .types import Module3Config, Module3Output
from .window_registry import WINDOWS, build_window_index_map, normalize_structural_windows, resolve_window_index

__all__ = [
    "WINDOWS",
    "ContextIdx",
    "FeatureIdx",
    "FingerprintIdx",
    "Module3Config",
    "Module3Output",
    "RegimeCode",
    "StructIdx",
    "build_context_tensor",
    "build_prefix_count",
    "build_prefix_sum",
    "build_window_index_map",
    "benchmark_naive_vs_prefix",
    "compute_poc_distance",
    "compute_profile_fingerprint_tensor",
    "compute_profile_kurtosis",
    "compute_profile_regime_tensor",
    "compute_profile_skew",
    "compute_value_area_width",
    "deterministic_digest_sha256_module3",
    "normalize_structural_windows",
    "resolve_window_index",
    "rolling_count_from_prefix",
    "rolling_mean_from_prefix",
    "rolling_sum_from_prefix",
    "run_forensic_validation",
    "run_structural_window_engine",
    "validate_context_causality",
    "validate_fingerprint_stability",
    "validate_output_contract",
    "validate_prefix_sum_parity",
    "validate_shared_tensor_contract",
    "validate_window_alignment",
]
