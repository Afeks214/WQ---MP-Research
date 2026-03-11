from __future__ import annotations

from enum import IntEnum


class FeatureIdx(IntEnum):
    """Stable F-axis indices for Module3 shared feature tensor access."""

    DCLIP = 0
    A_AFFINITY = 1
    Z_DELTA = 2
    GBREAK = 3
    GREJECT = 4
    DELTA_EFF = 5
    SCORE_BO_LONG = 6
    SCORE_BO_SHORT = 7
    SCORE_REJECT = 8
    X_POC = 9
    X_VAH = 10
    X_VAL = 11
    MU_PROF = 12
    SIGMA_PROF = 13
    SKEW_PROF = 14
    KURT_PROF = 15
    PROFILE_ENTROPY = 16
    PROFILE_BALANCE_RATIO = 17
    PROFILE_PEAK_COUNT = 18
    PROFILE_POC_DISTANCE = 19
    PROFILE_VALUE_AREA_WIDTH = 20
    BAR_VALID = 21
    IB_HIGH = 22
    IB_LOW = 23
    N_FIELDS = 24


class StructIdx(IntEnum):
    """Stable F_struct-axis indices. Kept compatible with historical Module3 usage."""

    VALID_RATIO = 0
    N_VALID_BARS = 1
    X_POC = 2
    X_VAH = 3
    X_VAL = 4
    VA_WIDTH_X = 5
    MU_ANCHOR = 6
    SIGMA_ANCHOR = 7
    SKEW_ANCHOR = 8
    KURT_EXCESS_ANCHOR = 9
    TAIL_IMBALANCE = 10
    DCLIP_MEAN = 11
    DCLIP_STD = 12
    AFFINITY_MEAN = 13
    ZDELTA_MEAN = 14
    GBREAK_MEAN = 15
    GREJECT_MEAN = 16
    DELTA_EFF_MEAN = 17
    SCORE_BO_LONG_MEAN = 18
    SCORE_BO_SHORT_MEAN = 19
    SCORE_REJECT_MEAN = 20
    TREND_GATE_SPREAD_MEAN = 21
    POC_DRIFT_X = 22
    VAH_DRIFT_X = 23
    VAL_DRIFT_X = 24
    DELTA_SHIFT = 25
    IB_HIGH_X = 26
    IB_LOW_X = 27
    POC_VS_PREV_VA = 28
    N_FIELDS = 29


class FingerprintIdx(IntEnum):
    PROFILE_SKEW = 0
    PROFILE_KURTOSIS = 1
    PROFILE_ENTROPY = 2
    PROFILE_BALANCE_RATIO = 3
    PROFILE_PEAK_COUNT = 4
    PROFILE_POC_DISTANCE = 5
    PROFILE_VALUE_AREA_WIDTH = 6
    N_FIELDS = 7


class ContextIdx(IntEnum):
    CTX_X_POC = 0
    CTX_X_VAH = 1
    CTX_X_VAL = 2
    CTX_VA_WIDTH_X = 3
    CTX_DCLIP_MEAN = 4
    CTX_AFFINITY_MEAN = 5
    CTX_ZDELTA_MEAN = 6
    CTX_DELTA_EFF_MEAN = 7
    CTX_TREND_GATE_SPREAD_MEAN = 8
    CTX_POC_DRIFT_X = 9
    CTX_VALID_RATIO = 10
    CTX_IB_HIGH_X = 11
    CTX_IB_LOW_X = 12
    CTX_POC_VS_PREV_VA = 13
    CTX_REGIME_CODE = 14
    CTX_REGIME_PERSISTENCE = 15
    N_FIELDS = 16


class RegimeCode(IntEnum):
    BALANCED = 0
    TREND_UP = 1
    TREND_DOWN = 2
    DOUBLE_DISTRIBUTION = 3
    EXTREME_IMBALANCE = 4
