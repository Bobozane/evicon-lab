"""Pure, local-only metric functions and JSON-serializable result contracts."""

from .counterfactual import holdout_social_influence_loss, social_influence_loss
from .distances import answer_distance, normalized_euclidean_distance, profile_distance
from .diversity import pairwise_diversity, structural_diversity
from .errors import DimensionMismatchError, InsufficientDataError, MetricError, MissingPairError
from .models import (
    AnswerDistanceResult,
    HoldoutProfileDriftResult,
    MetricResult,
    MinorityRetentionResult,
    PairDistance,
    PairwiseDiversityResult,
    SocialInfluenceLossResult,
    StructuralDiversityResult,
    ValueDimensionCoverageResult,
)
from .representation import holdout_profile_drift, minority_retention, value_dimension_coverage

__all__ = [
    "AnswerDistanceResult",
    "DimensionMismatchError",
    "HoldoutProfileDriftResult",
    "InsufficientDataError",
    "MetricError",
    "MetricResult",
    "MinorityRetentionResult",
    "MissingPairError",
    "PairDistance",
    "PairwiseDiversityResult",
    "SocialInfluenceLossResult",
    "StructuralDiversityResult",
    "ValueDimensionCoverageResult",
    "answer_distance",
    "holdout_profile_drift",
    "holdout_social_influence_loss",
    "minority_retention",
    "normalized_euclidean_distance",
    "pairwise_diversity",
    "profile_distance",
    "social_influence_loss",
    "structural_diversity",
    "value_dimension_coverage",
]
