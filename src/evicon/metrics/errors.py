"""Explicit error types for pure EviCon-Lab metric functions."""


class MetricError(ValueError):
    """Base class for invalid metric inputs."""


class InsufficientDataError(MetricError):
    """Raised when missing inputs would otherwise be mistaken for similarity."""


class DimensionMismatchError(MetricError):
    """Raised when vectors or profiles cannot be aligned safely."""


class MissingPairError(InsufficientDataError):
    """Raised when a pairwise graph is incomplete in strict mode."""
