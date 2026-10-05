"""Carry partial evaluation artifacts when a live rollout fails."""


class EvaluationRunError(RuntimeError):
    """A policy or infrastructure fault with any saved partial video."""

    def __init__(self, message: str, *, policy_fault: bool, artifact_paths: dict[str, str]) -> None:
        super().__init__(message)
        self.policy_fault = policy_fault
        self.artifact_paths = artifact_paths
