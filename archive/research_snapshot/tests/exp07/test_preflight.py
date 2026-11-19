import pytest
import torch

from src.experiments.exp07_phase_aware_residual.preflight import (
    PreflightFailure, benchmark_cuda, evaluate_learnability, validate_oracle_report,
)


def test_learnability_checks_each_approved_threshold():
    assert evaluate_learnability(20, 10, [1, 1, 1, 0], 0.01, True)
    assert not evaluate_learnability(19.99, 10, [1, 1, 1, 0], 0.01, True)
    assert not evaluate_learnability(20, 9.99, [1, 1, 1, 0], 0.01, True)
    assert not evaluate_learnability(20, 10, [1, 1, 0, 0], 0.01, True)
    assert not evaluate_learnability(20, 10, [1, 1, 1, 0], 0, True)
    assert not evaluate_learnability(20, 10, [1, 1, 1, 0], 1, False)


def test_absent_oracle_report_blocks_preflight(tmp_path):
    with pytest.raises(PreflightFailure, match="Oracle"):
        validate_oracle_report(tmp_path / "missing.json", "expected_hash")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_oom_does_not_retry_at_smaller_batch():
    def fail():
        raise torch.cuda.OutOfMemoryError("test")

    with pytest.raises(PreflightFailure, match="batch size 4"):
        benchmark_cuda(fail, warmup=1, measured=1)
