from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.evaluation.compare_experiments import (
    export_audio_bundle,
    join_metric_records,
    select_challenging_tracks,
    select_excerpt_start,
    summarize_deltas,
)


def test_join_metric_records_computes_signed_quality_deltas():
    exp04 = [
        {
            "track_name": "Track A",
            "vocals_median_sdr_db": 6.0,
            "inst_median_sdr_db": 12.0,
            "vocal_to_inst_bleed_coeff": 0.2,
        }
    ]
    exp06 = [
        {
            "track_name": "Track A",
            "vocals_median_sdr_db": 6.5,
            "inst_median_sdr_db": 11.8,
            "vocal_to_inst_bleed_coeff": 0.18,
        }
    ]

    rows = join_metric_records(exp04, exp06)

    assert rows[0]["delta_vocals_median_sdr_db"] == pytest.approx(0.5)
    assert rows[0]["delta_inst_median_sdr_db"] == pytest.approx(-0.2)
    assert rows[0]["delta_vocal_to_inst_bleed_coeff"] == pytest.approx(-0.02)


def test_join_metric_records_rejects_different_track_sets():
    with pytest.raises(ValueError, match="track sets"):
        join_metric_records(
            [{"track_name": "A", "vocals_median_sdr_db": 1.0}],
            [{"track_name": "B", "vocals_median_sdr_db": 1.0}],
        )


def test_delta_summary_counts_vocal_wins_and_regressions():
    rows = [
        {"delta_vocals_median_sdr_db": 0.2},
        {"delta_vocals_median_sdr_db": -0.1},
        {"delta_vocals_median_sdr_db": 0.0},
    ]

    summary = summarize_deltas(rows, "vocals_median_sdr_db")

    assert summary["wins"] == 1
    assert summary["regressions"] == 1
    assert summary["ties"] == 1
    assert summary["median_delta"] == pytest.approx(0.0)


def test_challenging_track_selection_is_deterministic_and_covers_categories():
    rows = [
        {
            "track_name": f"Track {index}",
            "delta_vocals_median_sdr_db": delta,
            "overlap_score": overlap,
        }
        for index, (delta, overlap) in enumerate(
            [
                (0.9, 0.1),
                (0.8, 0.2),
                (-0.9, 0.3),
                (-0.8, 0.4),
                (0.1, 0.9),
                (-0.1, 0.8),
                (0.0, 0.7),
            ]
        )
    ]

    selected = select_challenging_tracks(rows, count=6)

    assert [item["track_name"] for item in selected] == [
        "Track 4",
        "Track 5",
        "Track 0",
        "Track 1",
        "Track 2",
        "Track 3",
    ]
    assert [item["reason"] for item in selected] == [
        "overlap",
        "overlap",
        "improvement",
        "improvement",
        "regression",
        "regression",
    ]


def test_excerpt_start_selects_highest_ground_truth_vocal_rms_window():
    sample_rate = 10
    vocals = np.zeros((100, 2), dtype=np.float32)
    vocals[40:60] = 2.0

    start = select_excerpt_start(
        vocals,
        sample_rate=sample_rate,
        duration_seconds=2,
        step_seconds=1,
    )

    assert start == 40


def test_audio_bundle_export_preserves_model_amplitudes_without_normalization(tmp_path: Path):
    arrays = {
        "mixture": np.full((10, 2), 0.5, dtype=np.float32),
        "exp04_vocals": np.full((10, 2), 0.25, dtype=np.float32),
        "exp06_vocals": np.full((10, 2), 2.0, dtype=np.float32),
    }

    paths = export_audio_bundle(tmp_path, arrays, sample_rate=10)

    exported, sample_rate = sf.read(paths["exp06_vocals"], dtype="float32")
    assert sample_rate == 10
    assert np.max(exported) == pytest.approx(2.0)
    assert np.array_equal(exported, arrays["exp06_vocals"])
