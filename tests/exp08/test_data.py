import pytest
import numpy as np
import soundfile as sf
import torch

from src.experiments.exp08_subband.data import (
    TrainClips, reserve_run, verify_partition,
)


def test_partition_checks_order_and_resolved_paths(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    test = tmp_path / "test"
    test.mkdir()
    a, b = train / "a", train / "b"
    a.mkdir()
    b.mkdir()
    verify_partition(train, ([a], [b]), (["a"], ["b"]))
    for paths, names in [
        (([a], [b]), (["b"], ["a"])),
        (([a], [a]), (["a"], ["a"])),
        (([test], [b]), (["test"], ["b"])),
    ]:
        with pytest.raises(ValueError):
            verify_partition(train, paths, names)


def test_run_directory_cannot_be_reused(tmp_path):
    run = reserve_run(tmp_path / "run")
    with pytest.raises(FileExistsError):
        reserve_run(run)


def test_augmented_clips_repeat_and_share_channel_swap(tmp_path):
    track = tmp_path / "one"
    track.mkdir()
    for stem, factor in (("mixture", 3), ("vocals", 1), ("instrumental", 2)):
        signal = np.tile([.01 * factor, -.01 * factor], (50, 1))
        sf.write(track / f"{stem}.wav", signal, 44100, subtype="FLOAT")
    dataset = TrainClips([track], epoch=7, samples=2)
    a, b = dataset[1], dataset[1]
    torch.testing.assert_close(a["audio"], b["audio"], rtol=0, atol=0)
    assert a["audio"].shape == (3, 2, 441000)
    torch.testing.assert_close(a["audio"][0], a["audio"][1] + a["audio"][2])
    assert (a["audio"][1, 0, :50] * a["audio"][2, 0, :50] > 0).all()
    assert a["vocals_track"] == a["instrumental_track"] == "one"
