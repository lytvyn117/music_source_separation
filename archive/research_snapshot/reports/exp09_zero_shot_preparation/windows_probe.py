"""Verify the released Sony variant on synthetic audio only."""
import ast
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
import yaml


HERE = Path(__file__).resolve().parent
CHUNK = 176400
OVERLAP = CHUNK // 2
RELEASE_DIGEST = "1308be27073d6a0c667a6a480348e0b33c9f6720d352deb0a096c79ea6d33abd"
CODE_DIGEST = "9b2213e375bb6a5d0b095d23a67aa107030b50b509eccc9ce65d09f44ba88b88"
WHEELS = {
    "beartype-0.22.9-py3-none-any.whl": "d16c9bbc61ea14637596c5f6fbff2ee99cbe3573e46a716401734ef50c3060c2",
    "einops-0.8.1-py3-none-any.whl": "919387eb55330f5757c6bea9165c5ff5cfe63a642682ea788a6d472576d81737",
    "hyper_connections-0.4.11-py3-none-any.whl": "92a8d24340ef6dbf2a356b1f2e74ebb877d680aa96afc6f9f1cff636ab2e55f0",
    "rotary_embedding_torch-0.8.9-py3-none-any.whl": "700e8de9dfbefba5f9117a66652a2520648dcc60136895f0068b6a85347cab02",
    "torch_einops_utils-0.0.20-py3-none-any.whl": "babb960422982fe95b64a4c71f5e97c4bdb0f352da0fae03ebb974d6b860fb4d",
}


def verify_file(path, expected):
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return path


def load_module(name, code):
    module = ModuleType(name)
    sys.modules[name] = module
    exec(compile(code, name, "exec"), module.__dict__)
    return module


def upstream_overlap_add(source):
    # Import only the upstream inference functions needed by the probe.
    # The full evaluator imports metric and training packages unrelated to the model.
    import math
    from einops import rearrange

    names = {"make_audio_batch", "cross_fade", "separate"}
    tree = ast.parse(source)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in funcs} != names:
        raise ValueError("Unexpected upstream inference functions")
    import typing as tp

    scope = {"torch": torch, "nn": torch.nn, "math": math, "rearrange": rearrange, "tp": tp}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "upstream_evaluate", "exec"), scope)
    return scope


def load_sony():
    wheels = [verify_file(HERE / "wheels" / name, digest) for name, digest in WHEELS.items()]
    sys.path[:0] = [str(path) for path in wheels]
    with ZipFile(verify_file(HERE / "sony_code_v1.0.0.zip", CODE_DIGEST)) as z:
        paths = {name.partition("/")[2]: name for name in z.namelist()}
        load_module("utils", "")
        load_module("utils.torch_common", z.read(paths["src/utils/torch_common.py"]))
        backbone_module = load_module("sony_bsroformer", z.read(paths["src/model/backbone/bsroformer/bsroformer.py"]))
        wrapper_module = load_module("sony_siso", z.read(paths["src/model/siso_base.py"]))
        overlap_add = upstream_overlap_add(z.read(paths["src/evaluate.py"]))

    with ZipFile(verify_file(HERE / "bsroformer_small.zip", RELEASE_DIGEST)) as z:
        config = z.read(next(name for name in z.namelist() if name.endswith("/config.yaml")))
        weights = z.read(next(name for name in z.namelist() if name.endswith("/backbone_model.pth")))

    cfg = yaml.safe_load(config)
    model_cfg = cfg["model"]
    backbone_args = dict(model_cfg["backbone_model"])
    assert model_cfg["_target_"] == "model.SISOBase"
    assert backbone_args.pop("_target_") == "model.backbone.BSRoformer"
    assert backbone_args["audio_channels"] == "${..num_channels}"
    backbone_args["audio_channels"] = model_cfg["num_channels"]
    assert cfg["data"]["train"] == {"sample_rate": 44100, "sample_length": CHUNK, "out_channels": "stereo"}
    backbone = backbone_module.BSRoformer(**backbone_args)
    backbone.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True), strict=True)
    model = wrapper_module.SISOBase(
        num_channels=model_cfg["num_channels"],
        num_sources=model_cfg["num_sources"],
        input_normalize=model_cfg["input_normalize"],
        backbone_model=backbone,
    ).cuda().eval()
    return model, backbone_module, overlap_add, config, weights, wheels


@torch.inference_mode()
def probe():
    from contextlib import nullcontext

    model, backbone_module, upstream, config, weights, wheels = load_sony()
    attention = [module for module in model.modules() if isinstance(module, backbone_module.Attend)]
    assert len(attention) == 8 and all(module.flash for module in attention)
    count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert (count, trainable) == (12_126_676, 12_126_652)
    native_error = None
    if not torch.backends.cuda.is_flash_attention_available():
        try:
            model.inference(torch.zeros(1, 2, CHUNK, device="cuda"))
        except RuntimeError as exc:
            native_error = str(exc)
        if native_error is None or "No available kernel" not in native_error:
            raise RuntimeError(f"Unexpected native Flash outcome: {native_error}")
        # Retain the upstream scaled-dot-product equation and weights.
        # The stock Windows build lacks the configured Flash kernel.
        for module in attention:
            module.cuda_config = [SDPBackend.EFFICIENT_ATTENTION]

    class NoAutocast:
        autocast = staticmethod(nullcontext)

    def run(mixture):
        vocals = upstream["separate"](
            model, mixture, CHUNK, OVERLAP, 1, [1], NoAutocast()
        )[0, 0]
        return torch.stack((vocals, mixture - vocals))

    def waveform(length):
        t = torch.arange(length, device="cuda") / 44100
        return torch.stack((0.27 * torch.sin(2 * torch.pi * 233 * t),
                            0.07 * torch.cos(2 * torch.pi * 587 * t)))

    cases = {"silence": torch.zeros(2, 4410, device="cuda"), "short": waveform(44101),
             "irregular_tail": waveform(44100 * 5 + 17)}
    fixture = upstream["make_audio_batch"](cases["short"], CHUNK, OVERLAP)[0]
    direct = model.inference(fixture)[0, 0, :, :cases["short"].shape[-1]]
    assert torch.equal(direct, run(cases["short"])[0])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    results = {}
    for name, mixture in cases.items():
        start = perf_counter()
        estimates = run(mixture)
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        assert estimates.shape == (2, *mixture.shape)
        assert bool(estimates.isfinite().all())
        residual = (estimates.sum(0) - mixture).abs().max().item()
        assert residual <= 2e-5
        peak = estimates.abs().max().item()
        if name == "silence":
            assert peak == 0
        assert torch.equal(estimates, run(mixture))
        results[name] = {"samples": mixture.shape[-1], "first_timed_seconds": seconds,
                         "peak": peak, "peak_additivity_residual": residual, "exact_repeat": True}

    # Compare the FP32 Windows backend to PyTorch's math backend on identical input.
    selected = model.inference(fixture)
    working_backend = attention[0].cuda_config[0].name
    selected_allocated = torch.cuda.max_memory_allocated()
    selected_reserved = torch.cuda.max_memory_reserved()
    for module in attention:
        module.cuda_config = [SDPBackend.MATH]
    math_reference = model.inference(fixture)
    torch.testing.assert_close(selected, math_reference, atol=2e-5, rtol=1e-4)
    return {
        "scope": "synthetic compatibility only",
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "flash_compiled": torch.backends.cuda.is_flash_attention_available(), "native_flash_error": native_error,
        "working_backend": working_backend,
        "precision": "FP32", "chunk_samples": CHUNK, "overlap_samples": OVERLAP, "batch_size": 1,
        "parameters": count, "trainable_parameters": trainable, "frequency_bands": model.backbone_model.num_bands,
        "weight_sha256": sha256(weights).hexdigest(), "config_sha256": sha256(config).hexdigest(),
        "wheel_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in wheels},
        "synthetic_cases": results, "direct_upstream_vocal_parity": "exact",
        "math_backend_max_abs_delta": (selected - math_reference).abs().max().item(),
        "selected_backend_peak_allocated_bytes": selected_allocated,
        "selected_backend_peak_reserved_bytes": selected_reserved,
    }


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))

