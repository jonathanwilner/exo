from pytest import MonkeyPatch

from exo.shared.types.backends import Backend
from exo.utils.info_gatherer import info_gatherer


async def test_xpu_runtime_advertises_vllm_backend(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(info_gatherer, "IS_DARWIN", False)
    monkeypatch.setattr(info_gatherer, "_has_nvml_cuda", lambda: False)
    monkeypatch.setattr(info_gatherer, "_has_torch_xpu", lambda: True)

    gathered = await info_gatherer.NodeBackends.gather()

    assert gathered.backends == [Backend.MlxCpu, Backend.Vllm]


async def test_cuda_runtime_advertises_mlx_cuda_and_vllm(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(info_gatherer, "IS_DARWIN", False)
    monkeypatch.setattr(info_gatherer, "_has_nvml_cuda", lambda: True)
    monkeypatch.setattr(info_gatherer, "_has_torch_xpu", lambda: False)

    gathered = await info_gatherer.NodeBackends.gather()

    assert gathered.backends == [
        Backend.MlxCpu,
        Backend.MlxCuda,
        Backend.Vllm,
    ]


async def test_no_accelerator_does_not_advertise_vllm(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(info_gatherer, "IS_DARWIN", False)
    monkeypatch.setattr(info_gatherer, "_has_nvml_cuda", lambda: False)
    monkeypatch.setattr(info_gatherer, "_has_torch_xpu", lambda: False)

    gathered = await info_gatherer.NodeBackends.gather()

    assert gathered.backends == [Backend.MlxCpu]
