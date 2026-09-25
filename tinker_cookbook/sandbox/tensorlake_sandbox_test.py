"""Offline tests for TensorlakeSandboxPool. TensorlakeSandbox.create is mocked."""

import asyncio
from unittest import mock

import pytest

pytest.importorskip("tensorlake")

from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.sandbox import tensorlake_sandbox
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

_OK = SandboxResult(stdout="", stderr="", exit_code=0)


class _FakeSandbox:
    def __init__(self, live: set["_FakeSandbox"]) -> None:
        self._live = live
        live.add(self)

    async def cleanup(self) -> None:
        self._live.discard(self)

    async def write_file(self, *args: object, **kwargs: object) -> SandboxResult:
        return _OK

    async def run_command(self, *args: object, **kwargs: object) -> SandboxResult:
        return _OK

    async def checkpoint(self) -> str:
        return "snapshot"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup_command", "expected_created"),
    [
        (None, 2),  # max_concurrency creations are in flight
        ("pip install numpy", 1),  # only the template creation is in flight
    ],
)
async def test_terminate_stops_queued_and_in_flight_work(
    setup_command: str | None, expected_created: int
) -> None:
    live: set[_FakeSandbox] = set()
    created: list[_FakeSandbox] = []
    pending = 0
    release = asyncio.Event()

    async def fake_create(**kwargs: object) -> _FakeSandbox:
        nonlocal pending
        pending += 1
        await release.wait()
        sandbox = _FakeSandbox(live)
        created.append(sandbox)
        return sandbox

    with mock.patch.object(tensorlake_sandbox.TensorlakeSandbox, "create", fake_create):
        pool = tensorlake_sandbox.TensorlakeSandboxPool(
            max_concurrency=2, setup_command=setup_command
        )
        requests = [
            asyncio.create_task(pool.run_in_workdir({"a.py": "x"}, ["python", "a.py"]))
            for _ in range(6)
        ]
        while pending < expected_created:
            await asyncio.sleep(0)

        # terminate() waits for creations in progress, so let them finish.
        terminate = asyncio.create_task(pool.terminate())
        await asyncio.sleep(0)
        release.set()
        await terminate

        assert not live
        results = await asyncio.gather(*requests, return_exceptions=True)

    assert len(created) == expected_created
    assert not live
    assert all(isinstance(r, SandboxError) for r in results)


@pytest.mark.asyncio
async def test_run_after_terminate_raises() -> None:
    pool = tensorlake_sandbox.TensorlakeSandboxPool(max_concurrency=1)
    await pool.terminate()
    with pytest.raises(SandboxError, match="terminated"):
        await pool.run_in_workdir({}, ["true"])


@pytest.mark.asyncio
async def test_terminate_deletes_setup_snapshot() -> None:
    deleted: list[str] = []

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def delete_snapshot(self, snapshot_id: str) -> None:
            deleted.append(snapshot_id)

    live: set[_FakeSandbox] = set()

    async def fake_create(**kwargs: object) -> _FakeSandbox:
        return _FakeSandbox(live)

    with (
        mock.patch.object(tensorlake_sandbox.TensorlakeSandbox, "create", fake_create),
        mock.patch.object(tensorlake_sandbox, "AsyncSandboxClient", _FakeClient),
    ):
        pool = tensorlake_sandbox.TensorlakeSandboxPool(setup_command="pip install numpy")
        await pool.run_in_workdir({"a.py": "x"}, ["python", "a.py"])
        await pool.terminate()

    assert deleted == ["snapshot"]
    assert not live
