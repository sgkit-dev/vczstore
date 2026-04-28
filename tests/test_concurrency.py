import asyncio
import importlib

import pytest

from vczstore.concurrency import resolve_io_concurrency, run_bounded

append_module = importlib.import_module("vczstore.append")


def test_resolve_io_concurrency_explicit_beats_env(monkeypatch):
    monkeypatch.setenv("VCZSTORE_IO_CONCURRENCY", "7")
    assert resolve_io_concurrency(3) == 3


def test_resolve_io_concurrency_uses_env(monkeypatch):
    monkeypatch.setenv("VCZSTORE_IO_CONCURRENCY", "7")
    assert resolve_io_concurrency(None) == 7


def test_resolve_io_concurrency_uses_default(monkeypatch):
    monkeypatch.delenv("VCZSTORE_IO_CONCURRENCY", raising=False)
    assert resolve_io_concurrency(None) == 32


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_resolve_io_concurrency_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("VCZSTORE_IO_CONCURRENCY", value)
    with pytest.raises(ValueError, match="VCZSTORE_IO_CONCURRENCY|io_concurrency"):
        resolve_io_concurrency(None)


def test_run_bounded_limits_in_flight_workers():
    async def main():
        active = 0
        max_active = 0

        async def worker(item):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1

        await run_bounded(range(8), worker, max_concurrency=3)
        return max_active

    assert asyncio.run(main()) == 3


def test_run_bounded_calls_completion_callback_once_per_success():
    async def main():
        completed = []

        async def worker(item):
            await asyncio.sleep(0.02 if item == 0 else 0.01)

        await run_bounded(
            [0, 1, 2],
            worker,
            max_concurrency=2,
            on_item_done=completed.append,
        )
        return completed

    assert asyncio.run(main()) == [1, 0, 2]


def test_run_bounded_cancels_outstanding_tasks_after_failure():
    async def main():
        started = []
        cancelled = []
        completed = []

        async def worker(item):
            started.append(item)
            try:
                if item == "fail":
                    await asyncio.sleep(0.01)
                    raise RuntimeError("boom")
                await asyncio.sleep(0.05)
                completed.append(item)
            except asyncio.CancelledError:
                cancelled.append(item)
                raise

        with pytest.raises(RuntimeError, match="boom"):
            await run_bounded(
                ["ok-1", "fail", "ok-2"],
                worker,
                max_concurrency=3,
                on_item_done=lambda item: completed.append(f"callback:{item}"),
            )
        return started, cancelled, completed

    started, cancelled, completed = asyncio.run(main())
    assert started == ["ok-1", "fail", "ok-2"]
    assert set(cancelled) == {"ok-1", "ok-2"}
    assert completed == []


def test_append_sync_function_resolves_io_concurrency(monkeypatch):
    seen = {}

    async def fake_async(*call_args, **call_kwargs):
        seen["args"] = call_args
        seen["kwargs"] = call_kwargs

    monkeypatch.setattr(append_module, "_append_async", fake_async)
    monkeypatch.setenv("VCZSTORE_IO_CONCURRENCY", "5")

    append_module.append("left", "right")
    assert seen["kwargs"]["io_concurrency"] == 5

    append_module.append("left", "right", io_concurrency=2)
    assert seen["kwargs"]["io_concurrency"] == 2
