import asyncio
import inspect
import os

DEFAULT_IO_CONCURRENCY = 32


def resolve_io_concurrency(value):
    if value is not None:
        resolved = value
    else:
        env_value = os.environ.get("VCZSTORE_IO_CONCURRENCY")
        if env_value is None:
            resolved = DEFAULT_IO_CONCURRENCY
        else:
            try:
                resolved = int(env_value)
            except ValueError as e:
                raise ValueError("VCZSTORE_IO_CONCURRENCY must be an integer") from e
    if resolved < 1:
        raise ValueError("io_concurrency must be >= 1")
    return resolved


async def run_bounded(
    items,
    worker_fn,
    *,
    max_concurrency,
    on_item_done=None,
):
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    async def _call_on_item_done(item):
        if on_item_done is None:
            return
        result = on_item_done(item)
        if inspect.isawaitable(result):
            await result

    item_iter = iter(items)
    pending = {}

    def submit(item):
        pending[asyncio.create_task(worker_fn(item))] = item

    async def cancel_pending():
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    while len(pending) < max_concurrency:
        try:
            submit(next(item_iter))
        except StopIteration:
            break

    while pending:
        done, _ = await asyncio.wait(
            pending.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        completed_items = []
        first_error = None
        for task in done:
            item = pending.pop(task)
            try:
                task.result()
            except asyncio.CancelledError:
                continue
            except Exception as e:
                if first_error is None:
                    first_error = e
            else:
                completed_items.append(item)

        if first_error is not None:
            await cancel_pending()
            raise first_error

        try:
            for item in completed_items:
                await _call_on_item_done(item)
        except Exception:
            await cancel_pending()
            raise

        while len(pending) < max_concurrency:
            try:
                submit(next(item_iter))
            except StopIteration:
                break
