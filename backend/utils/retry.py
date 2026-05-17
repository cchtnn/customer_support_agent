import asyncio
from typing import Callable, Tuple, Type
from backend.utils.logger import get_logger

logger = get_logger(__name__)


async def async_retry(
    coro_func: Callable[[], object],
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,)
) -> object:
    """Run an async callable with retries and exponential backoff.

    coro_func should be a zero-argument callable that returns an awaitable
    (e.g. a lambda returning a coroutine like: lambda: chain.ainvoke({...})).
    """
    attempt = 1
    while True:
        try:
            return await coro_func()
        except exceptions as e:
            if attempt >= retries:
                logger.exception("All retries failed (attempt %s): %s", attempt, e)
                raise
            delay = initial_delay * (backoff_factor ** (attempt - 1))
            logger.warning("Transient error on attempt %s/%s: %s. Backing off %.1fs", attempt, retries, e, delay)
            await asyncio.sleep(delay)
            attempt += 1
