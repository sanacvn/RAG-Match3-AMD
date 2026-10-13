"""Bound complete remote SDK operations in this Unix, main-thread worker."""
import signal
import time


def bounded_call(deadline, function, *args, **kwargs):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Remote operation deadline exhausted")
    # Chroma 1.5.9 has no public total HTTP timeout. A socket/read timeout alone
    # also does not bound several embedding + search requests together.
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(*_):
        raise TimeoutError("Remote operation exceeded its deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(remaining, previous_timer[0]) if previous_timer[0] else remaining)
    try:
        result = function(*args, **kwargs)
        if time.monotonic() >= deadline:
            raise TimeoutError("Remote operation exceeded its deadline")
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(.000001, previous_timer[0] - (time.monotonic() - started)), previous_timer[1])
