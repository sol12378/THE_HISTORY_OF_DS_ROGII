from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer(label: str):
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.2f}s")
