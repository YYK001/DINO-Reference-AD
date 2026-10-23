"""Bounded image work, preserving order without the old eight-worker cap.

Same queue rule as neighborhood_segment_p0.prediction.bounded_images; kept
local so historical modules and their defaults are not modified.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def bounded_images(samples, function, workers):
    if workers < 1: raise ValueError('positive CPU workers required')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        source=iter(samples); pending=deque()
        for _ in range(workers):
            sample=next(source,None)
            if sample is not None:pending.append(pool.submit(function,sample))
        while pending:
            yield pending.popleft().result()
            sample=next(source,None)
            if sample is not None:pending.append(pool.submit(function,sample))
