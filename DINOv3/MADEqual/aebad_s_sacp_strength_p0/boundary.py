"""Enforce Torch-free CPU commands; model/retrieval APIs are never inputs."""
from contextlib import contextmanager
from importlib.abc import MetaPathFinder
import sys

COUNTERS=dict(query_forward_calls=0,exact_search_calls=0,subspace_fit_calls=0,training_updates=0)


class NoTorch(MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch' or fullname.startswith('torch.'):
            raise ImportError('Torch is forbidden in SACP strength CPU stages')


@contextmanager
def cpu_only():
    if 'torch' in sys.modules:raise RuntimeError('CPU command must start in a Torch-free process')
    guard=NoTorch();sys.meta_path.insert(0,guard)
    try:
        yield
        if 'torch' in sys.modules:raise RuntimeError('CPU stage imported Torch')
    finally:sys.meta_path.remove(guard)
