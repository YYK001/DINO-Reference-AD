"""Disable capacity estimates without falsifying measurements or cache identity.

The original frozen source files stay byte-identical. Only the exact reviewed
resource-guard statements below are removed from runtime functions. Mathematical
operations, input/hash validation and actual allocation failures stay intact.
"""
import ast
from contextlib import ExitStack,contextmanager
import hashlib
import inspect
import textwrap
import types
from unittest.mock import patch

POLICY='capacity_observation_only_v1'

GUARDS={
    'capture':(
        'capture requires estimated ',
        'insufficient free VRAM: selected admission requires ',
        'capture needs 12 GiB free host RAM for full supports and host'),
    'evaluate':(
        'insufficient RAM/budget for exact full pixels; no approximation',
        'insufficient G-drive pixel scratch space',
        'RAM headroom dropped before exact pixel sort; preserving failure without approximation'),
    'exact_binary':('exact pixel sort needs conservative ',),
    'aupro':('insufficient VRAM for requested AUPRO cap; no metric/resolution change',
             'insufficient RAM for full-resolution original AUPRO'),
    'diagnostic':('CPU analysis requires 2 GiB available RAM and 3 GiB disk headroom',),
}


def disable_guards(function,allowed):
    source=textwrap.dedent(inspect.getsource(function));tree=ast.parse(source);removed=[]
    class Remove(ast.NodeTransformer):
        def visit_If(self,node):
            if len(node.body)==1 and not node.orelse and isinstance(node.body[0],ast.Raise):
                exc=node.body[0].exc
                if isinstance(exc,ast.Call) and isinstance(exc.func,ast.Name) and exc.func.id in ('MemoryError','OSError'):
                    text=''.join(n.value for arg in exc.args for n in ast.walk(arg) if isinstance(n,ast.Constant) and isinstance(n.value,str))
                    matches=[message for message in allowed if text.startswith(message)]
                    if matches:
                        removed.append(dict(message=matches[0],condition=ast.unparse(node.test)))
                        return ast.copy_location(ast.Pass(),node)
            return self.generic_visit(node)
    tree=Remove().visit(tree);ast.fix_missing_locations(tree)
    if sorted(r['message'] for r in removed)!=sorted(allowed):
        raise ValueError('resource policy source differs from reviewed guard list: '+function.__name__)
    namespace={};exec(compile(tree,inspect.getsourcefile(function)+':'+POLICY,'exec'),function.__globals__,namespace)
    compiled=namespace[function.__name__]
    replacement=types.FunctionType(compiled.__code__,function.__globals__,function.__name__,function.__defaults__,function.__closure__)
    replacement.__kwdefaults__=function.__kwdefaults__;replacement.__annotations__=function.__annotations__
    audit=dict(function=function.__module__+'.'+function.__name__,removed_guards=removed,
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        runtime_source_sha256=hashlib.sha256(ast.unparse(tree).encode()).hexdigest())
    return replacement,audit


@contextmanager
def observation_only():
    # All these modules defer Torch imports until explicit AUPRO/capture calls.
    from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import capture,analysis,aupro
    from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic import metrics
    from . import runner
    bindings=((capture,'run','capture'),(analysis,'run','evaluate'),
              (metrics,'exact_binary','exact_binary'),(aupro,'run','aupro'),(runner,'analyze','diagnostic'))
    audit=dict(policy=POLICY,capacity_estimates_enforced=False,measurements_falsified=False,
        actual_allocation_errors_propagate=True,frozen_source_files_modified=False,functions=[])
    with ExitStack() as stack:
        for module,name,role in bindings:
            replacement,record=disable_guards(getattr(module,name),GUARDS[role])
            stack.enter_context(patch.object(module,name,replacement));audit['functions'].append(record)
            if role=='exact_binary':stack.enter_context(patch.object(analysis,'exact_binary',replacement))
        yield audit
