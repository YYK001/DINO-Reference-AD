"""Explicit extraction/preparation only; no Torch imported by preflight."""
from pathlib import Path
from types import SimpleNamespace, FunctionType
import numpy as np
from . import inputs as ip, storage as st
from .protocol import LAYERS, DOMAINS, CONFIG
from .matching import teacher

PRIOR_GPU_PEAK = dict(allocated=0,reserved=0)


def reliable_precision(device):
    import torch
    if str(device).startswith('cuda'):torch.cuda.set_device(torch.device(device))
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True  # Frozen backbone setting, not GEMM permission.


def host(a):
    # The frozen loader resets its counters. Preserve earlier work in THIS stage.
    import torch
    if torch.cuda.is_initialized():
        PRIOR_GPU_PEAK['allocated']=max(PRIOR_GPU_PEAK['allocated'],torch.cuda.max_memory_allocated())
        PRIOR_GPU_PEAK['reserved']=max(PRIOR_GPU_PEAK['reserved'],torch.cuda.max_memory_reserved())
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.runtime import host as original
    return original(a)


def extraction(model,image,device):
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.prediction import extraction as original
    x,m=original(model,image,device)
    return {l:x[f'layer_{l}'] for l in LAYERS},m


def reuse_compatible(a,r,c,frozen):
    """No graph-code identity requirement; verify actual extraction dependencies."""
    if r['catalog']['source']!=c['source'] or r['catalog']['supports']!=c['supports']:
        raise ValueError('recent support/Source image identity/order differs')
    conf=r['configuration']
    for n in ('numpy','scipy','torch','torchvision'):
        if conf['software'].get(n)!=frozen['software'][n]:raise ValueError('recent extraction environment differs: '+n)
    if 'host_config' in conf and conf['host_config']!=frozen['host_config']:raise ValueError('recent host configuration differs')
    code=conf['code']
    if 'external_baselines/SuperADD/config.json' in code and code['external_baselines/SuperADD/config.json']!=st.sha(a.official_root/'config.json'):
        raise ValueError('recent extraction host config changed')
    if conf.get('external_source_code'):
        for key,expected in conf['external_source_code'].items():
            name,relative=key.split('/',1);root=a.dinov3_root if name=='dinov3' else a.official_root
            if st.sha(root/relative)!=expected:raise ValueError('recent external extraction source changed: '+key)
    required=['DINOv3/relation_reliability/superadd_host.py',
        'DINOv3/ShiftADD/aebad_s_p0/memory.py','external_baselines/superadd_external/memory_variants.py']
    required += [p for p in code if p.startswith(('external_baselines/dinov3/',
        'external_baselines/SuperADD/tracks/industrial/src/industrial/'))]
    for rel in required:
        if rel.startswith('external_baselines/dinov3/'):path=a.dinov3_root/rel.removeprefix('external_baselines/dinov3/')
        elif rel.startswith('external_baselines/SuperADD/'):path=a.official_root/rel.removeprefix('external_baselines/SuperADD/')
        else:path=ip.ROOT/rel
        if code.get(rel)!=st.sha(path):raise ValueError('recent extraction implementation differs: '+rel)
    # Parent host loader validates the official checkpoint digest as well.
    from DINOv3.MADEqual.aebad_s_target_fewshot.replay12 import CHECKPOINT_SHA256
    if frozen['checkpoint_sha256']!=CHECKPOINT_SHA256:raise ValueError('different DINOv3 checkpoint')


def prepare(a):
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs as fi, fresh_runtime as fr
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import storage as oldst
    c=fi.prepare_data(a.run_root/'data_index',a.data,a.cache_root)
    frozen=ip.freeze(a,c);ident=frozen['identity'];recent=[];rejected=[]
    for p in a.recent_run:
        r=ip.recent(p)
        try:reuse_compatible(a,r,c,frozen)
        except ValueError as e:
            rejected.append(dict(run=str(p),reason=str(e)));continue
        recent.append(r)
    st.write_json(a.run_root/'reuse_audit.json',dict(compatible=[str(r['run']) for r in recent],rejected=rejected))
    source_path=a.cache_root/'source_original'
    if not (source_path/'manifest.json').exists():
        for r in recent:
            p=r['cache']/'source_bank.npz'
            if not p.exists():continue
            rec=oldst.restore(p,r['setup']['identity'])
            with np.load(p,allow_pickle=False) as z:arrays={f'L{l}':z[f'bank_{l}'].copy() for l in LAYERS}
            st.put(source_path,arrays,dict(identity=ident,origin=str(p),sha256=rec['sha256'],builder=rec['metadata']))
            break
        else:
            # Same production builder, original 521 ordering/66 exclusion/455 augmented images.
            build_run=a.run_root/'source_build';st.write_json(build_run/'setup.json',dict(identity=ident))
            st.write_json(build_run/'fresh_catalog.json',c)
            args=SimpleNamespace(**vars(a));args.run_root=build_run
            fn=fr.source
            FunctionType(fn.__code__,dict(fn.__globals__,host=host),fn.__name__,fn.__defaults__)(args)
            x,rec=oldst.arrays(a.cache_root/'source_bank.npz',ident)
            st.put(source_path,{f'L{l}':x[f'bank_{l}'] for l in LAYERS},dict(identity=ident,
                origin='fresh_original_augmented_100round_builder',builder=rec['metadata']))
    source,srec=st.get(source_path)
    for l in LAYERS:
        if source[f'L{l}'].shape!=(100000,1280) or source[f'L{l}'].dtype!=np.float32:
            raise ValueError('original Source bank dimensions/precision differ')
    model=None;calls=0;reused=0
    # Only normal Source and fixed normal supports. No test query/GT decoded.
    all_rows=c['source']+sum(c['supports'].values(),[])
    with fi.Data(Path(c['data_root'])) as data:
        for number,row in enumerate(all_rows,1):
            p=ip.feature_path(a,row)
            if (p/'manifest.json').exists():
                _,rec=st.get(p)
                if rec['metadata']['image']!=row or rec['metadata']['identity']!=ident:raise ValueError('feature input identity changed')
                continue
            layers=None;geometry=None;origin='fresh_original_host'
            if row.get('support_index') is not None:
                for r in recent:
                    old=r['cache']/'supports'/row['domain']/f"{row['support_index']:03d}.npz"
                    if not old.exists():continue
                    z,rec=oldst.arrays(old,r['setup']['identity'])
                    if str(z['relative_path'].item())!=row['relative_path']:raise ValueError('support order differs')
                    layers={l:z[f'layer_{l}'] for l in LAYERS};grid=z['grid'].tolist()
                    geometry=dict(grid=grid,positions=np.arange(np.prod(grid),dtype=np.int64))
                    origin=dict(path=str(old),sha256=rec['sha256']);reused+=1;break
            if layers is None:
                if model is None:model,_=host(a)
                layers,geometry=extraction(model,data.image(row),a.device);calls+=1
            grid=geometry['grid'];positions=np.asarray(geometry['positions'],np.int64)
            n=int(np.prod(grid))
            if not np.array_equal(np.sort(positions),np.arange(n)):raise ValueError('incomplete physical feature grid')
            for l in LAYERS:
                if layers[l].shape!=(n,1280) or layers[l].dtype!=np.float32 or not np.isfinite(layers[l]).all():
                    raise ValueError('invalid raw complete feature layer: '+str(l))
            arrays=teacher(layers,positions,grid,row['relative_path']) if not row.get('calibration',False) else {}
            if row.get('calibration',False) or row.get('support_index') is not None:
                arrays.update({f'full_L{l}':layers[l] for l in LAYERS});arrays['full_positions']=positions
            st.put(p,arrays,dict(identity=ident,image=row,grid=list(grid),origin=origin,
                training_allowed=not row.get('calibration',False),output_shape=[row['image_height']//4,row['image_width']//4]))
            print(f'[normal prepare] {number}/{len(all_rows)} {row["relative_path"]}',flush=True)
    st.sealed_json(a.run_root/'prepared.json',dict(identity=ident,source_seal=srec['seal'],
        features={r['relative_path']:st.read_json(ip.feature_path(a,r)/'manifest.json')['seal'] for r in all_rows}))
    return dict(status='normal_features_prepared',new_forward_calls=calls,reused_supports=reused,
        Source_bank_origin=srec['metadata']['origin'])


def source_scales(x,chunk=8192):
    """Small vector/statistic FP64 accumulation only; saved network statistics FP32."""
    total=np.zeros(x.shape[1],np.float64)
    for i in range(0,len(x),chunk):total+=np.asarray(x[i:i+chunk],np.float64).sum(0)
    mu=total/len(x);squared=0.
    for i in range(0,len(x),chunk):
        delta=np.asarray(x[i:i+chunk],np.float64)-mu;squared+=float(np.sum(delta*delta))
    sigma=np.sqrt(squared/(x.size))
    if not np.isfinite(mu).all() or not np.isfinite(sigma) or sigma<=0:raise ValueError('invalid original Source RMS')
    return mu.astype(np.float32),np.float32(sigma)
