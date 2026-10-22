"""Explicit original ADEval CUDA jobs; no model or support-feature dependency."""
import time
from pathlib import Path
import numpy as np
from . import storage as st,inputs as ip,prediction,evaluation
from .method import NEW


def job(bundle,out,domain,branch,device='cuda:0'):
    if branch not in NEW:raise ValueError('only two new branches need AUPRO')
    out=st.initialize(out,bundle);prediction.verified(bundle,out,domain)
    if evaluation.restore(bundle,out,domain) is None:raise ValueError('completed exact CPU evaluation required')
    expected=[v for v in st.read_json(out/'evaluation'/domain/'status.json')['files'] if v=='map_manifest.csv']
    if not expected:raise ValueError('CPU map identity missing')
    maps=[v for v in ip.parent.csv_file(out/'evaluation'/domain/'map_manifest.csv') if v['branch']==branch]
    map_id=st.digest(maps);path=out/'aupro'/domain/(branch+'.json')
    if path.exists():
        previous=st.read_json(path)
        if previous['identity']!=bundle.identity or previous['map_identity']!=map_id or previous['nstrips']!=200 or previous['FPR_max']!=.3 or not np.isfinite(previous['AUPRO']) or not 0<=previous['AUPRO']<=1:raise ValueError('invalid/completed AUPRO job differs')
        if previous['result_sha256']!=st.digest({k:v for k,v in previous.items() if k!='result_sha256'}):raise ValueError('AUPRO record modified')
        return previous
    import torch
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    from importlib.metadata import version
    dev=torch.device(device)
    if dev.type!='cuda':raise ValueError('original ADEval requires explicit CUDA; no alternative backend')
    torch.cuda.set_device(dev);torch.cuda.reset_peak_memory_stats(dev)
    if version('adeval')!='1.1.0':raise ValueError('frozen ADEval 1.1.0 required')
    tick=time.perf_counter();responses=[];masks=[];verified=[];byid={v['sample_id']:v for v in maps}
    ctx=dict(medians=bundle.medians,scales=bundle.scales,state=bundle.state)
    with ip.parent.cross.Data(bundle.paths['data']) as data:
        for s in prediction.selected(bundle,domain):
            arrays,meta,_=prediction.load_record(bundle,out,domain,s['sample_id'])
            response=evaluation.shared.score_maps(ctx,arrays[branch],dict(meta,positions=arrays['positions']))[2]
            mask=evaluation.shared.resized_mask(data.gt(s),meta['output_shape']).astype(np.uint8)
            evaluation.shared.verify_map_record(byid[s['sample_id']],response,mask)
            responses.append(response);masks.append(mask);verified.append(byid[s['sample_id']])
    value=official_aupro_adeval(masks,responses,nstrips=200)
    if len(verified)!=len(maps) or not np.isfinite(value) or not 0<=value<=1:raise ValueError('incomplete/nonfinite AUPRO')
    import psutil,adeval
    result=dict(identity=bundle.identity,domain=domain,branch=branch,map_identity=map_id,AUPRO=float(value),nstrips=200,FPR_max=.3,
        backend='original_ADEval_CUDA',seconds=time.perf_counter()-tick,verified_map_GT_images=len(verified),
        process_memory_info=psutil.Process().memory_info()._asdict(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(dev),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(dev),device=str(dev),torch=torch.__version__,CUDA=torch.version.cuda,
        ADEval=version('adeval'),ADEval_path=adeval.__file__,GPU=torch.cuda.get_device_name(dev),model_inference=False,feature_extraction=False)
    result['result_sha256']=st.digest(result);st.write_json(path,result)
    return result
