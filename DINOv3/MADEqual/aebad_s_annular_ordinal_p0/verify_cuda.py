"""Fixed small reference checks, invoked on the selected GPU before index build/smoke."""
import time
import numpy as np
import torch
from . import inputs as ip,pixels as cpu,geometry as cg,evidence as ce
from . import pixels_cuda as px,geometry_cuda as geo,evidence_cuda as ev
from . import index_cuda as ix
from .protocol import cuda_index_config
st=ip.st

def fingerprint(device):
    if torch.device(device).type!='cuda':return dict(device='cpu_tensor_test',gpu_executed=False,torch=torch.__version__)
    props=torch.cuda.get_device_properties(device)
    return dict(device=str(device),name=props.name,uuid=str(getattr(props,'uuid','unavailable')),
        capability=list(torch.cuda.get_device_capability(device)),memory_bytes=props.total_memory,
        torch=torch.__version__,compiled_CUDA=torch.version.cuda,gpu_executed=True)

def small_checks(device):
    rng=np.random.default_rng(128);gray=rng.random((192,192),dtype=np.float32);tensor=torch.as_tensor(gray,device=device)
    rgb=rng.integers(0,256,(24,32,3),np.uint8);np.testing.assert_array_equal(cpu.grayscale(rgb),px.grayscale(torch.as_tensor(rgb,device=device)).cpu().numpy())
    counts=np.arange(25,dtype=np.float32)
    np.testing.assert_array_equal(counts/np.float32(24),px.divide_by_constant(torch.as_tensor(counts,device=device),24.).cpu().numpy())
    sums=rng.random(256,dtype=np.float32)*np.float32(9)
    np.testing.assert_array_equal(sums/np.float32(9),px.divide_by_constant(torch.as_tensor(sums,device=device),9.).cpu().numpy())
    boxes=[(0,0),(64,64),(128,128)];infos=px.outer_to_host(px.outer_batch(tensor,boxes))
    for box,info in zip(boxes,infos):
        old=cg.prepare_outer(gray,box)
        for key in ('points','bits','directions','contrast','validation_pairs','validation_bits'):np.testing.assert_array_equal(old[key],info[key],err_msg=key)
    for version in range(9):
        spec=cpu.grid_spec(gray.shape,version)
        np.testing.assert_array_equal(cpu.describe_grid(gray,spec),px.describe_grid(tensor,spec).cpu().numpy(),err_msg='descriptor bits')
    points=np.array([[-.01,0],[0,0],[191,191],[191.01,0],[63.25,70.75]],np.float32)
    a,av=cpu.sample(gray,points);b,bv=px.sample(tensor,torch.as_tensor(points,device=device))
    np.testing.assert_array_equal(av,bv.cpu().numpy());np.testing.assert_array_equal(a,b.cpu().numpy())
    vectors=rng.integers(0,256,(48,16),np.uint8);v=torch.as_tensor(vectors,device=device)
    expected=np.unpackbits(vectors[:,None]^vectors[None],axis=-1).sum(-1)
    np.testing.assert_array_equal(expected,ix.hamming(v[:,None],v[None]).cpu().numpy())
    config=cuda_index_config();offsets,order=ix.build_csr(v,config['bit_positions'])
    current,visits=ix.probe(v[:3],v,offsets,order,torch.arange(len(v),device=device),config['bit_positions'],budget=4,tile=7)
    assert visits>0
    qi,distance,rows,physical=[a.cpu().numpy() for a in current]
    for j in range(3):assert rows[qi==j][0]==j and distance[qi==j][0]==0
    free=next(i for i in range(128) if i not in {b for table in config['bit_positions'] for b in table})
    tail=torch.zeros((64,16),device=device,dtype=torch.uint8);tail[:63,free//8]=1<<(free%8)
    offsets,order=ix.build_csr(tail,config['bit_positions'])
    hits,visits=ix.probe(torch.zeros((1,16),device=device,dtype=torch.uint8),tail,offsets,order,
        torch.arange(64,device=device),config['bit_positions'],budget=4,tile=5)
    assert visits==256 and int(hits[2][0].cpu())==63 and int(hits[1][0].cpu())==0,'bucket tail omitted'
    ref=[dict(index=i,instance_id=str(i),kind='Source',shape=gray.shape,read_gray=lambda:gray) for i in range(2)]
    box=(64,64);info=cg.prepare_outer(gray,box)
    matches=[dict(anchor=j,ref=i,point=p.tolist(),version=4,hamming=0) for i in range(2) for j,p in enumerate(info['points'])]
    old=cg.freeze(info,matches,ref,box);new=geo.freeze_batch([info],[matches],ref,[box],lambda i:tensor,device)[0]
    assert old['stats']==new['stats'] and old['qualified']==new['qualified'],'geometry accept/reject states differ'
    for name in ('source','single','multi'):
        assert len(old[name]['candidates'])==len(new[name]['candidates'])
        for a,b in zip(old[name]['candidates'],new[name]['candidates']):np.testing.assert_allclose(a['T'],b['T'],rtol=0,atol=1e-10)
    normal=ev.normal_expectations([new],ref,[box],lambda i:tensor,device)
    changed=gray.copy();changed[64:128,64:128]=1-changed[64:128,64:128]
    altered=torch.as_tensor(changed,device=device);after=px.outer_to_host(px.outer_batch(altered,[box]))[0]
    for key in ('points','bits','validation_bits'):np.testing.assert_array_equal(info[key],after[key])
    sparse=gray.copy();sparse[64:128:3,64:128:5]=1-sparse[64:128:3,64:128:5]
    for image in (gray,changed,sparse):
        result=ev.query_evidence(torch.as_tensor(image,device=device),normal,[box])
        for i,name in enumerate(('source','single','multi')):
            prediction=ce.normal_expectation(new[name],ref,box,multi=name!='single');expected=ce.query_evidence(image,prediction,box)
            np.testing.assert_array_equal(prediction['agreed'],normal['agreed'][0,i].cpu().numpy(),err_msg='valid mask')
            np.testing.assert_array_equal(expected['flip_counts'],result['flip_counts'][0,i].cpu().numpy(),err_msg='ordinal flips')
            assert expected['score']==float(result['scores'][0,i].cpu()),'final E differs'
    return dict(grayscale_and_constant_division='exact; device denominator',descriptor_bits='exact',Hamming='integer exact',geometry='same candidate counts; T atol1e-10',
        masks_and_flips='exact',final_E='exact',center_isolation=True,bucket_tail_not_clipped=True,
        scope='fixed synthetic small inputs, not real detection effectiveness or full-scale throughput')

def verify(a,device,store=None):
    path=a.run_root/'gpu_check.json';hardware=fingerprint(device)
    if path.exists():
        old=st.sealed_json(path)
        if old['identity']!=ip.identity(a) or old['hardware']!=hardware:raise ValueError('GPU check execution/hardware identity differs')
        return old
    tick=time.perf_counter()
    try:
        checks=small_checks(device)
        if store is not None:
            row=sorted(ip.assets(a).rows('calibration'),key=lambda r:r['sample_id'])[0];gray=store.gray(row);image=store.image(row)
            positions=cpu.windows(gray.shape);positions=[positions[i] for i in (0,len(positions)//2,len(positions)-1)]
            infos=px.outer_to_host(px.outer_batch(image,positions))
            for position,info in zip(positions,infos):
                original=cg.prepare_outer(gray,position)
                for key in ('points','bits','validation_bits'):np.testing.assert_array_equal(original[key],info[key],err_msg='real image '+key)
            checks['real_calibration_blocks']=dict(image=ip.image(row),windows=[list(b) for b in positions],full_image_repeated=False)
    except Exception as error:
        st.write_json(a.run_root/'gpu_numerical_mismatches.json',dict(status='failed',reason=str(error),hardware=hardware,no_tolerance_relaxation=True))
        raise
    return st.sealed_json(path,dict(status='GPU_small_check_passed',identity=ip.identity(a),hardware=hardware,checks=checks,seconds=time.perf_counter()-tick,
        observed_engine_startup_seconds=getattr(a,'gpu_startup_seconds',None)))
