"""Stream full images in fixed window batches; resume committed chunks/items."""
import os
import time
import zipfile
import zlib
from pathlib import Path
import numpy as np
from . import inputs as ip,pixels as px,geometry as geo,evidence as ev
from .index import Store
from .protocol import E_NAMES
st=ip.st
SETS=('source','single','multi')

def result_path(a,row):return a.cache_root/'images'/ip.key(row)/'result.json'

def validate_chunk(a,directory,record):
    """Check committed NPZ size/readability/layout once per unchanged file."""
    path=Path(directory)/record['arrays']
    try:
        info=path.stat()
        if not path.is_file() or info.st_size!=record['array_bytes']:raise ValueError('recorded NPZ size differs')
        stamp=(info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_ino,record.get('seal'))
        memo=getattr(a,'_validated_chunks',None)
        if memo is None:a._validated_chunks={};memo=a._validated_chunks
        if memo.get(path)==stamp:return
        n=len(record['boxes'])
        if not 1<=n<=64:raise ValueError('invalid window batch size')
        layout=dict(boxes=((n,2),'int32'),scores=((n,3),'float32'),
            evidence_present=((n,3),'bool'),missing_mask_packed=((n,3,12288),'uint8'),
            flip_counts=((n,3,64,64),'uint8'),candidate_references=((n,3,12),'int16'),
            transforms=((n,3,12,2,3),'float64'),agreement_fraction=((n,3),'float32'),
            disagreement_fraction=((n,3),'float32'),distinct_references=((n,3),'int16'))
        archive=np.load(path,allow_pickle=False)
        if not isinstance(archive,np.lib.npyio.NpzFile):raise ValueError('NPZ archive required')
        with archive as arrays:
            for name,(shape,dtype) in layout.items():
                # Loading each required member also detects damaged compressed payloads/CRC.
                value=arrays[name]
                if value.shape!=shape or value.dtype!=np.dtype(dtype):raise ValueError('invalid '+name+' array layout')
                if name=='boxes' and not np.array_equal(value,record['boxes']):raise ValueError('window coordinates differ')
        memo[path]=stamp
    except (OSError,ValueError,EOFError,KeyError,zipfile.BadZipFile,zlib.error) as e:
        raise ValueError(f'committed window NPZ invalid: {path}: {e}') from e

def completed(a,row):
    p=result_path(a,row)
    if not p.exists():return None
    memo=getattr(a,'_completed',None)
    if memo is None:a._completed={};memo=a._completed
    if p in memo:r,chunks=memo[p]
    else:
        r=st.sealed_json(p)
        if r['identity']!=ip.identity(a) or r['image']!=ip.image(row):raise ValueError('completed image dependencies differ')
        chunks=[st.sealed_json(p.parent/name) for name in r['chunks']]
        if any(c['identity']!=r['identity'] for c in chunks):raise ValueError('committed image window chunk identity differs')
    for c in chunks:validate_chunk(a,p.parent,c)
    memo[p]=(r,chunks);return r

def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)

def diagnostics(frozen,results,matches,saturated):
    return dict(no_matches=not matches,no_same_image_geometry=not frozen['qualified'],
        geometry_unstable=frozen['stats']['geometry_unstable']>0,
        insufficient_same_image=frozen['stats']['insufficient_same_image']>0,
        hypotheses=frozen['stats']['hypotheses'],
        retrieval_budget_saturated=bool(saturated or frozen['stats']['source_image_budget_saturated']),
        near_best_overflow=bool(frozen['multi']['search_unresolved']),
        normal_ambiguous=results[2]['disagreement_fraction']>0,
        multi_insufficient_instances=results[2]['distinct_references']<2)

def visualize(a,row,gray,box,frozen,normal,result,references,number):
    """Fixed image IDs/window positions only; no score-based case selection."""
    from PIL import Image,ImageDraw
    dest=a.run_root/'fixed_cases'/row['sample_id']/str(number);dest.mkdir(parents=True,exist_ok=True)
    x,y=box;h,w=gray.shape;left,top=max(0,x-32),max(0,y-32)
    outer=np.uint8(np.clip(gray[top:min(h,y+96),left:min(w,x+96)]*255,0,255))
    pic=Image.fromarray(outer).convert('RGB');draw=ImageDraw.Draw(pic)
    draw.rectangle((x-left,y-top,x+63-left,y+63-top),outline='red');draw.rectangle((x-8-left,y-8-top,x+71-left,y+71-top),outline='yellow')
    pic.save(dest/'query_outer_center.png')
    Image.fromarray(np.uint8(normal['disagreement'].mean(0)*255)).save(dest/'normal_disagreement.png')
    Image.fromarray(np.uint8(result['evidence']*255)).save(dest/'query_flips.png')
    Image.fromarray(np.uint8(result['missing_mask'].mean(0)*255)).save(dest/'missing_comparisons.png')
    yy,xx=np.mgrid[:64,:64];points=np.stack((xx+x,yy+y),-1)
    for j,hypothesis in enumerate(frozen['multi']['candidates']):
        ref=references[hypothesis['ref']];values,valid=px.sample(ref['read_gray'](),geo.apply(np.asarray(hypothesis['T']),points))
        # Invalid pixels are displayed in magenta and never used as evidence.
        rgb=np.repeat(np.uint8(np.clip(values,0,1)*255)[...,None],3,axis=2);rgb[~valid]=[255,0,255]
        Image.fromarray(rgb).save(dest/f'normal_{j:02d}_mapped_center.png')
    st.write_json(dest/'frozen_candidates.json',frozen)

class Engine:
    def __init__(self,a,store=None):self.a=a;self.store=store or Store(a)
    def close(self):self.store.close()
    def process(self,row,case=False):
        old=completed(self.a,row)
        if old is not None:return old
        a=self.a;tick=time.perf_counter();index=self.store.load();gray_tick=time.perf_counter();gray=self.store.gray(row)
        gray_seconds=time.perf_counter()-gray_tick
        boxes=px.windows(gray.shape);dest=result_path(a,row).parent;dest.mkdir(parents=True,exist_ok=True)
        chunks=[];all_summaries=[]
        for start in range(0,len(boxes),64):
            name=f'chunk_{start:06d}.json';path=dest/name
            if path.exists():
                saved=st.sealed_json(path)
                if saved['identity']!=ip.identity(a) or saved['index']!=index['seal'] or saved['boxes']!=[list(b) for b in boxes[start:start+64]]:raise ValueError('window chunk dependencies differ')
                validate_chunk(a,dest,saved)
                chunks.append(name);all_summaries.append(saved);continue
            began=time.perf_counter();batch=boxes[start:start+64]
            infos=[geo.prepare_outer(gray,box) for box in batch]
            bits=np.concatenate([info['bits'] for info in infos]);lookups=self.store.lookup_many(bits) if len(bits) else []
            cursor=0;masks=[];flips=[];stats=[];scores=[];valids=[];agreements=[];disagreements=[];refs=[];usage=[];candidate_refs=[];transforms=[]
            for offset,(box,info) in enumerate(zip(batch,infos)):
                count=len(info['bits']);matches,saturated=self.store.matches(info,lookups[cursor:cursor+count]);cursor+=count
                frozen=geo.freeze(info,matches,self.store.references,box)
                # All candidates are frozen before any mapped normal center is read.
                expectations=[ev.normal_expectation(frozen[k],self.store.references,box,multi=k!='single') for k in SETS]
                # All normal expectations are frozen before reading this query center.
                results=[ev.query_evidence(gray,expectation,box) for expectation in expectations]
                masks.append(np.stack([np.packbits(r['missing_mask'].reshape(-1),bitorder='little') for r in results]))
                flips.append(np.stack([r['flip_counts'] for r in results]));stats.append(diagnostics(frozen,results,matches,saturated))
                scores.append([r['score'] for r in results]);valids.append([r['evidence_present'] for r in results])
                agreements.append([r['agreement_fraction'] for r in results]);disagreements.append([r['disagreement_fraction'] for r in results]);refs.append([r['distinct_references'] for r in results])
                candidates=[h for k in SETS for h in frozen[k]['candidates']]
                ids=np.full((3,12),-1,np.int16);ts=np.zeros((3,12,2,3),np.float64)
                for k,key in enumerate(SETS):
                    for j,hypothesis in enumerate(frozen[key]['candidates']):ids[k,j]=hypothesis['ref'];ts[k,j]=hypothesis['T']
                candidate_refs.append(ids);transforms.append(ts)
                usage.append([any(h['kind']==kind for h in candidates) for kind in ('Source','Target')])
                if case and start+offset in (0,len(boxes)//2):visualize(a,row,gray,box,frozen,expectations[2],results[2],self.store.references,start+offset)
            array_name=f'chunk_{start:06d}.npz'
            atomic_npz(dest/array_name,boxes=np.asarray(batch,np.int32),scores=np.asarray(scores,np.float32),
                evidence_present=np.asarray(valids,bool),missing_mask_packed=np.asarray(masks,np.uint8),
                flip_counts=np.asarray(flips,np.uint8),candidate_references=np.asarray(candidate_refs),transforms=np.asarray(transforms),
                agreement_fraction=np.asarray(agreements,np.float32),disagreement_fraction=np.asarray(disagreements,np.float32),
                distinct_references=np.asarray(refs,np.int16))
            summary=dict(identity=ip.identity(a),index=index['seal'],boxes=[list(b) for b in batch],arrays=array_name,
                array_bytes=(dest/array_name).stat().st_size,seconds=time.perf_counter()-began,
                E=np.max(scores,axis=0).tolist(),valid_windows=np.sum(valids,axis=0).tolist(),
                agreement_sum=np.sum(agreements,axis=0).tolist(),disagreement_sum=np.sum(disagreements,axis=0).tolist(),
                distinct_reference_sum=np.sum(refs,axis=0).tolist(),reference_used_windows=np.sum(usage,axis=0).tolist(),
                counters={k:int(sum(r[k] for r in stats)) for k in stats[0]},
                mask_layout='window,branch,packed(24,64,64) little bit order; True=missing; original candidate transforms FP64')
            saved=st.sealed_json(path,summary);chunks.append(name);all_summaries.append(saved)
            print(f'[annular full image] {row["sample_id"]} {min(start+64,len(boxes))}/{len(boxes)} windows',flush=True)
        def sums(k):return np.sum([r[k] for r in all_summaries],axis=0).tolist() if all_summaries else [0,0,0]
        valid=sums('valid_windows');n=len(boxes)
        record=dict(identity=ip.identity(a),index=index['seal'],image=ip.image(row),sample_id=row['sample_id'],windows=n,
            E=dict(zip(E_NAMES,np.max([r['E'] for r in all_summaries],axis=0).tolist() if all_summaries else [0.,0.,0.])),
            valid_windows=valid,evidence_absent=[v==0 for v in valid],agreement_fraction=(np.asarray(sums('agreement_sum'))/max(1,n)).tolist(),
            disagreement_fraction=(np.asarray(sums('disagreement_sum'))/max(1,n)).tolist(),
            mean_distinct_references=(np.asarray(sums('distinct_reference_sum'))/max(1,n)).tolist(),
            reference_used_windows=sums('reference_used_windows'),
            counters={k:sum(r['counters'][k] for r in all_summaries) for k in all_summaries[0]['counters']} if all_summaries else {},
            seconds=sum(r['seconds'] for r in all_summaries)+gray_seconds,gray_decode_or_cache_seconds=gray_seconds,
            current_session_seconds=time.perf_counter()-tick,
            bytes=sum(r['array_bytes'] for r in all_summaries),chunks=chunks,GT_used=False,missing_is_normal=False)
        return st.sealed_json(result_path(a,row),record)

def engine(a):
    if not hasattr(a,'_engine'):a._engine=Engine(a)
    return a._engine
