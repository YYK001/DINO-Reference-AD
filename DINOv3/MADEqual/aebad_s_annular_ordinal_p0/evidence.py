"""Freeze normal expectations first; the query center is read only afterwards."""
import numpy as np
from .protocol import OFFSETS
from . import pixels as px,geometry as geo

def normal_expectation(frozen,references,box,multi=True):
    yy,xx=np.mgrid[:64,:64];z=np.stack((xx,yy),-1);p=z[None]+OFFSETS[:,None,None,:]
    inside=((p>=0)&(p<64)).all(-1);global_z=z+np.array(box);global_p=p+np.array(box)
    candidates=frozen['candidates'];ids={references[h['ref']]['instance_id'] for h in candidates}
    if any(h['instance_id']!=references[h['ref']]['instance_id'] for h in candidates):raise ValueError('candidate original-image instance identity differs')
    shape=(24,64,64)
    if frozen.get('search_unresolved') or not candidates or (multi and len(ids)<2):
        return dict(expected=np.zeros(shape,bool),agreed=np.zeros(shape,bool),disagreement=np.zeros(shape,bool),
            refs=len(ids),status='search_unresolved' if frozen.get('search_unresolved') else 'insufficient_distinct_references')
    first=None;all_valid=inside.copy();same=np.ones(shape,bool)
    for h in candidates:
        gray=references[h['ref']]['read_gray']();T=np.asarray(h['T'])
        base,v0=px.sample(gray,geo.apply(T,global_z));neighbor,v1=px.sample(gray,geo.apply(T,global_p))
        bits=neighbor>base[None];all_valid &= v0[None]&v1
        if first is None:first=bits
        else:same &= bits==first
    agreed=all_valid&same;disagreement=all_valid&~same
    return dict(expected=first,agreed=agreed,disagreement=disagreement,refs=len(ids),status='normal_expectation_frozen')

def query_evidence(gray,normal,box):
    """No candidate generation/selection/fitting is reachable from this call."""
    x,y=box;center=np.asarray(gray[y:y+64,x:x+64]);yy,xx=np.mgrid[:64,:64]
    count=np.zeros((64,64),np.uint8)
    for k,(dx,dy) in enumerate(OFFSETS):
        nx=xx+dx;ny=yy+dy;inside=(nx>=0)&(nx<64)&(ny>=0)&(ny<64)
        values=center[np.clip(ny,0,63),np.clip(nx,0,63)]
        flipped=(values>center)!=normal['expected'][k]
        count+=(normal['agreed'][k]&inside&flipped).astype(np.uint8)
    e=count.astype(np.float32)/np.float32(24.)
    valid=normal['agreed'].any(0)
    blocks=sum(e[dy:dy+62,dx:dx+62] for dy in range(3) for dx in range(3))/np.float32(9.)
    valid_blocks=np.logical_and.reduce([valid[dy:dy+62,dx:dx+62] for dy in range(3) for dx in range(3)])
    score=float(blocks[valid_blocks].max()) if valid_blocks.any() else 0.
    return dict(score=score,valid_blocks=int(valid_blocks.sum()),flip_count=int(count.sum()),
        agreement_fraction=float(normal['agreed'].mean()),disagreement_fraction=float(normal['disagreement'].mean()),
        distinct_references=normal['refs'],evidence_present=bool(valid_blocks.any()),
        evidence=e,missing_mask=~normal['agreed'],flip_counts=count)
