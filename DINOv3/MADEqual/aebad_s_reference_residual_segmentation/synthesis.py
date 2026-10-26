"""RGB-only local changes; donor/foreground never read from test or calibration."""
import numpy as np
from scipy import ndimage
from . import storage as st
from .protocol import CONFIG,VERSION


def area_resize(mask,shape):
    """Exact area-overlap average, including when a thin region crosses bin edges."""
    from scipy.sparse import csr_matrix
    x=np.asarray(mask,np.float32)
    def weights(n,m):
        data=[];rows=[];cols=[]
        for j in range(m):
            lo=j*n/m;hi=(j+1)*n/m
            for i in range(int(np.floor(lo)),min(n,int(np.ceil(hi)))):
                value=max(0.,min(hi,i+1)-max(lo,i))/(hi-lo)
                if value:rows.append(j);cols.append(i);data.append(value)
        return csr_matrix((np.asarray(data,np.float32),(rows,cols)),shape=(m,n))
    return np.asarray((weights(x.shape[0],shape[0])@x)@weights(x.shape[1],shape[1]).T,dtype=np.float32)


def foreground(features,axis,mu):
    import cv2
    cv2.setNumThreads(1)
    f=np.asarray(features,np.float32);score=(f-mu)@axis
    h,w=score.shape;inner=score[h//4:max(h//4+1,3*h//4),w//4:max(w//4+1,3*w//4)]
    border=np.concatenate((score[0],score[-1],score[:,0],score[:,-1]))
    if float(inner.mean())<float(border.mean()):score=-score
    span=float(np.ptp(score))
    if not np.isfinite(score).all() or span<=1e-7:return np.zeros((h,w),bool),dict(status='invalid_constant_foreground')
    value=np.round((score-score.min())/span*255).astype(np.uint8)
    threshold,binary=cv2.threshold(value,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    labelled,n=ndimage.label(binary,np.ones((3,3),bool))
    if not n:return np.zeros((h,w),bool),dict(status='invalid_empty_foreground')
    sizes=np.bincount(labelled.ravel());sizes[0]=0;mask=labelled==int(sizes.argmax());fraction=float(mask.mean())
    okay=CONFIG['foreground']['minimum_area']<=fraction<=CONFIG['foreground']['maximum_area']
    return mask,dict(status='valid_auxiliary_not_GT' if okay else 'invalid_foreground_area',area_fraction=fraction,threshold=int(threshold))


def changed_label(before,after,requested):
    delta=np.abs(after.astype(np.float32)-before.astype(np.float32))/255
    effective=(delta.max(2)>=CONFIG['change_minimum'])&np.asarray(requested,bool)
    soft=(np.minimum(1.,delta.mean(2)/CONFIG['soft_change_scale'])*effective).astype(np.float32)
    return soft,delta.max(2).astype(np.float32)


def generate(before,donor,fg,origin,variant):
    """Finite retries; invalid slots remain explicit, never silently relabelled positive."""
    import cv2
    cv2.setNumThreads(1)
    before=np.asarray(before,np.uint8);donor=np.asarray(donor,np.uint8);fg=np.asarray(fg,bool)
    kind=CONFIG['synthesis_cycle'][variant%4];scale_name=('small','medium','large')[(variant+st.seed(origin)%3)%3]
    h,w=before.shape[:2];pixels=np.argwhere(fg)
    errors=[]
    if not len(pixels):return None,dict(status='invalid_no_foreground',kind=kind,scale=scale_name,attempts=0)
    full_fg=area_resize(fg.astype(np.float32),(h,w))>=.5
    for attempt in range(CONFIG['retry_limit']):
        rng=st.rng(VERSION,'RGB',CONFIG['synthesis_seed'],origin,variant,attempt)
        y,x=pixels[int(rng.integers(len(pixels)))];cy=int((y+.5)*h/fg.shape[0]);cx=int((x+.5)*w/fg.shape[1])
        side=max(8,int(min(h,w)*rng.uniform(*CONFIG['scales'][scale_name])))
        ph,pw=min(h-2,side),min(w-2,max(8,int(side*rng.uniform(.7,1.3))))
        y0=max(1,min(h-ph-1,cy-ph//2));x0=max(1,min(w-pw-1,cx-pw//2))
        requested=np.zeros((h,w),np.uint8);after=before.copy();params=dict(attempt=attempt,box=[y0,x0,ph,pw])
        try:
            if kind=='scratch':
                length=min(min(h,w)-2,max(8,side*2));angle=float(rng.uniform(0,np.pi));width=int(rng.integers(1,4))
                dx=int(np.cos(angle)*length/2);dy=int(np.sin(angle)*length/2)
                start=(int(np.clip(cx-dx,1,w-2)),int(np.clip(cy-dy,1,h-2)))
                end=(int(np.clip(cx+dx,1,w-2)),int(np.clip(cy+dy,1,h-2)))
                cv2.line(requested,start,end,1,width,lineType=cv2.LINE_8)
                color=rng.integers(0,256,size=3).astype(np.uint8);take=requested.astype(bool)
                after[take]=np.round(.15*before[take]+.85*color).astype(np.uint8)
                params.update(start=list(start),end=list(end),width=width,color=color.tolist())
            else:
                yy,xx=np.mgrid[:ph,:pw];ellipse=((yy-(ph-1)/2)/(ph/2))**2+((xx-(pw-1)/2)/(pw/2))**2<1
                noise=cv2.resize(rng.uniform(size=(4,4)).astype(np.float32),(pw,ph),interpolation=cv2.INTER_CUBIC)
                patchmask=(ellipse&(noise>.22)).astype(np.uint8)
                requested[y0:y0+ph,x0:x0+pw]=patchmask
                dh,dw=donor.shape[:2];dy=int(rng.integers(max(1,dh-min(ph,dh)+1)));dx=int(rng.integers(max(1,dw-min(pw,dw)+1)))
                content=cv2.resize(donor[dy:dy+min(ph,dh),dx:dx+min(pw,dw)],(pw,ph))
                params['donor_box']=[dy,dx,min(ph,dh),min(pw,dw)]
                if kind=='poisson':
                    after=cv2.seamlessClone(content,before,patchmask*255,(x0+pw//2,y0+ph//2),cv2.NORMAL_CLONE)
                else:
                    # Normal-content structure rearrangement plus program texture, no DTD.
                    yy0,xx0=np.mgrid[:ph,:pw];frequency=float(rng.uniform(.15,.5));phase=float(rng.uniform(0,2*np.pi))
                    texture=(127.5+100*np.sin(xx0*frequency+yy0*frequency*.6+phase))[:,:,None]
                    pattern=.5*content[::-1]+.5*texture
                    alpha=float(rng.uniform(*CONFIG['texture_alpha']));region=after[y0:y0+ph,x0:x0+pw]
                    blend=np.clip((1-alpha)*region+alpha*pattern,0,255).astype(np.uint8)
                    region[patchmask.astype(bool)]=blend[patchmask.astype(bool)];params.update(alpha=alpha,frequency=frequency)
            if not requested.any() or float(full_fg[requested.astype(bool)].mean())<CONFIG['foreground']['minimum_mask_overlap']:
                errors.append('insufficient_foreground_overlap');continue
            # OpenCV clones a mask bounding ROI; commit only the declared local
            # support so no shifted boundary pixel becomes an unlabelled change.
            after[requested==0]=before[requested==0]
            soft,change=changed_label(before,after,requested)
            if int((soft>0).sum())<CONFIG['minimum_changed_pixels']:
                errors.append('no_effective_RGB_change');continue
            return dict(rgb=after,requested=requested,soft=soft,change=change),dict(status='valid',kind=kind,scale=scale_name,
                attempts=attempt+1,parameters=params,changed_pixels=int((soft>0).sum()),soft_area=float(soft.sum()),errors=errors)
        except cv2.error as e:errors.append('opencv: '+str(e).splitlines()[0])
    return None,dict(status='invalid_retry_exhausted',kind=kind,scale=scale_name,attempts=CONFIG['retry_limit'],errors=errors)


def manifest(c):
    result=[]
    for row in [r for r in c['source'] if not r['calibration']]+sum(c['supports'].values(),[]):
        count=CONFIG['target_versions'] if row.get('support_index') is not None else CONFIG['source_versions']
        donors=[r for r in (c['supports'][row['domain']] if row.get('support_index') is not None else c['source'])
            if not r.get('calibration') and r['relative_path']!=row['relative_path']]
        if not donors:raise ValueError('no eligible normal donor')
        for variant in range(count):
            donor=sorted(donors,key=lambda r:st.seed(VERSION,'donor',row['relative_path'],variant,r['relative_path']))[0]
            result.append(dict(origin=row['relative_path'],variant=variant,donor=donor['relative_path'],
                item_id=st.digest([row['relative_path'],variant])[:28],kind=CONFIG['synthesis_cycle'][variant%4]))
    return result
