"""Origin mapping for augmented Official candidates, not context_reference.NoAug."""
import numpy as np
from .numerics import neighbors


def layout(catalog):
    rows=[];offset=0
    for r in catalog['source']:
        if r['calibration']:continue
        # Original AeBAD images exceed the host's minimum crop size.
        h,w=int(r['image_height']*.625)//16,int(r['image_width']*.625)//16
        if min(h,w)<40:raise ValueError('use recorded actual augmented grid for non-AeBAD minimum-size inputs')
        rows.append(dict(relative_path=r['relative_path'],source_index=r['source_index'],grid=[h,w],start=offset,stop=offset+h*w))
        offset+=h*w
    return rows


def map_indices(indices,rows):
    ids=np.asarray(indices)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids):raise ValueError('unique selected candidate indices required')
    if not rows or (ids<0).any() or (ids>=rows[-1]['stop']).any():raise ValueError('selected candidate outside ordered augmented grids')
    image=np.searchsorted([r['stop'] for r in rows],ids,side='right')
    local=ids-np.asarray([rows[i]['start'] for i in image],np.int64)
    return image,local


def gather_selected(features,indices,rows):
    """Read only selected centers/four neighbors while the original spool exists."""
    image,local=map_indices(indices,rows);dim=features.shape[1]
    delta=np.zeros((len(indices),4,dim),np.float32);valid=np.zeros(len(indices),bool)
    for number,row in enumerate(rows):
        take=np.flatnonzero(image==number)
        if not len(take):continue
        ns,ok=neighbors(row['grid'],local[take]);valid[take]=ok
        center=np.asarray(features[indices[take]],np.float32)
        delta[take]=np.asarray(features[row['start']+ns],np.float32)-center[:,None]
    delta[~valid]=0
    return delta,valid,image,local


def check_centers(actual,bank):
    """No nearest-vector recognition or tolerance relaxation for origin recovery."""
    actual,bank=np.asarray(actual,np.float32),np.asarray(bank,np.float32)
    return dict(verified=bool(np.array_equal(actual,bank)),
        max_abs_error=float(np.max(np.abs(actual-bank),initial=0)),atol=0.,rtol=0.)


class AugmentationRecorder:
    """Observe a cloned RNG. The actual preprocessing consumes its ONE usual draw."""
    def __init__(self,delegate):self.delegate=delegate;self.records=[]
    def __getattr__(self,k):return getattr(self.delegate,k)
    def __call__(self,value):
        clone=np.random.RandomState();clone.set_state(np.random.get_state())
        factor=float(clone.uniform(self.delegate.min_brightness_factor,self.delegate.max_brightness_factor))
        self.records.append(dict(brightness_factor=factor,augmentation='original_clip_brightness_then_bicubic_then_ImageNet'))
        return self.delegate(value)
