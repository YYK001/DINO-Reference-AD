"""All methodological choices are frozen before any image metric is read."""
import numpy as np
from functools import lru_cache
import os

VERSION='annular_ordinal_v1'
BASE='SEGMENT_CH1';SOURCE='SOURCE_MULTI_FUSION';SINGLE='ST_SINGLE_FUSION';PRIMARY='ST_MULTI_FUSION'
E_NAMES=('E_SOURCE_MULTI','E_ST_SINGLE','E_ST_MULTI')
FUSIONS=(SOURCE,SINGLE,PRIMARY);BRANCHES=(BASE,*FUSIONS,*E_NAMES)
CONFIG=dict(version=VERSION,domain='background',support_seed=42,Source_images=455,Source_calibration=66,
    K=8,evaluation_images=289,normal_images=77,anomaly_images=212,primary=PRIMARY,
    window=64,stride=32,outer=128,guard=8,anchor_grid=8,anchors_per_direction=4,
    anchor_grid_origin='centers of 8px cells from visible annulus top-left (offset +4)',
    contrast='3x3 range; strictly positive; descriptor full 17x17 support must be annular',
    descriptor_bits=128,descriptor_radius=8,seed=42,rotations=[-15,0,15],scales=[.9,1.,1.1],
    normal_grid=8,Source_matches=16,Target_matches_per_image=4,Source_candidate_images=8,
    hypotheses=128,reprojection_pixels=2.,min_inliers=6,min_directions=3,
    singular_min=.75,singular_max=1.33,outer_validation_pairs=128,outer_flip_max=.25,
    candidate_cost='outer validation ordinal flip fraction',near_best=.05,corner_dedup=.5,
    max_explanations=12,leave_direction_corner_shift_query_pixels=1.,
    comparisons='eight integer compass directions times radii 1,2,4 (L-infinity steps)',
    denominator=24,readout='max valid 3x3 mean; each pixel has >=1 agreed comparison',
    sigma_E_floor=1e-6,sigma_E_fallback=1.,ddof=0,calibration='Source66 with Source455 plus original K8',
    gray='float32 (0.299 R + 0.587 G + 0.114 B)/255; pointwise',
    decoding='Pillow RGB, encoded raster orientation; no EXIF transpose, resize, smoothing or padding',
    interpolation='FP32 bilinear; out-of-bounds invalid, zero-weight neighbors not required',
    index=dict(backend='OpenCV FLANN LSH',approximate=True,algorithm=6,table_number=4,key_size=20,multi_probe_level=1,
        layout='source_sharded_int32_v1',source_shard_rows=(2**31-1)//16,seed_per_native_index=42),
    no_retrieval_completeness_claim=True,training=False,parameter_search=False,new_pixel_jobs=0,new_AUPRO_jobs=0,
    duration_budget_hours=12,smoke_calibration_images=4)

def comparison_pairs():
    r=np.random.default_rng(42);pairs=r.integers(-8,9,(128,2,2),dtype=np.int32)
    for i in range(128):
        while np.array_equal(pairs[i,0],pairs[i,1]):pairs[i,1]=r.integers(-8,9,2)
    return pairs

def outer_pairs():
    y,x=np.mgrid[-32:96,-32:96];p=np.c_[x.ravel(),y.ravel()]
    p=p[~((p[:,0]>=-8)&(p[:,0]<72)&(p[:,1]>=-8)&(p[:,1]<72))]
    r=np.random.default_rng(42);out=p[r.integers(0,len(p),(128,2))]
    for i in range(len(out)):
        while np.array_equal(out[i,0],out[i,1]):out[i,1]=p[r.integers(len(p))]
    return out.astype(np.int32)

PAIRS=comparison_pairs();OUTER_PAIRS=outer_pairs()
VARIANTS=tuple((angle,scale) for angle in CONFIG['rotations'] for scale in CONFIG['scales'])
OFFSETS=np.array([(dx*r,dy*r) for r in (1,2,4) for dx,dy in ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))],np.int32)

def version_matrix(version):
    angle,scale=VARIANTS[version];t=np.deg2rad(angle)
    M=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]],np.float32)*np.float32(scale)
    return M

def version_offsets(version):return PAIRS.astype(np.float32)@version_matrix(version).T

def fixed():return dict(CONFIG,descriptor_pairs=PAIRS.tolist(),outer_pixel_pairs=OUTER_PAIRS.tolist(),comparison_offsets=OFFSETS.tolist())

@lru_cache(maxsize=1)
def cuda_index_config():
    r=np.random.default_rng(42)
    return dict(backend='torch_binary_CSR_LSH_v1',approximate=True,table_number=4,key_size=20,multi_probe_level=1,
        seed=42,bit_positions=[r.choice(128,20,replace=False).tolist() for _ in range(4)],
        bucket_truncation=False,rerank='integer Hamming, physical dedup, deterministic global Top-k',
        OpenCV_candidate_equivalence=False)

def execution(a):
    backend=getattr(a,'backend','cpu')
    if backend=='cpu':return dict(backend='cpu',retrieval=CONFIG['index'],storage='dense_npz_v1')
    return dict(backend='cuda',device=a.device,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','not-set'),retrieval=cuda_index_config(),storage='sparse_lossless_v1',
        precision='FP32 pixels/sampling; integer descriptors/counts; batched FP64 small geometry',
        custom_Triton=False,parameters={name:getattr(a,name) for name in (
            'window_batch','index_shard_rows','candidate_tile','index_cache_gib','image_cache_gib',
            'geometry_groups','hypothesis_batch','writer_depth','prefetch_depth')})
