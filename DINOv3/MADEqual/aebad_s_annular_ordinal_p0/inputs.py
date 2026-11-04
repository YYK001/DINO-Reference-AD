"""Scoped read-only adapters to the original split, CH1 statistics and maps."""
import sys
import importlib.util
from functools import lru_cache
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .protocol import CONFIG,fixed
from . import pixels as px
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import storage as st
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.inputs import validate_source
from DINOv3.MADEqual.aebad_s_conditional_image_score_p0 import inputs as conditional
from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import reporting as parent_reporting
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import Data
ROOT=Path(__file__).resolve().parents[3]
MAP='UNION_SOURCE_WEIGHT'

def image(row):return {k:row[k] for k in ('relative_path','image_sha256','image_height','image_width')}
def key(row):return st.digest(image(row))[:24]
def package(name):
    try:return version(name)
    except PackageNotFoundError:return None
@lru_cache(maxsize=1)
def pixel_contract():return dict(gray=CONFIG['gray'],decoding=CONFIG['decoding'],Pillow=package('Pillow'),numpy=package('numpy'),
    implementation=st.sha(Path(px.__file__)))

class Assets:
    def __init__(self,a):
        self.a=a;self.parent=st.sealed_json(a.parent_run/'inputs.json');self.common=st.sealed_json(a.parent_run/'common_manifest.json')
        if self.parent['common_manifest']!=self.common['seal'] or self.parent['Source_seed']!=42:raise ValueError('frozen two-K8 Source42/common manifest required')
        if self.parent['identity']!=st.digest({k:v for k,v in self.parent.items() if k not in ('identity','seal')}):raise ValueError('parent identity differs')
        group=self.common['domains']['background'];self.evaluation=group['common'];self.target=group['supports']['42']
        excluded={r['relative_path'] for rows in group['supports'].values() for r in rows}
        if (len(self.evaluation),sum(r['label']==0 for r in self.evaluation),sum(r['label']==1 for r in self.evaluation))!=(289,77,212):raise ValueError('289 common images, 77 normal/212 anomaly required')
        if len(self.target)!=8 or excluded&{r['relative_path'] for r in self.evaluation}:raise ValueError('original K8/order and support-union exclusion required')
        self.raw=st.sealed_json(a.raw_run/'catalog.json');rows=self.raw['catalog']['source'];validate_source(dict(source=rows))
        self.source=[r for r in rows if not r['calibration']];self.calibration=[r for r in rows if r['calibration']]
        if len(self.source)!=455 or len(self.calibration)!=66 or [image(r) for r in self.calibration]!=[image(r) for r in self.parent['Source_calibration']]:raise ValueError('original Source455/66 split differs')
        if {r['relative_path'] for r in self.source}&{r['relative_path'] for r in self.calibration}:raise ValueError('calibration image in pixel reference pool')
        self.metrics,_=parent_reporting.read_complete(SimpleNamespace(run_root=a.parent_run,domain='background',support_seed=42))
        predictions=st.sealed_json(a.parent_run/'background/seed42/predictions_frozen.json')
        if len(predictions['records'])!=289:raise ValueError('complete parent prediction population required')
        self.P={};self.map_refs={};self.score_refs=[]
        for row,ref in zip(self.evaluation,predictions['records']):
            r=st.sealed_json(a.parent_run/'background/seed42/predictions'/(row['sample_id']+'.json'))
            if r['seal']!=ref['seal'] or r['image']!=row:raise ValueError('parent SEGMENT image/score record differs')
            value=float(r['scores']['SEGMENT_K8']['CH1'])
            if not np.isfinite(value):raise ValueError('nonfinite parent CH1')
            self.P[row['relative_path']]=value;self.map_refs[row['relative_path']]=r['map_hashes'][MAP]
            self.score_refs.append(dict(sample_id=row['sample_id'],seal=r['seal']))
        # The conditional reader's identity uses six tiny weight commits; no
        # other-domain maps, features or prediction populations are opened.
        weights={f'{d}/{s}':st.sealed_json(a.parent_run/d/f'seed{s}'/'weights.json')['seal'] for d in ('background','illumination','view') for s in (42,43)}
        parent_id=st.digest(dict(parent_inputs=self.parent['seal'],common=self.common['seal'],weights=weights))
        self.bridge=SimpleNamespace(run_root=a.conditional_run,statistics_provider=None,
            _joint_parent=SimpleNamespace(inputs=self.parent,common=self.common,identity=parent_id))
        self.missing=[];self.calibration_refs=[]
        try:
            ci=st.sealed_json(a.conditional_run/'inputs.json')
            if ci['parent_identity']!=parent_id or ci['common_manifest']!=self.common['seal']:raise ValueError('conditional P provider belongs to another parent')
            for row in self.calibration:
                c=conditional.contract(self.bridge,'background',42,'Source',row)
                r=conditional.bp(self.bridge,c)
                if r is None:self.missing.append(dict(role='Source_plus_K8_SEGMENT_CH1',path=str(conditional.record_path(self.bridge,c)),image=row['relative_path'],action='restore compatible P; no automatic memory rebuild'))
                else:self.P[row['relative_path']]=float(r['P']);self.calibration_refs.append(dict(image=image(row),seal=r['seal']))
        except FileNotFoundError as e:self.missing.append(dict(role='conditional_Source66_K8_statistics',path=str(e),action='restore small original records; Source-only scores incompatible'))
        self.normal=[dict(image(r),kind='Source',index=i,instance_id=r['image_sha256']) for i,r in enumerate(self.source)]
        self.normal += [dict(image(r),kind='Target',index=455+i,instance_id=r['image_sha256']) for i,r in enumerate(self.target)]

    def rows(self,kind):
        rows=self.calibration if kind=='calibration' else self.evaluation
        return [dict(image(r),sample_id=r['sample_id'] if 'sample_id' in r else f'source_{r["source_index"]:04d}') for r in rows]

    def provenance(self):
        return dict(parent_inputs=self.parent['seal'],common_manifest=self.common['seal'],raw_catalog=self.raw['seal'],
            source_pixel_images=[image(r) for r in self.source],Source_calibration=[image(r) for r in self.calibration],
            supports=[image(r) for r in self.target],evaluation_images=self.rows('evaluation'),
            parent_score_records=self.score_refs,calibration_P_records=self.calibration_refs,
            localization_reference=MAP,localization_identities=self.map_refs,
            old_Source_bank_reference=self.parent['source'],pixel_Source_is_original_455_not_augmented_bank=True)

def assets(a):
    if not hasattr(a,'_assets'):a._assets=Assets(a)
    return a._assets

def freeze(a,commit=True):
    x=assets(a);value=dict(config=fixed(),assets=x.provenance(),pixel_contract=pixel_contract(),
        data=str(a.data),code={p.name:st.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    value['identity']=st.digest(value)
    st.sealed_json(a.run_root/'fixed_config.json',fixed())
    return st.sealed_json(a.run_root/'inputs.json',value) if commit else dict(value,seal=st.digest(value))

def identity(a):
    if not hasattr(a,'_identity'):a._identity=st.sealed_json(a.run_root/'inputs.json')['identity']
    return a._identity

def preflight(a):
    available=[];missing=[];new=[]
    try:
        x=assets(a);snapshot=freeze(a,False);st.write_json(a.run_root/'preflight_inputs.json',snapshot)
        available.append(dict(role='common_split_CH1_P_localization_refs',path=str(a.parent_run),evaluation=289,calibration_P=len(x.calibration_refs),Source_pixel_images=455,Target_images=8))
        missing.extend(x.missing)
        try:
            with Data(a.data) as data:
                for row in [*x.source,*x.calibration,*x.target,*x.evaluation]:
                    rel=row['relative_path'];present=rel in data.names if data.zip else (data.root/rel).is_file()
                    (available if present else missing).append(dict(role='original_RGB',image=rel,action='decode once when consumed' if present else 'restore original image'))
        except (OSError,ValueError) as e:missing.append(dict(role='original_RGB_dataset',path=str(a.data),reason=str(e)))
        backend=importlib.util.find_spec('cv2') is not None
        if not backend:missing.append(dict(role='OpenCV_FLANN_LSH',action='locate available environment; no automatic installation'))
        source_rows=sum(sum(px.grid_spec((r['image_height'],r['image_width']),v)['count'] for v in range(9)) for r in x.source)
        target_rows=sum(sum(px.grid_spec((r['image_height'],r['image_width']),v)['count'] for v in range(9)) for r in x.target)
        window_counts={kind:sum(len(px.windows((r['image_height'],r['image_width']))) for r in rows) for kind,rows in (('calibration',x.calibration),('evaluation',x.evaluation))}
        gray_bytes=sum(r['image_height']*r['image_width']*4 for r in x.normal)
        new=[dict(role='original_gray_and_binary_index',images=463,Source_descriptors=source_rows,Target_descriptors=target_rows,
            binary_bytes=(source_rows+target_rows)*16,gray_cache_bytes=gray_bytes,global_Source_copy_bytes=source_rows*16,
            calibration_and_evaluation_gray_cache_bytes=sum(r['image_height']*r['image_width']*4 for r in x.calibration+x.evaluation),
            native_LSH_RAM_and_saved_index_bytes='unmeasured; native data copies and hash-table entries additional'),
            dict(role='full_image_annular_geometry_center_evidence',images=355,windows=window_counts,performance='unmeasured',
                missing_masks_uncompressed_bytes=sum(window_counts.values())*3*24*64*64//8,
                flip_counts_uncompressed_bytes=sum(window_counts.values())*3*64*64,
                candidate_transforms_uncompressed_bytes=sum(window_counts.values())*3*12*6*8,
                cache='compressed 64-window NPZ chunks; actual compression and native-index size must be measured'),
            dict(role='image_evaluation',branches=7,metrics=2,new_pixel_jobs=0,new_AUPRO_jobs=0)]
        result=dict(status='missing_inputs' if missing else 'inputs_available',missing=len(missing),new_work=new,
            torch_imported='torch' in sys.modules,GPU_initialized=False,model_assets_required=False,
            time_estimate='unknown until index build and 4 full normal calibration images; no hours guarantee',pixel_audit='existing encoded PNG identity checked once on first decode; cached grayscale reused, no repeated array SHA sweep')
    except (OSError,ValueError,KeyError) as e:
        missing.append(dict(role='metadata',reason=f'{type(e).__name__}: {e}'));result=dict(status='blocked',reason=str(e),torch_imported='torch' in sys.modules)
    st.write_csv(a.run_root/'asset_inventory.csv',available);st.write_csv(a.run_root/'missing_inputs.csv',missing);st.write_json(a.run_root/'new_work.json',new)
    st.write_json(a.run_root/'preflight.json',result);return result
