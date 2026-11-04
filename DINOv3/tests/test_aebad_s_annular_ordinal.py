"""Synthetic CPU logic and read-only real metadata, never a detection experiment."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import unittest
import tempfile
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0 import protocol as p,pixels as px,geometry as g,evidence as ev,inputs as ip,pipeline as pl,runtime as rt
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0.runner import parser,normalize
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0.index import Store
st=ip.st

def pattern():return np.random.default_rng(128).random((192,192),dtype=np.float32)
def reference(gray,i=0,identity=None,kind='Source'):
    return dict(index=i,instance_id=identity or f'original-{i}',kind=kind,shape=gray.shape,read_gray=lambda:gray)
def candidate(i=0,identity=None,T=None,cost=0.):
    return dict(ref=i,instance_id=identity or f'original-{i}',kind='Source',T=(np.array([[1.,0,0],[0,1.,0]]) if T is None else T).tolist(),outer_cost=cost,inliers=16)
def frozen(*candidates,unresolved=False):return dict(candidates=list(candidates),search_unresolved=unresolved)

class TestAnnularOrdinal(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.tmp_path=Path(self.tmp.name)

    def test_default_fixed_population_and_no_training(self):
        assert parser().parse_args([]).stage=='preflight'
        assert (p.CONFIG['Source_images'],p.CONFIG['Source_calibration'],p.CONFIG['evaluation_images'],p.CONFIG['K'])==(455,66,289,8)
        assert p.CONFIG['training'] is False and p.CONFIG['new_AUPRO_jobs']==0
        assert p.PAIRS.shape==(128,2,2) and p.OFFSETS.shape==(24,2)
        assert np.all(np.any(p.PAIRS[:,0]!=p.PAIRS[:,1],axis=1))

    def test_windows_edge_dedup_and_real_visible_boundaries(self):
        assert px.windows((63,100))==[]
        assert px.windows((64,64))==[(0,0)]
        boxes=px.windows((101,101));assert len(boxes)==len(set(boxes))==9 and boxes[-1]==(37,37)
        gray=pattern();info=g.prepare_outer(gray,(0,0))
        assert len(info['points'])<=16
        assert px.annular(info['validation_pairs'],(0,0),gray.shape).all()
        with self.assertRaises(ValueError):px.ring_sample(gray,np.array([[30,30]]),(0,0))
        values,valid=px.sample(gray,np.array([[-.1,0],[191,191],[192,0]],np.float32))
        assert valid.tolist()==[False,True,False]

    def test_dense_descriptors_equal_same_fp32_bilinear_samples(self):
        gray=pattern()
        for version in range(9):
            spec=px.grid_spec(gray.shape,version);dense=px.describe_grid(gray,spec)
            ids=np.linspace(0,len(dense)-1,8,dtype=int)
            points=np.c_[spec['x0']+(ids%spec['nx'])*8,spec['y0']+(ids//spec['nx'])*8].astype(np.float32)
            sampled,valid=px.sample(gray,points[:,None,None]+p.version_offsets(version)[None])
            assert valid.all()
            expected=np.packbits(sampled[:,:,1]>sampled[:,:,0],axis=1,bitorder='little')
            np.testing.assert_array_equal(dense[ids],expected)

    def test_query_center_isolation_including_fitting_rng_and_normal_masks(self):
        gray=pattern();changed=gray.copy();box=(64,64);changed[64:128,64:128]=1-gray[64:128,64:128]
        a=g.prepare_outer(gray,box);b=g.prepare_outer(changed,box)
        for key in ('points','bits','directions','contrast','validation_pairs','validation_bits'):np.testing.assert_array_equal(a[key],b[key])
        refs=[reference(gray),reference(gray.copy(),1)]
        matches=[dict(anchor=j,ref=i,point=point.tolist(),version=4,hamming=0) for i in (0,1) for j,point in enumerate(a['points'])]
        first=g.freeze(a,matches,refs,box);second=g.freeze(b,matches,refs,box)
        assert first==second and len(first['multi']['candidates'])==2
        n1=ev.normal_expectation(first['multi'],refs,box);n2=ev.normal_expectation(second['multi'],refs,box)
        np.testing.assert_array_equal(n1['agreed'],n2['agreed']);np.testing.assert_array_equal(n1['expected'],n2['expected'])
        assert ev.query_evidence(gray,n1,box)['score']==0
        assert ev.query_evidence(changed,n2,box)['score']>0

    def test_known_integer_translation_and_direction_stability(self):
        gray=pattern();box=(64,64);info=g.prepare_outer(gray,box);q=info['points'].astype(float)
        T=np.array([[1.,0,3.],[0,1.,5.]]);r=g.apply(T,q)
        assert g.admissible(T)
        assert g.leave_direction_stability(T,q,r,info['directions'],box)<1e-10
        np.testing.assert_allclose(g.apply(T,np.array([[66,69],[67,69]])),[[69,74],[70,74]],rtol=0,atol=0)
        singular=np.array([[1.,0,0],[0,0,0]])
        assert not g.admissible(singular)
        assert g.leave_direction_stability(T,q[:3],r[:3],np.zeros(3),box) is None
        # Measured corner changes are converted back to query units, including scale.
        scale=np.array([[1.2,0,3],[0,1.1,5]])
        assert g.leave_direction_stability(scale,q,g.apply(scale,q),info['directions'],box)<1e-10

    def test_known_affine_recovered_from_outer_correspondences(self):
        reference_gray=np.random.default_rng(7).random((320,320),dtype=np.float32)
        angle=np.deg2rad(10);M=1.02*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        T=np.c_[M,[40.,40.]];yy,xx=np.mgrid[:192,:192]
        query,valid=px.sample(reference_gray,g.apply(T,np.stack((xx,yy),-1)))
        assert valid.all();box=(64,64);info=g.prepare_outer(query,box)
        ref=reference(reference_gray)
        matches=[dict(anchor=j,ref=0,point=point.tolist(),version=4,hamming=0) for j,point in enumerate(g.apply(T,info['points']))]
        pool=g.freeze(info,matches,[ref],box)
        assert pool['single']['candidates']
        np.testing.assert_allclose(pool['single']['candidates'][0]['T'],T,rtol=0,atol=1e-10)
        assert pool['single']['candidates'][0]['outer_cost']==0

    def test_strict_monotone_integer_bits_and_bilinear_limitation(self):
        gray=pattern();info=g.prepare_outer(gray,(64,64));transformed=g.prepare_outer(gray**3,(64,64))
        # Anchor contrast ranking may change; fixed positions are what is invariant.
        values=px.ring_sample(gray,info['points'][:,None,None]+p.PAIRS[None],(64,64))
        transformed_values=px.ring_sample(gray**3,info['points'][:,None,None]+p.PAIRS[None],(64,64))
        np.testing.assert_array_equal(values[:,:,1]>values[:,:,0],transformed_values[:,:,1]>transformed_values[:,:,0])
        a=np.array([[0.,1.],[.6,.6]],np.float32);points=np.array([[.5,0],[.5,1]])
        v,_=px.sample(a,points);u,_=px.sample(a**2,points)
        assert (v[1]>v[0])!=(u[1]>u[0])

    def test_two_normal_center_shapes_produce_disagreement_without_selection(self):
        gray=pattern();different=gray.copy();different[64:128,64:128]=1-different[64:128,64:128]
        refs=[reference(gray),reference(different,1)];pool=frozen(candidate(),candidate(1))
        expected=ev.normal_expectation(pool,refs,(64,64))
        assert expected['disagreement'].any() and not expected['agreed'][:,4:-4,4:-4].any()
        old=copy.deepcopy(pool);ev.query_evidence(gray,expected,(64,64));ev.query_evidence(different,expected,(64,64));assert pool==old

    def test_duplicate_originals_do_not_count_as_two_instances(self):
        gray=pattern();refs=[reference(gray,identity='same'),reference(gray,1,identity='same')]
        n=ev.normal_expectation(frozen(candidate(identity='same'),candidate(1,identity='same')),refs,(64,64))
        assert n['refs']==1 and not n['agreed'].any()
        with self.assertRaises(ValueError):ev.normal_expectation(frozen(candidate(),candidate(1)),refs,(64,64))

    def test_fixed_denominator_no_evidence_and_budget_overflow(self):
        gray=pattern();empty=ev.normal_expectation(frozen(),[],(64,64));r=ev.query_evidence(gray,empty,(64,64))
        assert r['score']==0 and not r['evidence_present'] and r['missing_mask'].all()
        ref=reference(gray);n=ev.normal_expectation(frozen(candidate()),[ref],(64,64),multi=False)
        n['agreed'][1:]=False
        changed=gray.copy();changed[64:128,64:128]=1-gray[64:128,64:128]
        assert ev.query_evidence(changed,n,(64,64))['score']<=1/24+1e-8
        candidates=[candidate(i%8,identity=str(i%8),T=np.array([[1.,0,4*(i//8)],[0,1.,0]])) for i in range(13)]
        for h in candidates:h['kind']='Target'
        info=g.prepare_outer(gray,(64,64));refs=[reference(gray,i,identity=str(i),kind='Target') for i in range(8)]
        matches=[dict(anchor=0,ref=i,point=[10,10],version=4,hamming=0) for i in range(8)]
        # Mock only the geometric fit to isolate the competition budget rule.
        with patch.object(g,'fit_image',side_effect=lambda info,m,ref,box,stats:[h for h in candidates if h['ref']==ref['index']]):
            # Eight selected images can each contain spatially distinct explanations.
            pool=g.freeze(info,matches,refs,(64,64))
        assert pool['multi']['search_unresolved'] and pool['multi']['candidates']==[]
        assert len(pool['single']['candidates'])==1

    def test_shared_calibration_scale_and_degenerate_P(self):
        P=np.arange(66,dtype=float);E=np.zeros((66,3));model=pl.fit_calibration(P,E)
        assert model['E_scale_degenerate'] and model['sigma_E']==1
        values=pl.scores(7.,[0,0,0],model)
        assert values[p.BASE]==7. and len({values[k] for k in p.FUSIONS})==1
        E[:,2]=np.linspace(0,1,66);model=pl.fit_calibration(P,E);values=pl.scores(7.,[.1,.2,.3],model)
        assert abs((values[p.PRIMARY]-values[p.SINGLE])-.1/model['sigma_E'])<1e-12
        with self.assertRaises(ValueError):pl.fit_calibration(np.ones(66),E)

    def test_native_lsh_save_load_hamming_and_version_cap(self):
        tmp_path=self.tmp_path
        import cv2 as cv
        cv.setRNGSeed(42)
        values=np.random.default_rng(42).integers(0,256,(128,16),np.uint8)
        index=cv.flann_Index(values,{k:p.CONFIG['index'][k] for k in ('algorithm','table_number','key_size','multi_probe_level')})
        path=tmp_path/'test.lsh';index.save(str(path));native=cv.flann_Index();assert native.load(values,str(path))
        ids,distance=native.knnSearch(values[:4],4,params={})
        assert ids[:,0].tolist()==[0,1,2,3] and (distance[:,0]==0).all()
        changed=values[:1].copy();changed[0,0]^=np.uint8(3)
        ids,distance=native.knnSearch(changed,4,params={})
        assert ids[0,0]==0 and distance[0,0]==2  # Hamming, not scalar/L2 distance.
        store=Store.__new__(Store);store.query_cache=__import__('collections').OrderedDict();store.rows=[dict(instance_id='one')]
        # 9 versions of two physical positions; return budget is applied after merging.
        class Native:
            def knnSearch(self,q,k,params):return np.tile(np.arange(k),(len(q),1)),np.zeros((len(q),k),np.int32)
        store.backend=[dict(name='Target0',rows=18,native=Native(),blocks=[dict(start=2*i,end=2*i+2,ref=0,x0=8,y0=8,nx=2,version=i) for i in range(9)])]
        store.load=lambda:None
        hits,flag=store.lookup_many(values[:1])[0]
        assert len(hits)==2 and len({(h[2],h[3]) for h in hits})==2

    def test_real_metadata_pool_P_and_baseline_identity(self):
        tmp_path=self.tmp_path
        a=normalize(parser().parse_args(['--run-root',str(tmp_path/'run'),'--cache-root',str(tmp_path/'cache')]))
        x=ip.assets(a)
        assert len(x.source)==455 and len(x.calibration)==66 and len(x.evaluation)==289 and len(x.calibration_refs)==66 and not x.missing
        assert not ({r['image_sha256'] for r in x.calibration}&{r['instance_id'] for r in x.normal})
        assert len(x.P)==355 and [r['image_sha256'] for r in x.normal[455:]]==[r['image_sha256'] for r in x.target]
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
        metric=image_metrics([r['label'] for r in x.evaluation],[x.P[r['relative_path']] for r in x.evaluation])
        parent=next(r for r in x.metrics if r['branch']=='SEGMENT_K8' and r['scoring']=='CH1')
        for k,value in metric.items():assert value==float(parent[k])

    def test_preflight_does_not_import_torch_or_decode_images(self):
        tmp_path=self.tmp_path
        code="""
    from unittest.mock import patch
    from DINOv3.MADEqual.aebad_s_annular_ordinal_p0.runner import main
    from DINOv3.MADEqual.aebad_s_annular_ordinal_p0.inputs import Data
    import sys
    with patch.object(Data,'image',side_effect=AssertionError('real decoding forbidden')):
        assert main(['preflight','--run-root',sys.argv[1],'--cache-root',sys.argv[2]])==0
    assert 'torch' not in sys.modules
    """
        import textwrap
        result=subprocess.run([sys.executable,'-B','-c',textwrap.dedent(code),str(tmp_path/'run'),str(tmp_path/'cache')],cwd=ip.ROOT,capture_output=True,text=True)
        assert result.returncode==0,result.stderr

    def test_completed_image_resume_skips_processing_and_keeps_absence(self):
        tmp_path=self.tmp_path
        row=dict(relative_path='x.png',image_sha256='a',image_height=64,image_width=64,sample_id='x')
        a=SimpleNamespace(cache_root=tmp_path/'cache',run_root=tmp_path/'run',_identity='id')
        record=st.sealed_json(rt.result_path(a,row),dict(identity='id',image=ip.image(row),chunks=[],E=dict.fromkeys(p.E_NAMES,0.),evidence_absent=[True]*3))
        engine=rt.Engine.__new__(rt.Engine);engine.a=a;engine.store=SimpleNamespace(load=lambda:self.fail('completed image loaded native index'))
        assert engine.process(row)==record

    def test_output_isolation_and_smoke_gate(self):
        tmp_path=self.tmp_path
        args=parser().parse_args(['--run-root',str(tmp_path/'x'),'--cache-root',str(tmp_path/'x/cache')])
        with self.assertRaises(ValueError):normalize(args)
        a=SimpleNamespace(run_root=tmp_path/'run',_identity='id',allow_over_12h=False)
        st.sealed_json(a.run_root/'smoke.json',dict(identity='id',valid_multi_evidence=False))
        with self.assertRaisesRegex(RuntimeError,'no effective multi'):pl.guard(a)

    def cached_synthetic_image(self,folder='image'):
        tmp_path=self.tmp_path/folder
        row=dict(relative_path='synthetic.png',image_sha256='synthetic',image_height=64,image_width=64,sample_id='synthetic')
        a=SimpleNamespace(cache_root=tmp_path/'cache',run_root=tmp_path/'run',_identity='id')
        store=SimpleNamespace(load=lambda:dict(seal='index'),gray=lambda r:np.zeros((64,64),np.float32),references=[],
            matches=lambda info,lookups:([],False))
        engine=rt.Engine(a,store);result=engine.process(row)
        return a,row,store,engine,result

    def test_full_small_image_keeps_missing_masks_and_resumes_without_reads(self):
        a,row,store,engine,result=self.cached_synthetic_image()
        assert result['windows']==1 and result['evidence_absent']==[True]*3 and result['valid_windows']==[0]*3
        with np.load(rt.result_path(a,row).parent/'chunk_000000.npz',allow_pickle=False) as arrays:
            assert arrays['missing_mask_packed'].shape==(1,3,12288)
            assert (arrays['missing_mask_packed']==255).all() and not arrays['flip_counts'].any()
            assert (arrays['candidate_references']==-1).all()
        store.gray=lambda r:self.fail('completed synthetic image decoded again')
        assert engine.process(row)==result

    def test_truncated_chunks_rejected_for_complete_and_partial_resume(self):
        for complete in (True,False):
            with self.subTest(complete=complete):
                a,row,store,engine,result=self.cached_synthetic_image(str(complete))
                final=rt.result_path(a,row);arrays=final.parent/'chunk_000000.npz'
                # Prime complete-image and NPZ validation memos before corruption.
                assert engine.process(row)==result
                if not complete:final.unlink()
                arrays.write_bytes(b'')
                with self.assertRaisesRegex(ValueError,'committed window NPZ invalid'):
                    engine.process(row)
                if not complete:assert not final.exists()
                else:
                    fresh=SimpleNamespace(cache_root=a.cache_root,run_root=a.run_root,_identity='id')
                    with self.assertRaisesRegex(ValueError,'committed window NPZ invalid'):rt.completed(fresh,row)

    def test_npz_readability_and_required_layout_without_hash_scans(self):
        a,row,store,engine,result=self.cached_synthetic_image()
        directory=rt.result_path(a,row).parent;path=directory/'chunk_000000.npz'
        record=st.sealed_json(directory/'chunk_000000.json')
        with np.load(path,allow_pickle=False) as original:valid={key:original[key] for key in original.files}
        cases=dict(missing=lambda arrays:arrays.pop('transforms'),
            shape=lambda arrays:arrays.update(flip_counts=arrays['flip_counts'][:,:,:,:63]),
            dtype=lambda arrays:arrays.update(candidate_references=arrays['candidate_references'].astype(np.float32)))
        for name,mutate in cases.items():
            with self.subTest(case=name):
                arrays=dict(valid);mutate(arrays);rt.atomic_npz(path,**arrays)
                matching_size=dict(record,array_bytes=path.stat().st_size)
                with self.assertRaisesRegex(ValueError,'committed window NPZ invalid'):
                    rt.validate_chunk(a,directory,matching_size)
        # ZIP remains parseable and the byte count matches, but a compressed member is damaged.
        import zipfile
        rt.atomic_npz(path,**valid)
        size=path.stat().st_size
        with patch.object(st,'sha',side_effect=AssertionError('no full-file SHA scan')):
            rt.validate_chunk(a,directory,dict(record,array_bytes=size))
        payload=bytearray(path.read_bytes())
        with zipfile.ZipFile(path) as archive:member=archive.getinfo('flip_counts.npy')
        offset=member.header_offset
        start=offset+30+int.from_bytes(payload[offset+26:offset+28],'little')+int.from_bytes(payload[offset+28:offset+30],'little')
        payload[start+member.compress_size//2]^=1;path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError,'committed window NPZ invalid'):
            rt.validate_chunk(a,directory,dict(record,array_bytes=size))
        # Matching the recorded size alone is insufficient even for a ZIP-looking file.
        rt.atomic_npz(path,**valid);size=path.stat().st_size;path.write_bytes(b'PK'+bytes(size-2))
        with self.assertRaisesRegex(ValueError,'committed window NPZ invalid'):
            rt.validate_chunk(a,directory,dict(record,array_bytes=size))

    def test_unchanged_npz_validated_once_per_process(self):
        a,row,store,engine,result=self.cached_synthetic_image()
        with patch.object(rt.np,'load',wraps=rt.np.load) as load:
            assert engine.process(row)==result
            assert load.call_count==1
            assert engine.process(row)==result
            assert load.call_count==1

    def test_normal_smoke_requires_coverage_not_a_positive_anomaly_score(self):
        a=SimpleNamespace(run_root=self.tmp_path/'run',_identity='id',allow_over_12h=False)
        st.sealed_json(a.run_root/'smoke.json',dict(identity='id',valid_multi_evidence=True,images=[dict(E=dict.fromkeys(p.E_NAMES,0.))]*4))
        with patch.object(pl,'estimate',return_value=dict(projected_pixel_total_seconds=500.)):
            assert pl.guard(a)['projected_pixel_total_seconds']==500.
        with patch.object(pl,'estimate',return_value=dict(projected_pixel_total_seconds=43201.)):
            with self.assertRaisesRegex(RuntimeError,'exceeds 12h'):pl.guard(a)
            saved=dict(status='normal_smoke_complete',valid_multi_evidence=True)
            assert pl.smoke_status(a,saved)['status']=='pending' and saved['status']=='normal_smoke_complete'
        with patch.object(pl,'estimate',return_value=dict(projected_pixel_total_seconds=None)):
            with self.assertRaisesRegex(RuntimeError,'unknown'):pl.guard(a)

if __name__=="__main__":unittest.main()
