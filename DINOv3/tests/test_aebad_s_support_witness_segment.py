"""Synthetic production CPU paths only. No real support search or model/GPU."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import method as m,neighbors,storage as st,inputs as ip,supports,prediction,reporting
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as ev

ROOT=Path(__file__).resolve().parents[2]
SCRATCH=ROOT/'DINOv3/MADEqual/aebad_s_support_witness_segment_p0/local_validation'


def geometry_fixture(third=(1.,0.)):
    banks={1:np.array([[0.,0.]]),2:np.array([[2.,0.]]),3:np.array([third])}
    nb={(a,c):np.zeros((len(v),1),np.int64) for a,v in banks.items() for c in banks if c!=a}
    return banks,nb


class WitnessGeometry(unittest.TestCase):
    def test_internal_point(self):
        b,n=geometry_fixture();v=m.witness(b,n,(1,0,2,0))
        self.assertTrue(v['accepted']);self.assertEqual(v['sources'],[3]);p=v['third_images'][0]['passing_points'][0]
        self.assertEqual(p['alpha'],.5);self.assertEqual(p['rho'],0.)

    def test_endpoint_near_points_rejected(self):
        for point in ((.1,0),(1.9,0),(0,0),(2,0)):
            b,n=geometry_fixture(point);self.assertFalse(m.witness(b,n,(1,0,2,0))['accepted'])

    def test_off_line_rejected(self):
        b,n=geometry_fixture((1.,3.));v=m.witness(b,n,(1,0,2,0));self.assertFalse(v['accepted'])
        self.assertEqual(v['third_images'][0]['counts'],{'too_far_from_segment':1})

    def test_alpha_boundaries_inclusive(self):
        for x in (.5,1.5):
            b,n=geometry_fixture((x,0));self.assertTrue(m.witness(b,n,(1,0,2,0))['accepted'])

    def test_rho_boundary_and_independent_formula(self):
        y=1/np.sqrt(3);b,n=geometry_fixture((1,y));v=m.witness(b,n,(1,0,2,0));p=v['third_images'][0]['representative']
        self.assertAlmostEqual(p['rho'],.5);self.assertEqual(v['accepted'],p['rho']<=.5)
        w=b[3][0];u=b[1][0];d=b[2][0]-u;alpha=np.dot(w-u,d)/np.dot(d,d)
        self.assertEqual(p['alpha'],alpha);self.assertEqual(p['rho'],np.linalg.norm(w-u-alpha*d)/min(np.linalg.norm(w-u),np.linalg.norm(w-b[2][0])))

    def test_endpoint_image_cannot_witness(self):
        b,n=geometry_fixture((0.,0.));b[1]=np.array([[0.,0.],[1.,0.]])
        v=m.witness(b,n,(1,0,2,0));self.assertFalse(v['accepted']);self.assertEqual([t['support_index'] for t in v['third_images']],[3])

    def test_whole_heldout_image_cannot_witness(self):
        b,n=geometry_fixture();v=m.witness(b,n,(1,0,2,0),excluded=(3,));self.assertFalse(v['accepted']);self.assertEqual(v['third_images'],[])
        with self.assertRaises(ValueError):m.witness(b,n,(1,0,2,0),excluded=(1,))

    def test_swap_canonical_result(self):
        b,n=geometry_fixture();self.assertEqual(m.witness(b,n,(1,0,2,0)),m.witness(b,n,(2,0,1,0)))

    def test_zero_length(self):
        b,n=geometry_fixture();b[2]=b[1].copy();v=m.witness(b,n,(1,0,2,0));self.assertFalse(v['accepted']);self.assertEqual(v['reason'],'zero_length_connection')

    def test_nonfinite_is_explicit_and_json_serializable(self):
        b,n=geometry_fixture((np.nan,0));v=m.witness(b,n,(1,0,2,0));self.assertFalse(v['accepted']);self.assertIn('nonfinite_witness',v['third_images'][0]['counts']);json.dumps(v,allow_nan=False)
        b[1][0,0]=np.inf;self.assertEqual(m.witness(b,n,(1,0,2,0))['reason'],'nonfinite_endpoint')

    def test_one_source_count_even_multiple_points(self):
        b,n=geometry_fixture();b[3]=np.array([[.75,0],[1.25,0]]);n[1,3]=n[2,3]=np.array([[0,1]])
        v=m.witness(b,n,(1,0,2,0));self.assertEqual(v['sources'],[3]);self.assertEqual(len(v['third_images'][0]['passing_points']),2)

    def test_finite_candidates_not_exhaustive_segment_search(self):
        b,n=geometry_fixture((0.,2.));b[3]=np.array([[0.,2.],[1.,0.]])
        self.assertFalse(m.witness(b,n,(1,0,2,0))['accepted'])


class NeighborsAndReduction(unittest.TestCase):
    def test_actual_blocked_scipy_matches_direct_enumeration(self):
        rng=np.random.default_rng(8);q=rng.normal(size=(9,7)).astype(np.float32);r=rng.normal(size=(13,7)).astype(np.float32);r[3]=r[0]
        d=np.linalg.norm(q.astype(float)[:,None]-r.astype(float)[None],axis=2);ref=np.argsort(d,axis=1,kind='stable')[:,:5]
        for qc,rc in ((1,1),(4,3),(10,50)):
            idx,dist=neighbors.topk(q,r,query_chunk=qc,reference_chunk=rc)
            np.testing.assert_array_equal(idx,ref);np.testing.assert_allclose(dist,np.take_along_axis(d,ref,1),rtol=1e-15,atol=1e-15)

    def test_all_ties_original_indices_across_blocks(self):
        idx,d=neighbors.topk(np.zeros((3,2)),np.ones((12,2)),query_chunk=2,reference_chunk=2)
        np.testing.assert_array_equal(idx,np.tile(np.arange(5),(3,1)))

    def test_small_pool_not_padded_or_duplicated(self):
        with self.assertRaises(ValueError):neighbors.topk(np.zeros((2,2)),np.zeros((4,2)))

    def test_each_support_searched_separately(self):
        banks={1:np.zeros((5,2)),2:np.ones((6,2)),3:np.full((7,2),10.)};t=neighbors.tables(banks,query_chunk=3,reference_chunk=4)
        self.assertEqual(len(t),6);self.assertEqual(t[1,3][0].shape,(5,5))

    def test_matching_and_declared_float32_rounding(self):
        rng=np.random.default_rng(42);u=rng.uniform(.1,4,(1000,)).astype(np.float32);s=(u*.6).astype(np.float32);g=rng.random(1000)<.3
        d,v=m.reduction(u,s,g);np.testing.assert_array_equal(d[m.W][~g],u[~g]);np.testing.assert_array_equal(d[m.W][g],s[g])
        self.assertLessEqual(v['uniform_matching_abs_error'],v['uniform_matching_rounding_bound'])
        self.assertAlmostEqual(v['beta'],np.sum(g*(u.astype(float)-s))/np.sum(u.astype(float)-s))

    def test_no_reduction_and_rejection_fall_back_full_union(self):
        u=np.array([1,2],np.float32);s=np.array([.2,.5],np.float32)
        d,v=m.reduction(u,s,np.zeros(2,bool));np.testing.assert_array_equal(d[m.W],u);np.testing.assert_array_equal(d[m.C],u)
        d,v=m.reduction(u,u,np.ones(2,bool));self.assertEqual(v['beta'],0)

    def test_tolerated_negative_logged_significant_error_rejected(self):
        u=np.ones(2,np.float32);s=u.copy();s[0]+=np.float32(1e-7)
        d,v=m.reduction(u,s,np.ones(2,bool));self.assertEqual(v['tolerated_negative_reductions'],1);np.testing.assert_array_equal(d[m.W],u)
        with self.assertRaises(ValueError):m.reduction(u,u+.01,np.ones(2,bool))

    def test_mechanism_signatures_have_no_GT_label_or_query_response(self):
        import inspect
        self.assertEqual(tuple(inspect.signature(m.witness).parameters),('banks','neighbors','edge','excluded'))
        self.assertEqual(tuple(inspect.signature(supports.candidate_gates).parameters),('target','indices','relations','excluded','support_ids'))


class SyntheticArtifacts(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=SCRATCH);self.root=Path(self.temp.name).resolve()
        self.assertIn(SCRATCH.resolve(),self.root.parents)
    def tearDown(self):self.temp.cleanup()

    def support_fixture(self):
        records={'background':[]};locations={}
        for i in range(1,9):
            bank=np.zeros((9,1280),np.float32);bank[:,0]=np.arange(9)+i*.13;bank[:,1]=i*.07
            p=self.root/f'{i:02d}.npz';rel=f'test/good/background/{i}.png'
            np.savez(p,protocol_id=np.asarray(ip.PROTOCOL_ID),relative_path=np.asarray(rel),grid=np.array([3,3]),**{f'layer_{l}':bank for l in m.LAYERS})
            records['background'].append(dict(support_index=i,relative_path=rel,grid=[3,3],feature_sha256=st.sha(p),domain='background'))
            locations['background',i]=str(p)
        bundle=SimpleNamespace(identity='synthetic',frozen={'fixture':'original_complete_rows'},protocol=m.PROTOCOL,expected=records,protected_paths=list(map(Path,locations.values())))
        return bundle,locations

    def test_real_npz_header_and_original_feature_rows(self):
        p=self.root/'01.npz';bank=np.arange(9*1280,dtype=np.float32).reshape(9,1280)
        np.savez(p,protocol_id=np.asarray(ip.PROTOCOL_ID),relative_path=np.asarray('test/good/background/a.png'),grid=np.array([3,3]),**{f'layer_{l}':bank for l in m.LAYERS})
        row=dict(support_index=1,relative_path='test/good/background/a.png',grid=[3,3],feature_sha256=st.sha(p));records={'background':[row]}
        ip.feature_header(str(p),row);got=ip.load_layer(records,{('background',1):str(p)},'background',7)
        np.testing.assert_array_equal(got[1],bank)
        wrong=dict(row,feature_sha256='0'*64)
        with self.assertRaises(ValueError):ip.feature_header(str(p),wrong)

    def test_hash_mutation_is_not_reenrolled(self):
        path=self.root/'result.npz';st.seal(path,'identity',{'distance':np.ones(3,np.float32)},{'scope':'test'})
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaises(ValueError):st.restore(path,'identity')

    def test_unique_edge_cache_order_resume_and_holdout_namespace(self):
        b,n=geometry_fixture();path=self.root/'connections.sqlite'
        with supports.Connections(path,'id',b,n,'neighbors') as c:
            v=c.get((1,0,2,0));self.assertEqual(c.get((2,0,1,0)),v);self.assertEqual(c.new,1)
            self.assertFalse(c.get((1,0,2,0),excluded=(3,))['accepted'])
        with supports.Connections(path,'id',b,n,'neighbors') as c:
            self.assertEqual(c.get((2,0,1,0)),v);self.assertEqual(c.new,0)
        with self.assertRaises(ValueError):supports.Connections(path,'other',b,n,'neighbors')

    def test_compact_reordering_and_manifest_ties(self):
        b,n=geometry_fixture();t=np.array([[1.,3.,1.],[1.,1.,2.],[3.,1.,3.]])
        with supports.Connections(self.root/'x.sqlite','id',b,n,'nb') as c:
            idx=np.zeros(t.shape,np.int64);g,v=supports.candidate_gates(t,idx,c)
            order=[2,0,1];again,_=supports.candidate_gates(t[:,order],idx[:,order],c)
            np.testing.assert_array_equal(again,g[order])
            roworder=[2,0,1];again,_=supports.candidate_gates(t[roworder],idx[roworder],c,support_ids=np.array([1,2,3])[roworder])
            np.testing.assert_array_equal(again,g)

    def test_actual_production_prepare_and_whole_image_loo(self):
        # Complete synthetic 8x4 FP32 banks, actual header reader, SciPy NN,
        # sealed tables, original geometry_layer, and actual LOO orchestration.
        bundle,locations=self.support_fixture()
        out=self.root/'run';st.initialize(out,bundle);supports.configure(out,'cpu',3,4)
        status=supports.prepare(bundle,out,locations,'background',7,query_chunk=3,reference_chunk=4)
        self.assertEqual(status['new_pairs'],56);self.assertEqual(supports.prepare(bundle,out,locations,'background',7,query_chunk=3,reference_chunk=4)['new_pairs'],0)
        result=supports.support_loo(bundle,out,locations,'background',7);self.assertEqual(len(result['rows']),32)
        for row in result['rows']:
            self.assertNotIn(str(row['heldout_support']),row['reference_supports'].split(','));self.assertEqual(row['scope'],'TARGET_ONLY_LOO_COVERAGE')
        self.assertEqual(supports.support_loo(bundle,out,locations,'background',7)['rows_sha256'],result['rows_sha256'])
        path=supports.pair_path(out,'background',7,1,2);path.write_bytes(path.read_bytes()+b'bad')
        with self.assertRaises(ValueError):supports.prepare(bundle,out,locations,'background',7,query_chunk=3,reference_chunk=4)

    def test_production_cached_prediction_freeze_scores_and_resume(self):
        bundle,locations=self.support_fixture();out=self.root/'run';st.initialize(out,bundle);supports.configure(out,'cpu',3,4)
        banks=ip.load_layer(bundle.expected,locations,'background',7)
        query=np.zeros((9,1280),np.float32);query[:,0]=np.arange(9)+.7;query[:,1]=.3
        top=[neighbors.topk(query,banks[c],k=1) for c in sorted(banks)]
        distance=np.stack([d[:,0] for _,d in top]).astype(np.float64)/1280;distance=distance.astype(np.float32)
        index=np.stack([i[:,0] for i,_ in top]);source=np.ones(9,np.float32)
        geometry,_=supports.geometry_layer(distance,index,source,list(banks.values()))
        order=np.array([8,0,4,7,2,1,5,3,6],np.int64)
        raw=dict(positions=order,source_distance=np.tile(source[order],(4,1)),target_distance=np.repeat(distance[:,None,order],4,axis=1),
            target_index=np.repeat(index[:,None,order],4,axis=1),segment_distance=np.tile(geometry['K8_SEGMENT'][order],(4,1)),
            union_distance=np.tile(distance.min(0)[order],(4,1)))
        meta=dict(grid=[3,3],output_shape=[6,6]);origin={'synthetic_original_raw':'immutable'}
        sample=dict(domain='background',sample_id='test_fake',relative_path='test/good/background/query.png',label=0,image_height=24,image_width=24)
        bundle.samples=[sample];bundle.load=lambda s:(copy.deepcopy(raw),dict(meta),dict(origin))
        bundle.medians=np.zeros(4,np.float32);bundle.scales=np.ones(4,np.float32)
        bundle.state={part:dict(medians=[0.]*4,scales=[1.]*4) for part in ('peak','local')};bundle.references={}
        for b,value in zip(m.BASES,(raw['union_distance'],raw['segment_distance'])):
            sc=prediction.score(bundle,value,raw['positions'],meta);bundle.references['background','test_fake',b]=dict(image_score=sc['CH1'],current_check_score=sc['CURRENT'])
        for l in m.LAYERS:
            supports.prepare(bundle,out,locations,'background',l,query_chunk=3,reference_chunk=4)
            result=prediction.predict_layer(bundle,out,locations,'background',l);self.assertEqual(result['new_images'],1)
        frozen=prediction.freeze_domain(bundle,out,'background');self.assertEqual(frozen['status'],'predictions_frozen')
        prediction.verified(bundle,out,'background');a,metadata,_=prediction.load_record(bundle,out,'background','test_fake')
        np.testing.assert_array_equal(a['positions'],order);self.assertFalse(metadata['GT_used']);self.assertFalse(metadata['labels_used'])
        for b in m.NEW:self.assertEqual(a[b].shape,(4,9))
        old=st.sha(out/'prediction_manifests/background.json')
        for l in m.LAYERS:self.assertEqual(prediction.predict_layer(bundle,out,locations,'background',l)['reused_images'],1)
        prediction.freeze_domain(bundle,out,'background');self.assertEqual(st.sha(out/'prediction_manifests/background.json'),old)
        audit=out/'prediction_tables/background/reduction_summary.csv';before=audit.read_bytes();audit.write_bytes(before+b'changed')
        changed=st.sha(audit)
        with self.assertRaises(ValueError):prediction.freeze_domain(bundle,out,'background')
        self.assertEqual(st.sha(audit),changed)  # failed resume preserves evidence
        audit.write_bytes(before)
        p=prediction.image_path(out,'background','test_fake');p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaises(ValueError):prediction.verified(bundle,out,'background')

    def test_heldout_reference_bank_never_read_by_LOO_geometry_or_witness(self):
        bundle,locations=self.support_fixture();banks=ip.load_layer(bundle.expected,locations,'background',7)
        table=neighbors.tables(banks,query_chunk=4,reference_chunk=5);idx={k:v[0] for k,v in table.items()};dist={k:v[1] for k,v in table.items()}
        class Guard(dict):
            def __getitem__(self,key):
                if key==3:raise AssertionError('held-out normal query was used as endpoint/witness')
                return super().__getitem__(key)
        guarded=Guard(banks)
        with supports.Connections(self.root/'guard.sqlite','id',guarded,idx,'n') as c:
            rows,_=supports.loo_one(guarded,idx,dist,3,c)
        self.assertEqual(len(rows),4)

    def test_incomplete_predictions_cannot_enter_evaluation(self):
        out=self.root/'run';(out/'prediction_manifests').mkdir(parents=True)
        st.write_json(out/'prediction_manifests/background.json',dict(identity='id',status='partial_predictions',images=0,GT_used=False,labels_used=False))
        b=SimpleNamespace(identity='id',samples=[dict(domain='background',sample_id='x')])
        with self.assertRaises(ValueError):prediction.verified(b,out,'background')

    def test_missing_features_are_explicit_not_auto_extracted(self):
        bundle,_=self.support_fixture()
        location,available,missing,rejected,archives=ip.locate(bundle,[self.root/'not_present'],['background'])
        self.assertEqual(len(missing),8);self.assertFalse(location);self.assertFalse(available)
        with self.assertRaises(FileNotFoundError):ip.load_layer(bundle.expected,location,'background',7)

    def test_support_discovery_uses_header_and_SHA_not_numeric_filename(self):
        bundle,locations=self.support_fixture();first=Path(locations['background',1]);renamed=first.with_name('restored_original_bank.npz');first.rename(renamed)
        found,available,missing,_,_=ip.locate(bundle,[self.root],['background'])
        self.assertFalse(missing);self.assertEqual(len(available),8);self.assertEqual(Path(found['background',1]),renamed)

    def test_incomplete_LOO_cannot_be_complete_even_with_valid_rows_digest(self):
        saved=dict(identity='id',neighbors='nb',status='target_only_support_LOO_complete',domain='background',layer=7,rows=[],rows_sha256=st.digest([]))
        with self.assertRaises(ValueError):supports.validate_loo(saved,'id','background',7,'nb')

    def test_old_branch_lists_unchanged(self):
        from DINOv3.MADEqual.aebad_s_gap_soft_segment_p0 import method as old
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import method as cross
        self.assertEqual(len(old.BRANCHES),6);self.assertEqual(len(cross.BRANCHES),3)
        self.assertNotIn(m.W,old.BRANCHES);self.assertNotIn(m.W,cross.BRANCHES)

    def test_loss_transfers_keep_partial_and_NA(self):
        rows=[]
        for iid,values in [(1,[1.,0.,.5,.7]),(2,[None]*4)]:
            for b,v in zip(m.BRANCHES,values):rows.append(dict(domain='background',sample_id='x',instance_id=iid,branch=b,margin=v,fixed_union_margin=v,small_group=True,response_status='available' if v is not None else 'mapped_instance_disappeared'))
        detail,summary=reporting.margin_transitions(rows)
        row=next(v for v in summary if v['branch']==m.W and v['definition']=='margin' and v['group']=='all')
        self.assertEqual(row['NA'],1);self.assertEqual(row['old_loss_partially_repaired'],1);self.assertEqual(row['old_loss_fully_repaired'],0)

    def test_CPU_import_and_NN_do_not_import_torch(self):
        code='import sys; from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import runner, neighbors; import numpy as np; neighbors.topk(np.zeros((1,2)),np.zeros((5,2))); assert "torch" not in sys.modules; assert runner.parser().parse_args([]).stage=="preflight"; print("CPU_NO_TORCH")'
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('CPU_NO_TORCH',result.stdout)


class ExactPixelCheckpoint(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=SCRATCH)
        self.root=Path(self.temp.name).resolve();self.assertIn(SCRATCH.resolve(),self.root.parents)
        self.path=self.root/'WITNESS_SEGMENT_K8_exact_pixels.json';self.seal=self.path.with_name(self.path.stem+'.seal.json')
        self.labels=np.array([0,1,0,1,0,1],np.uint8);self.scores=np.array([.1,.7,.8,.9,.2,.6],np.float32)
        self.counts=dict(pixels=6,positive_pixels=3,negative_pixels=3)
        self.options=dict(domain='background',branch=m.W,prediction_sha='frozen_prediction',counts=self.counts)
    def tearDown(self):self.temp.cleanup()

    def run_result(self,identity='id',map_identity='maps',**options):
        return ev.exact_pixel_result(self.path,identity,map_identity,self.labels,self.scores,**(self.options|options))

    def test_actual_kernel_checkpoint_reused_without_sort_or_domain_status(self):
        first=self.run_result();self.assertTrue(self.seal.is_file());self.assertFalse((self.root/'status.json').exists())
        saved=self.path.read_bytes();seal=self.seal.read_bytes()
        with patch.object(ev.pixels,'exact_binary',side_effect=AssertionError('unexpected repeated sort')):self.assertEqual(self.run_result(),first)
        self.assertEqual(self.path.read_bytes(),saved);self.assertEqual(self.seal.read_bytes(),seal)

    def test_tampered_metric_rejected_and_original_seal_preserved(self):
        self.run_result();original=st.read_json(self.path);seal=self.seal.read_bytes()
        for value in (1.5,.7):
            changed=copy.deepcopy(original);changed['result']['AUROC']=value;st.write_json(self.path,changed);before=self.path.read_bytes()
            with patch.object(ev.pixels,'exact_binary',side_effect=AssertionError('must not conceal tampering by recomputing')):
                with self.assertRaisesRegex(ValueError,'modified/incompatible'):self.run_result()
            self.assertEqual(self.path.read_bytes(),before);self.assertEqual(self.seal.read_bytes(),seal)

    def test_legacy_unsealed_intermediate_is_rejected_not_reenrolled(self):
        st.write_json(self.path,dict(identity='id',map_identity='maps',result=dict(AUROC=.7,AP=.8)))
        before=self.path.read_bytes()
        with patch.object(ev.pixels,'exact_binary',side_effect=AssertionError('unexpected sort')):
            with self.assertRaisesRegex(ValueError,'unsealed'):self.run_result()
        self.assertEqual(self.path.read_bytes(),before);self.assertFalse(self.seal.exists())

    def test_orphan_seal_is_rejected(self):
        self.run_result();self.path.unlink();before=self.seal.read_bytes()
        with self.assertRaisesRegex(ValueError,'unsealed'):self.run_result()
        self.assertFalse(self.path.exists());self.assertEqual(self.seal.read_bytes(),before)

    def test_identity_domain_branch_and_pixel_counts_must_match_seal(self):
        self.run_result()
        cases=[dict(identity='different'),dict(map_identity='different'),dict(domain='view'),dict(branch=m.C),
            dict(prediction_sha='different'),dict(counts=dict(pixels=7,positive_pixels=3,negative_pixels=4))]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):self.run_result(**options)

    def test_sealed_but_semantically_invalid_values_counts_and_flags_rejected(self):
        self.run_result();original=st.read_json(self.path);seal=st.read_json(self.seal)
        cases=[('AUROC',1.5),('AUROC',-.1),('AUROC',float('nan')),('AUROC',float('inf')),
            ('AUROC',None),('AUROC',True),('AP',1.1),('AP','0.7'),('pixels',7),('pixels',6.),
            ('positive_pixels',2),('negative_pixels',2),('valid',False),('AUROC_valid',False),
            ('AP_valid',False),('status','unfinished')]
        for key,value in cases:
            with self.subTest(key=key,value=value):
                changed=copy.deepcopy(original);changed['result'][key]=value
                # This tests semantic validation AFTER a valid file-hash check,
                # including malformed JSON numeric NaN/Infinity. Not a model mock.
                self.path.write_text(json.dumps(changed,allow_nan=True),encoding='utf-8')
                st.write_json(self.seal,dict(seal,result_file_sha256=st.sha(self.path)))
                with self.assertRaises(ValueError):self.run_result()

    def test_invalid_new_backend_result_is_never_committed(self):
        from contextlib import nullcontext
        value=dict(AUROC=1.5,AP=.8,**self.counts,valid=True,AUROC_valid=True,AP_valid=True,status='exact_all_pixels_cpu_tie_grouped')
        # This is only an injected invalid-result test, not an assertion about
        # resource-policy AST rewriting (which requires the real function).
        with patch.object(ev,'observation_only',new=nullcontext),patch.object(ev.pixels,'exact_binary',return_value=value):
            with self.assertRaises(ValueError):self.run_result()
        self.assertFalse(self.path.exists());self.assertFalse(self.seal.exists())

    def test_interruption_between_result_and_seal_cannot_become_trusted(self):
        original_write=st.write_json
        def interrupted(path,value):
            if Path(path)==self.seal:raise OSError('simulated interruption before seal commit')
            original_write(path,value)
        with patch.object(st,'write_json',side_effect=interrupted):
            with self.assertRaises(OSError):self.run_result()
        self.assertTrue(self.path.exists());self.assertFalse(self.seal.exists());before=self.path.read_bytes()
        with patch.object(ev.pixels,'exact_binary',side_effect=AssertionError('unexpected sort')):
            with self.assertRaisesRegex(ValueError,'unsealed'):self.run_result()
        self.assertEqual(self.path.read_bytes(),before)


if __name__=='__main__':unittest.main()
