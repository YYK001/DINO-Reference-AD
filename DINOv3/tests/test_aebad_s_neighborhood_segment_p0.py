"""Small synthetic production-kernel tests. No real 297-image prediction is run."""
from __future__ import annotations

import ast
import importlib.abc
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0 import gates as g, inputs as ip
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0 import prediction as pr
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0 import analysis as an

ROOT=Path(__file__).resolve().parents[2]
PYTHON=sys.executable


def fixture(grid=(5,5)):
    n=int(np.prod(grid));positions=np.arange(n,dtype=np.int64)
    arrays=dict(positions=positions,source_distance=np.full((4,n),3,np.float32),
                target_distance=np.stack([np.full((4,n),v,np.float32) for v in (1,1.5,3,4,5,6,7,8)]),
                target_index=np.broadcast_to(positions,(8,4,n)).copy())
    meta=dict(grid=list(grid),supports=[dict(grid=list(grid)) for _ in range(8)])
    segments={l:dict(positions=positions.copy(),segment_distance=np.full(n,.5,np.float32),
                     endpoint_support=np.broadcast_to(np.array([1,2],np.int8)[:,None],(2,n)).copy(),
                     endpoint_index=np.broadcast_to(positions,(2,n)).copy()) for l in g.LAYERS}
    return arrays,meta,segments


class GatesTest(unittest.TestCase):
    def test_identity_correspondence_all_accept_existing_segment(self):
        a,m,s=fixture();out,rows=g.compute(a,m,s)
        self.assertTrue(out['geom_gate'].all());self.assertTrue(out['pair_gate'].all())
        np.testing.assert_array_equal(out['neighbor_distance'],.5*np.ones((4,25),np.float32))
        self.assertEqual(rows[0]['segment_reduced_patches'],25)

    def test_compact_permutations_in_both_inputs_are_physical_invariant(self):
        a,m,s=fixture();original,_=g.compute(a,m,s)
        rng=np.random.default_rng(42);p=rng.permutation(25);q=rng.permutation(25)
        altered={k:v[...,p] for k,v in a.items()}
        seg={l:{k:v[...,q] for k,v in data.items()} for l,data in s.items()}
        actual,_=g.compute(altered,m,seg)
        for name in ('pair_gate','geom_gate','neighbor_count','pair_votes','geom_votes'):
            np.testing.assert_array_equal(actual[name],original[name])
        for name in ('pair_distance','neighbor_distance'):
            np.testing.assert_array_equal(g.scatter(actual[name],actual['positions'],(5,5)),
                                          g.scatter(original[name],original['positions'],(5,5)))

    def test_unordered_pair_swap(self):
        a,m,s=fixture();data=s[7];p=a['positions'];grids=[(5,5)]*8
        first=g.neighborhood(data['endpoint_support'],data['endpoint_index'],p,(5,5),grids)
        support=data['endpoint_support'].copy();index=data['endpoint_index'].copy()
        support[:,::2]=support[::-1,::2];index[:,::2]=index[::-1,::2]
        other=g.neighborhood(support,index,p,(5,5),grids)
        for k in first: np.testing.assert_array_equal(first[k],other[k])

    def test_translation_cancels_without_absolute_coordinate_equality(self):
        a,m,s=fixture();data=s[7];yy,xx=np.indices((5,5))
        translated=np.clip(yy-1,0,4)*5+np.clip(xx-1,0,4)
        indices=np.stack([translated.ravel()]*2)
        out=g.neighborhood(data['endpoint_support'],indices,a['positions'],(5,5),[(5,5)]*8)
        self.assertEqual(out['geom_votes'][2,2],8)
        self.assertTrue(out['geom_gate'][2,2])

    def test_same_source_but_correspondence_jump_rejected(self):
        a,m,s=fixture();indices=np.zeros((2,25),np.int64);indices[:,12]=24
        out=g.neighborhood(s[7]['endpoint_support'],indices,a['positions'],(5,5),[(5,5)]*8)
        self.assertTrue(out['pair_gate'][2,2]);self.assertFalse(out['geom_gate'][2,2])
        self.assertEqual(out['geom_votes'][2,2],0)

    def test_both_endpoints_must_support_the_same_neighbor_vote(self):
        a,m,s=fixture();indices=s[7]['endpoint_index'].copy()
        indices[1]=0;indices[1,12]=24
        out=g.neighborhood(s[7]['endpoint_support'],indices,a['positions'],(5,5),[(5,5)]*8)
        self.assertTrue(out['pair_gate'][2,2]);self.assertFalse(out['geom_gate'][2,2])

    def test_center_excluded_real_boundary_neighbors_half_ceil(self):
        a,m,s=fixture((3,3));data=s[7]
        support=np.broadcast_to(np.array([3,4],np.int8)[:,None],(2,9)).copy();support[:,4]=[1,2]
        out=g.neighborhood(support,data['endpoint_index'],a['positions'],(3,3),[(3,3)]*8)
        self.assertEqual(out['neighbor_count'][0,0],3)
        self.assertEqual(out['neighbor_count'][0,1],5)
        self.assertEqual(out['neighbor_count'][1,1],8)
        self.assertEqual(out['pair_votes'][1,1],0)
        self.assertFalse(out['pair_gate'][1,1])
        # A corner needs 2 of its 3 real neighbors; no padding/center votes.
        support[:]=np.array([3,4])[:,None];support[:,0]=[1,2];support[:,1]=[1,2];support[:,3]=[1,2]
        out=g.neighborhood(support,data['endpoint_index'],a['positions'],(3,3),[(3,3)]*8)
        self.assertTrue(out['pair_gate'][0,0]);self.assertEqual(out['pair_votes'][0,0],2)

    def test_no_neighbors_reject_and_small_shapes(self):
        support=np.array([[1],[2]],np.int8);indices=np.zeros((2,1),np.int64)
        out=g.neighborhood(support,indices,np.arange(1,dtype=np.int64),(1,1),[(1,1)]*8)
        self.assertFalse(out['pair_gate'].any());self.assertFalse(out['geom_gate'].any())

    def test_repeated_correspondence_recorded_not_assumed_unique(self):
        a,m,s=fixture();indices=np.zeros((2,25),np.int64)
        out=g.neighborhood(s[7]['endpoint_support'],indices,a['positions'],(5,5),[(5,5)]*8)
        self.assertEqual(out['repeated_both_votes'][2,2],8)
        self.assertTrue(out['geom_gate'][2,2]) # tolerance one permits this; no uniqueness claim.

    def test_invalid_geometry_indices_and_stable_tie_selection(self):
        a,m,s=fixture();m['supports'][0]['grid']=[4,5]
        with self.assertRaises(ValueError): g.compute(a,m,s)
        a,m,s=fixture();s[7]['endpoint_index'][0,0]=1000
        with self.assertRaises(ValueError): g.compute(a,m,s)
        a,m,s=fixture();a['target_distance'][1]=1
        out,_=g.compute(a,m,s);self.assertTrue(out['pair_gate'].all())
        s[7]['endpoint_support']=s[7]['endpoint_support'][::-1].copy()
        with self.assertRaises(ValueError): g.compute(a,m,s) # audit must preserve original tie rule

    def test_full_rejection_restores_full_union_not_source(self):
        a,m,s=fixture();colors=[(i,j) for i in range(1,9) for j in range(i+1,9)][:9]
        for pos in range(25):
            y,x=divmod(pos,5);pair=colors[(y%3)*3+x%3]
            a['target_distance'][:,:,pos]=4
            a['target_distance'][pair[0]-1,:,pos]=1
            a['target_distance'][pair[1]-1,:,pos]=1.5
            for layer in g.LAYERS: s[layer]['endpoint_support'][:,pos]=pair
        out,_=g.compute(a,m,s)
        self.assertFalse(out['pair_gate'].any());self.assertFalse(out['geom_gate'].any())
        np.testing.assert_array_equal(out['pair_distance'],np.ones((4,25),np.float32))
        self.assertFalse(np.array_equal(out['pair_distance'],a['source_distance']))

    def test_containment_no_silent_clip(self):
        u=np.array([1,2,3,4],np.float32);s=u-.5
        pair=np.array([False,True,True,True]);geom=np.array([False,False,True,False])
        p,gated=g.bounded(u,s,pair,geom)
        self.assertTrue(np.all(s<=p));self.assertTrue(np.all(p<=gated));self.assertTrue(np.all(gated<=u))
        with self.assertRaises(ValueError): g.bounded(u,u+1,pair,geom)
        with self.assertRaises(ValueError): g.bounded(u,s,geom,pair)

    def test_labels_are_not_prediction_inputs(self):
        a,m,s=fixture();before,_=g.compute(a,m,s)
        m['label']=1;m['GT']='arbitrary ignored metadata'
        after,_=g.compute(a,m,s)
        for k in before:np.testing.assert_array_equal(before[k],after[k])
        self.assertNotIn('sample',inspect.signature(g.compute).parameters)


class IntegrationTest(unittest.TestCase):
    def test_cuda_continuation_must_reproduce_CPU_map_and_GT(self):
        response=np.arange(12,dtype=np.float32).reshape(3,4);mask=np.zeros((3,4),np.uint8)
        record=dict(output_height=3,output_width=4,map_float32_sha256=an.array_sha(response),mask_uint8_sha256=an.array_sha(mask))
        an.verify_map_record(record,response,mask)
        with self.assertRaises(ValueError):an.verify_map_record(record,response+1e-5,mask)
        mask[0,0]=1
        with self.assertRaises(ValueError):an.verify_map_record(record,response,mask)

    def small_evaluation(self,directory,break_baseline=False):
        """Four tiny synthetic images through REAL evaluation math/outputs.

        Only archive admission, Dataset IO and the frozen-cache handoff are test
        doubles. These do not claim the real archive or CUDA chain was evaluated.
        Real production cache/gate/tamper paths have separate tests above.
        """
        from contextlib import ExitStack
        from types import SimpleNamespace
        root=Path(directory);pred=root/'prediction';pred.mkdir()
        ip.write_json(pred/'prediction_manifest.json',dict(synthetic=True))
        samples=[dict(sample_id=f'q{i}',relative_path=f'test/x/background/{i}.png',label=int(i>=2),
                      image_height=12,image_width=12,union_score=v) for i,v in enumerate((.5,.75,2.,3.))]
        state={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')};state['state_sha256']='synthetic_state'
        ctx=dict(identity='synthetic_fixture',manifest={},paths={'raw_archive':root/'synthetic.tar'},raw_index={},samples=samples,
                 medians=np.zeros(4,np.float32),scales=np.ones(4,np.float32),state=state)
        old=dict(scores={},instances={},peaks={},fixed10={'q0'},small3={('q2',1)},ch1_scores={},ch1_metrics=[],ch1_rank=[])
        bank={};masks={};dummy_rows=[]
        for sample in samples:
            sid=sample['sample_id'];a,m,s=fixture((3,3));value=sample['union_score']
            a['target_distance'][0]=value;a['target_distance'][1]=value+.2
            a['target_distance'][2:]=value+1;a['source_distance'][:]=value+2
            m.update(relative_path=sample['relative_path'],output_shape=[3,3])
            for l in g.LAYERS:s[l]['segment_distance'][:]=np.float32(value*.8)
            saved,gate_rows=g.compute(a,m,s);bank[sid]=(a,m,s,saved)
            dummy_rows.extend(dict(sample_id=sid,**r) for r in gate_rows)
            mask=np.zeros((12,12),bool)
            if sample['label']:mask[4:8,4:8]=True
            masks[sid]=mask
            u,seg=pr.original_distances(a,m,s)
            for b,d in ((an.U,u),(an.S,seg)):
                current,score,response,_=an.score_maps(ctx,d,dict(m,positions=a['positions']))
                old['scores'][(sid,an.HISTORY_BRANCH[b])]=current
                old['ch1_scores'][(sid,an.U)]=value
                old['peaks'][(sid,'full',0)]=(0,0)
                if sample['label']:
                    inside,ring=an.instance_regions(mask,mask,(3,3))
                    for name,region in (('instance',inside),('ring',ring)):
                        old['peaks'][(sid,name,1)]=an.peak_position(response,region)
                    reference=dict(area_original_pixels=16,instance_area_output_pixels=int(inside.sum()),
                                   ring_area_output_pixels=int(ring.sum()),patch_coverage=int(an.adaptive_max(mask,(3,3)).sum()),
                                   instance_max=float(response[inside].max()),ring_max=float(response[ring].max()),margin=0.)
                    old['instances'][(sid,1,an.HISTORY_BRANCH[b])]=reference
                    old['instances'][(sid,1,'K8_MIDPOINT')]=reference
            # Historical metric rows also identify the pixel-source provenance.
        y,groups=an.group_indices(samples,old['fixed10'])
        vals=np.array([x['union_score'] for x in samples])
        for name,use in groups.items():
            old['ch1_metrics'].append(dict(branch=an.U,normal_group=name,**an.image_metrics(y[use],vals[use])))
            old['ch1_rank'].append(dict(branch=an.U,normal_group=name,weighted_pair_errors=float(an.pair_errors(y[use],vals[use]).sum())))
        old['metrics.csv']=[dict(branch=b,pixel_AUROC=.8,pixel_AP=.4,AUPRO=.7) for b in ('K8_UNION','K8_SEGMENT')]
        ip.write_csv(pred/'gate_statistics.csv',dummy_rows)
        class NoArchive:
            def __init__(self,*args,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
        class SyntheticData(NoArchive):
            def gt(self,sample):return masks[sample['sample_id']]
        real_score=an.score_maps
        def changed(*args):
            current,score,response,error=real_score(*args)
            return current+1,score+1,response+1,error
        with ExitStack() as stack:
            for target,replacement in (
                ('verified_predictions',lambda *args:(pred,{s['sample_id']:dict(sample_id=s['sample_id']) for s in samples})),
                ('history',lambda *args:old),('Dataset',SyntheticData),
                ('load_prediction',lambda ctx,pred,record,*args:bank[record['sample_id']][3])):
                stack.enter_context(mock.patch.object(an,target,replacement))
            stack.enter_context(mock.patch.object(ip,'Archive',NoArchive))
            stack.enter_context(mock.patch.object(ip,'load_image',lambda ctx,archive,sample:bank[sample['sample_id']][:3]))
            stack.enter_context(mock.patch.object(ip,'context',lambda *args,**kwargs:ctx))
            if break_baseline:stack.enter_context(mock.patch.object(an,'score_maps',changed))
            output=root/'evaluation'
            if break_baseline:
                with self.assertRaises(ValueError):an.evaluate(ctx,pred,output,workers=2,memory_budget_bytes=1<<30)
                self.assertFalse((output/'metrics.csv').exists())
                self.assertNotEqual(ip.json_file(output/'status.json')['status'],'cpu_evaluation_complete_aupro_pending')
            else:
                an.evaluate(ctx,pred,output,workers=2,memory_budget_bytes=1<<30)
                status=ip.json_file(output/'status.json');self.assertFalse(status['all_metrics_complete'])
                rows=ip.read_csv((output/'metrics.csv').read_bytes())
                new=[r for r in rows if r['branch'] in (an.P,an.G) and r['normal_group']=='all85']
                self.assertEqual(len(new),2);self.assertTrue(all(r['AUPRO']=='' for r in new))
                self.assertTrue(all('new_exact_CPU' in r['pixel_source'] for r in new))
                self.assertTrue((output/'objective_report.md').is_file());self.assertFalse((output/'_scratch').exists())
                failures=[r for r in ip.read_csv((output/'numerical_verification.csv').read_bytes()) if r['verified']!='True']
                self.assertEqual(failures,[])

    def test_small_production_evaluation_integration_aupro_stays_pending(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'downloads') as directory:self.small_evaluation(directory)

    def test_consistently_changed_scoring_backend_cannot_publish_metrics(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'downloads') as directory:self.small_evaluation(directory,break_baseline=True)

    def test_synthetic_cache_handoff_and_tamper_rejection(self):
        a,m,s=fixture();out,_=g.compute(a,m,s)
        ctx=dict(identity='synthetic',captures={'q':{'sha256':'cap'}},segments={('q',l):{'sha256':str(l)} for l in g.LAYERS})
        meta=dict(m,relative_path='test/good/background/x.png',output_shape=[10,10])
        metadata=dict(identity='synthetic',sample_id='q',relative_path=meta['relative_path'],grid=meta['grid'],
                      output_shape=meta['output_shape'],distance_unit=ip.PROTOCOL['distance_unit'],capture_sha256='cap',
                      segment_sha256={str(l):str(l) for l in g.LAYERS})
        with tempfile.TemporaryDirectory(dir=ROOT/'downloads') as directory:
            root=Path(directory);file=root/'q.npz'
            np.savez_compressed(file,metadata=np.asarray(json.dumps(metadata)),**out)
            record=dict(sample_id='q',file='q.npz',sha256=ip.sha(file))
            saved=pr.load_prediction(ctx,root,record,a,meta,s)
            np.testing.assert_array_equal(saved['neighbor_distance'],out['neighbor_distance'])
            out['neighbor_distance'][0,0]+=.1
            np.savez_compressed(file,metadata=np.asarray(json.dumps(metadata)),**out)
            with self.assertRaises(ValueError):pr.load_prediction(ctx,root,record,a,meta,s)
            # Even manually refreshing a checksum does not bypass the actual kernel check.
            record['sha256']=ip.sha(file)
            with self.assertRaises(ValueError):pr.load_prediction(ctx,root,record,a,meta,s)

    def test_incomplete_predictions_not_complete(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'downloads') as directory:
            ip.write_json(Path(directory)/'status.json',dict(status='predictions_frozen',images=12))
            with self.assertRaises(ValueError):pr.verified_predictions(dict(identity='x'),directory)

    def test_exact_pixels_and_negative_MAD_maps(self):
        from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.metrics import exact_binary
        from sklearn.metrics import roc_auc_score,average_precision_score
        labels=np.array([0,1,1,0,1,0],np.uint8);scores=np.array([-2,-1,-1,0,0,0],np.float32)
        result=exact_binary(labels,scores,memory_budget_bytes=1<<30,chunk_size=2)
        self.assertAlmostEqual(result['AUROC'],roc_auc_score(labels,scores))
        self.assertAlmostEqual(result['AP'],average_precision_score(labels,scores))
        a,m,s=fixture((3,3));m['output_shape']=[6,6];m['positions']=a['positions']
        state={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')}
        ctx=dict(medians=np.ones(4,np.float32)*2,scales=np.ones(4,np.float32),state=state)
        current,score,response,_=an.score_maps(ctx,np.ones((4,9),np.float32),m)
        self.assertTrue(np.all(response<0));self.assertAlmostEqual(current,-1);self.assertAlmostEqual(score,-1)

    def test_mechanism_zero_denominators_are_NA_signed_values_preserved(self):
        a,m,s=fixture((3,3));raw=np.ones((4,9),np.float32)
        raws={b:raw.copy() for b in g.BRANCHES};sample=dict(sample_id='q',label=0)
        prediction={'pair_gate':np.zeros((4,3,3),bool),'geom_gate':np.zeros((4,3,3),bool)}
        rows=an.mechanism_rows(sample,np.zeros((6,6),bool),raws,a['positions'],(3,3),np.ones(4,np.float32),prediction)
        self.assertTrue(all(r['extra_reduction_amount_retained_ratio'] is None for r in rows))
        raws[an.G]+=np.float32(1e-8) # float32 may round; use explicit representable delta below
        raws[an.G][0]+=np.float32(1e-7)
        rows=an.mechanism_rows(sample,np.zeros((6,6),bool),raws,a['positions'],(3,3),np.ones(4,np.float32),prediction)
        self.assertLess(next(r for r in rows if r['branch']==an.G and r['layer']==7)['R_value'],0)

    def test_ranking_direction_and_half_ties(self):
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import pair_errors
        y=np.array([0,0,1]);np.testing.assert_array_equal(pair_errors(y,[1,2,2]),[[0],[.5]])
        self.assertEqual((pair_errors(y,[3,3,2])-pair_errors(y,[1,2,2])).sum(),1.5)

    def test_default_preflight_no_torch_no_output(self):
        # Subprocess blocks all attempted Torch imports, including lazy dependencies.
        script='''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='torch' or name.startswith('torch.'): raise RuntimeError('FORBIDDEN TORCH IMPORT')
sys.meta_path.insert(0,Block())
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.runner import main
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0 import analysis, visualize
assert 'torch' not in sys.modules
assert main(['--raw-archive','G:/definitely_missing_cache_no_write.tar','--no-write'])==2
assert 'torch' not in sys.modules
'''
        result=subprocess.run([PYTHON,'-B','-c',script],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('missing_assets',result.stdout)

    def test_small_evaluation_with_Torch_imports_forbidden(self):
        script='''
import sys, importlib.abc, unittest
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='torch' or name.startswith('torch.'): raise RuntimeError('FORBIDDEN TORCH IMPORT')
sys.meta_path.insert(0,Block())
test='DINOv3.tests.test_aebad_s_neighborhood_segment_p0.IntegrationTest.test_small_production_evaluation_integration_aupro_stays_pending'
result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromName(test))
assert result.wasSuccessful()
assert 'torch' not in sys.modules
'''
        result=subprocess.run([PYTHON,'-B','-c',script],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_production_call_graph_and_math_helpers_no_model_paths(self):
        # Supplement dynamic checks: AUPRO module is not imported by CPU stages.
        from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.runner import parser
        self.assertEqual(parser().parse_args([]).stage,'preflight')
        self.assertFalse(any(name.endswith('aebad_s_neighborhood_segment_p0.aupro') for name in sys.modules))
        self.assertNotIn('fit_image_mad',inspect.getsource(an))
        self.assertNotIn('enabled=',inspect.getsource(g))


if __name__=='__main__': unittest.main()
