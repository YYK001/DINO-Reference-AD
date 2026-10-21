"""Production CPU kernels on small constructions; no 297-image formal analysis."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic import profiles as pr,conditional as co,inputs as ip,spatial as sp,analysis as an
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import bilinear
from DINOv3.tests.test_aebad_s_neighborhood_segment_p0 import fixture

ROOT=Path(__file__).resolve().parents[2]


def constructed_data():
    # Pooled b distribution is uniform; group proportions vary at b=0.
    ids=[];groups=[];b=[]
    for group in (0,1):
        for level in range(6):
            count=(6-level) if group==0 else level+1
            for j in range(count):
                ids.append(len(ids));groups.append(group);b.append(float(level))
    b=np.asarray(b);x=(b==0).astype(float)
    data={k:x.copy() for k in pr.FEATURES}
    data.update(image_id=np.asarray(ids),group=np.asarray(groups),b=b,delta=np.ones(len(b)),effective_reduction=np.ones(len(b),bool))
    return data


class NativeProfiles(unittest.TestCase):
    def test_distance_units_signed_values_ties_zero_improvement(self):
        s=np.array([3,1,2],np.float32)
        t=np.array([[1,1,3],[2,1,4],[4,1,5],[5,1,6],[6,1,7],[7,1,8],[8,1,9],[9,1,10]],np.float32)
        v=np.array([.5,1,2],np.float32)
        p=pr.native(s,t,v,np.float32(0),np.float32(1))
        np.testing.assert_array_equal(p['union'],[1,1,2])
        np.testing.assert_array_equal(p['effective_reduction'],[True,False,False])
        np.testing.assert_array_equal(p['count_better_source'],[2,0,0])
        self.assertEqual(p['count_tie_source'][1],8)
        self.assertAlmostEqual(p['effective_support_count'][0],9/5)
        self.assertTrue(np.isnan(p['effective_support_count'][1:]).all())
        self.assertGreater(p['source_advantage'][0],0);self.assertLess(p['source_advantage'][2],0)
        self.assertEqual(p['loo_1'][0],1.)
        self.assertEqual(p['loo_2'][0],0.)
        self.assertEqual(p['delta'][0],.5)

    def test_all_supports_tied_and_near_source_tie(self):
        s=np.array([2,1],np.float32);t=np.ones((8,2),np.float32)
        t[0,1]=np.float32(1-5e-7)
        p=pr.native(s,t,np.array([.5,.5],np.float32),0,1)
        self.assertAlmostEqual(p['effective_support_count'][0],8.)
        self.assertEqual(p['count_better_source'][1],0)
        self.assertEqual(p['count_tie_source'][1],8)
        self.assertEqual(p['loo_max'][0],0.)

    def test_support_order_is_preserved_and_changes_loo_coordinates(self):
        s=np.array([9],np.float32);t=np.arange(1,9,dtype=np.float32)[:,None]
        a=pr.native(s,t,np.array([.5],np.float32),0,1)
        perm=np.array([1,0,2,3,4,5,6,7]);b=pr.native(s,t[perm],np.array([.5],np.float32),0,1)
        for k in range(8):self.assertEqual(b[f'target_{k+1}'][0],a[f'target_{perm[k]+1}'][0])
        self.assertEqual(a['loo_1'][0],b['loo_2'][0])
        for k in (2,4,8):np.testing.assert_array_equal(a[f'gap{k}'],b[f'gap{k}'])

    def test_both_compact_orders_map_to_physical_profiles(self):
        raw,meta,segments=fixture();n=len(raw['positions'])
        for data in segments.values():
            data['alpha']=np.linspace(0,1,n,dtype=np.float32);data['endpoint_length']=np.ones(n,np.float32)
        ctx=dict(medians=np.zeros(4,np.float32),scales=np.ones(4,np.float32))
        u,s=ip.original_distances(raw,meta,segments);expected=pr.layer_profiles(ctx,raw,meta,segments,u,s)
        rng=np.random.default_rng(42);p=rng.permutation(n);q=rng.permutation(n)
        changed={k:v[...,p] for k,v in raw.items()};seg={l:{k:v[...,q] for k,v in a.items()} for l,a in segments.items()}
        u,s=ip.original_distances(changed,meta,seg);actual=pr.layer_profiles(ctx,changed,meta,seg,u,s)
        for a,b in zip(expected,actual):
            for key in (*pr.FEATURES,'alpha','endpoint_support_1','endpoint_index_2'):
                np.testing.assert_array_equal(np.asarray(a[key])[p],b[key])

    def test_high_response_ties_physical_order_and_ceil(self):
        x=np.ones((10,11),np.float32)
        np.testing.assert_array_equal(pr.high_response(x),[0,1])
        x.flat[73]=2;np.testing.assert_array_equal(pr.high_response(x),[73,0])

    def test_b_and_delta_match_original_CPU_MAD_and_distance_audit(self):
        from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.analysis import mechanism_rows
        from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.gates import BRANCHES
        raw,meta,segments=fixture();u,v=ip.original_distances(raw,meta,segments)
        med=np.array([.1,.2,.3,.4],np.float32);scales=np.array([.2,.4,.6,.8],np.float32)
        z,_,_=pr.reconstruct(u,raw['positions'],(5,5),(5,5),med,scales)
        profile=pr.native(raw['source_distance'][0],raw['target_distance'][:,0],v[0],med[0],scales[0])
        np.testing.assert_array_equal(profile['b'],z[0].ravel()[raw['positions']])
        branches=dict(zip(BRANCHES,(u,v,v,v)))
        prediction=dict(pair_gate=np.zeros((4,5,5),bool),geom_gate=np.zeros((4,5,5),bool))
        audit=mechanism_rows(dict(sample_id='x',label=0),np.zeros((5,5),bool),branches,
                             raw['positions'],(5,5),scales,prediction)
        self.assertEqual(profile['delta'].mean(),audit[0]['R_value'])

    def test_min_must_precede_interpolation(self):
        s=np.array([[0,2]],np.float32);t=np.array([[2,0]],np.float32)
        before=bilinear(np.minimum(s,t),(1,3));after=np.minimum(bilinear(s,(1,3)),bilinear(t,(1,3)))
        self.assertEqual(before[0,1],0);self.assertEqual(after[0,1],1)

    def test_micro_gt_and_empty_ring_are_not_dropped(self):
        gt=np.zeros((16,16),bool);gt[0,0]=True
        labels,desc=sp.components(gt)
        self.assertEqual(len(desc),1)
        inside,ring=sp.instance_regions(labels==1,gt,(4,4))
        self.assertFalse(inside.any());self.assertTrue(pr.adaptive_max(gt,(4,4)).any())
        full=np.ones((16,16),bool);inside,ring=sp.instance_regions(full,full,(4,4))
        self.assertTrue(inside.all());self.assertFalse(ring.any())

    def test_all_nonzero_stencil_contributors_and_NA_mass(self):
        stencil=sp.interpolation_stencil((3,4),(3,3),(8,8))
        self.assertEqual(len(stencil),4);self.assertAlmostEqual(sum(w for _,_,w in stencil),1)
        profile={k:np.arange(9,dtype=float) for k in pr.FEATURES}
        profile['effective_support_count'][:]=np.nan
        offsets=[y*3+x for y,x,_ in stencil];weights=[w for _,_,w in stencil]
        value=pr.summary_at_stencil(profile,offsets,weights)
        self.assertIsNone(value['effective_support_count'])
        self.assertEqual(value['effective_support_count_valid_stencil_mass'],0)
        self.assertAlmostEqual(value['gap8'],np.dot(np.arange(9)[offsets],weights))


class ConditionalTests(unittest.TestCase):
    def test_image_balance_and_cluster_bootstrap(self):
        ids=np.array([0]*100+[1]+[2]*3+[3]);y=np.array([0]*101+[1]*4)
        w=co.ImageWeights(ids,y);base=w.weights()
        for i in range(4):self.assertAlmostEqual(base[ids==i].sum(),.5)
        draw=w.draw(np.random.Generator(np.random.PCG64(42)))
        for i in range(4):self.assertEqual(len(np.unique(draw[ids==i])),1)
        self.assertAlmostEqual(draw[y==0].sum(),1);self.assertAlmostEqual(draw[y==1].sum(),1)
        with self.assertRaises(ValueError):co.ImageWeights([0,0],[0,1])

    def test_weighted_auc_ties_direction_no_pair_matrix(self):
        stat=co.Statistic([1,2,2,3],[0,0,1,1]);r=stat.evaluate(np.ones(4)*.5)
        self.assertEqual(r['diagnostic_AUROC'],.875)
        reverse=co.Statistic([-1,-2,-2,-3],[0,0,1,1]).evaluate(np.ones(4)*.5)
        self.assertEqual(reverse['diagnostic_AUROC'],.125)

    def test_control_only_apparent_difference_disappears_after_matching(self):
        data=constructed_data()
        _,rows,coverage,bins,_=co.describe(data,7,bootstrap_count=8)
        a=next(r for r in rows if r['mode']=='uncontrolled_effective' and r['feature']=='gap8')
        b=next(r for r in rows if r['mode']=='matched_b_delta' and r['feature']=='gap8')
        self.assertGreater(abs(a['mean_difference_GT_minus_normal']),.1)
        self.assertAlmostEqual(b['mean_difference_GT_minus_normal'],0,places=12)
        self.assertAlmostEqual(b['diagnostic_AUROC'],.5,places=12)
        self.assertEqual(bins['delta_edges'],[1.])
        self.assertTrue(any(r['row_kind']=='image_weight' for r in coverage))

    def test_no_overlap_and_no_reduction_explicit_status(self):
        data=constructed_data();data['b']=data['group'].astype(float)*100
        # Force a frozen matching grid with disjoint coverage directly.
        y=np.array([0]*5+[1]*5);base=np.ones(10)/5
        m=co.Match(np.arange(10,dtype=float),np.arange(10,dtype=float),y,base)
        w,mass,common=m.weights(base)
        self.assertEqual(common.sum(),0);self.assertEqual(w.sum(),0)
        self.assertIsNone(co.Statistic(np.arange(10),y).evaluate(w)['diagnostic_AUROC'])
        data['effective_reduction'][:]=False
        _,rows,coverage,_,_=co.describe(data,7,bootstrap_count=3)
        self.assertTrue(all(r['status']=='one_group_has_no_effective_reduction' for r in rows))
        self.assertTrue(all(r['effective_patches']==0 for r in coverage))

    def test_missing_feature_mass_and_reproducible_parallel_draws(self):
        data=constructed_data();data['effective_support_count'][data['group']==0]=np.nan
        def work(_):return co.describe(data,15,bootstrap_count=5)[1]
        a=work(0)
        with ThreadPoolExecutor(2) as pool:b=list(pool.map(work,range(2)))
        self.assertEqual(a,b[0]);self.assertEqual(a,b[1])
        missing=[r for r in a if r['feature']=='effective_support_count']
        self.assertTrue(all(r['diagnostic_AUROC'] is None for r in missing))
        self.assertTrue(all(r['group0_valid_mass']==0 for r in missing))

    def test_default_production_200_image_replicates(self):
        data=constructed_data();rows=co.describe(data,23)[1]
        self.assertTrue(all(r['bootstrap_replicates']==200 for r in rows))
        self.assertTrue(all(r['mean_difference_GT_minus_normal_valid_bootstraps']<=200 for r in rows))


class InputAndIntegration(unittest.TestCase):
    def test_small_output_report_and_CDF_CPU_integration(self):
        import os
        from DINOv3.MADEqual.aebad_s_support_distance_diagnostic import figures
        _,comparisons,coverage,_,plot=co.describe(constructed_data(),7,bootstrap_count=3)
        with tempfile.TemporaryDirectory(dir=ROOT/'downloads') as tmp:
            out=Path(tmp);config=out/'matplotlib';config.mkdir()
            with mock.patch.dict(os.environ,{'MPLCONFIGDIR':str(config)}):
                import matplotlib
                matplotlib.use('Agg')
                figures.cdfs({7:plot},out)
            an.make_report(out,comparisons,coverage,[],[])
            self.assertTrue((out/'objective_report.md').is_file())
            self.assertEqual(len(list(out.glob('*.png'))),2)
            import gzip
            with an.GzipRows(out/'small.csv.gz') as writer:
                writer.write(dict(effective_support_count=np.nan,status='no_positive_target_improvement'))
            with gzip.open(out/'small.csv.gz','rt',encoding='utf-8') as f:
                self.assertIn(',no_positive_target_improvement',f.read())
            self.assertNotIn('torch',sys.modules)

    def test_archive_checksum_rejects_modified_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'downloads') as tmp:
            path=Path(tmp)/'tiny.tar';blob=b'original'
            with tarfile.open(path,'w') as t:
                info=tarfile.TarInfo('data/a');info.size=len(blob);t.addfile(info,io.BytesIO(blob))
            import hashlib
            with ip.Archive(path) as a:
                self.assertEqual(a.bytes('data/a',hashlib.sha256(blob).hexdigest()),blob)
                with self.assertRaises(ValueError):a.bytes('data/a','0'*64)

    def test_missing_preflight_no_torch_output_or_model_import(self):
        script='''
import importlib.abc,sys,tempfile
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torchvision','industrial','dinov3'):raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic.runner import parser,preflight
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic import analysis,spatial,figures
with tempfile.TemporaryDirectory(dir='downloads') as tmp:
 args=parser().parse_args(['--raw-archive',tmp+'/missing','--output-dir',tmp+'/not_created'])
 r=preflight(args)
 assert r['status']=='blocked' and not Path(args.output_dir).exists()
 assert 'torch' not in sys.modules
print('CPU_ONLY_OK')
'''
        r=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertIn('CPU_ONLY_OK',r.stdout)

    def test_real_one_image_profile_margin_and_support_identity_only(self):
        paths=ip.defaults()
        if not all(paths[k].exists() for k in ('raw_archive','review_archive','formal_dir','data')):
            self.skipTest('optional local frozen packages absent')
        ctx=ip.context(paths);sample=ctx['samples'][0]
        r=an.one_image(ctx,sample,0)
        self.assertFalse(r['failed'],[c for c in r['checks'] if not c['verified']])
        self.assertTrue(r['instances']);self.assertTrue(r['stencils']);self.assertTrue(r['links'])
        self.assertTrue(all(c['verified'] for c in r['checks']))
        for inst in r['instances']:
            if inst['response_status']=='available':
                self.assertAlmostEqual(inst['fixed_margin_change'],inst['ring_response_reduction']-inst['inside_response_reduction'],places=10)
        # Use the actual production metadata validator; only archive transport is stubbed.
        with ip.Archive(paths['raw_archive'],ctx['raw_index']) as arc:
            rec=ctx['captures'][sample['sample_id']]
            raw,meta=arc.npz(ip.CROOT+'/'+rec['file'],rec['sha256'])
        meta['supports']=list(reversed(meta['supports']))
        class Transport:
            def npz(self,*args):return raw,meta
        with self.assertRaises(ValueError):ip.load_image(ctx,Transport(),sample)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
