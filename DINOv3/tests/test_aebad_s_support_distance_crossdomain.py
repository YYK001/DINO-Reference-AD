"""Production CPU adapters on small constructions; no formal domain analysis."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain import inputs as ip,analysis as an,runner as run,reporting,discovery
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic import profiles as bp,conditional as co,spatial as bs
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import reconstruct
from DINOv3.tests.test_aebad_s_support_distance_diagnostic import constructed_data


def packed(raw,meta):
    f=io.BytesIO();np.savez_compressed(f,metadata=np.array(json.dumps(meta)),**raw)
    blob=f.getvalue();return blob,hashlib.sha256(blob).hexdigest()


def fixture(domain='illumination',label=1):
    n=16;pos=np.array([3,8,2,10,9,0,1,4,6,5,7,11,15,13,12,14],np.int64)
    source=np.tile(np.linspace(2,4,n,dtype=np.float32),(4,1))
    target=np.stack([source-.5+k*.1 for k in range(8)]).astype(np.float32)
    segment=np.minimum(source,target.min(0))-.1
    raw=dict(positions=pos,source_distance=source,target_distance=target,segment_distance=segment)
    sample=dict(domain=domain,sample_id='test_same',relative_path=f'test/defect/{domain}/same.png',
                image_sha256='f'*64,image_height=32,image_width=32,label=label)
    gt=np.zeros((32,32),bool)
    if label:gt[9:21,9:21]=True
    f=io.BytesIO();Image.fromarray(gt.astype(np.uint8)*255).save(f,format='PNG');gt_blob=f.getvalue()
    sample['gt_sha256']=hashlib.sha256(gt_blob).hexdigest()
    supports=[dict(support_index=k,relative_path=f'test/good/{domain}/{k}.png',image_sha256=str(k)*64) for k in range(1,9)]
    parent=SimpleNamespace(identity='old_capture_identity_not_new_program_hash',medians=np.zeros(4,np.float32),scales=np.ones(4,np.float32),
        catalog={'domains':{domain:dict(supports=supports,samples=[sample])}},
        state={x:dict(medians=[0]*4,scales=[1]*4) for x in ('peak','local')},scores={},source_scores={},instances={})
    meta=dict(format='same_call_target_support_loo_v1',tool=ip.old.TOOL,identity=parent.identity,domain=domain,
        sample_id=sample['sample_id'],relative_path=sample['relative_path'],image_sha256=sample['image_sha256'],
        supports=supports,grid=[4,4],output_shape=[8,8],distance_unit='euclidean_over_1280_once')
    maps={}
    for branch,values in (('SOURCE',source),('UNION',target.min(0)),('SEGMENT',segment)):
        current,peak,local=statistics(values,pos,(4,4),(8,8),parent.medians,parent.scales)
        if branch=='SOURCE':
            parent.source_scores[domain,'test_same']=current;sample['source_score']=current+2e-5
        else:
            parent.scores[domain,'test_same',branch+'_K8_CH1']=dict(CURRENT_check_score=current,image_score=ch1_score(peak,local,parent.state))
            maps[branch]=reconstruct(values,pos,(4,4),(8,8),parent.medians,parent.scales)[1]['FUSED']
            if branch=='UNION':sample['union_score']=current+2e-5
    labelled,descs=bs.components(gt)
    for desc in descs:
        iid=desc['instance_id'];inside,ring=bs.instance_regions(labelled==iid,gt,(8,8))
        anchors={reg:bs.peak_position(maps['UNION'],mask) for reg,mask in (('instance',inside),('ring',ring))}
        for b in ('UNION','SEGMENT'):
            inst=float(maps[b][inside].max());rg=float(maps[b][ring].max())
            ref=dict(**desc,patch_coverage=int(an.adaptive_max(labelled==iid,(4,4)).sum()),
                instance_area_output_pixels=int(inside.sum()),ring_area_output_pixels=int(ring.sum()),
                instance_max=inst,ring_max=rg,margin=inst-rg,smallest_area_quartile='True',
                fixed_union_instance_response=float(maps[b][anchors['instance']]),
                fixed_union_ring_response=float(maps[b][anchors['ring']]),
                fixed_union_margin=float(maps[b][anchors['instance']])-float(maps[b][anchors['ring']]))
            for reg,xy in anchors.items():ref.update({f'union_anchor_{reg}_y':str(xy[0]),f'union_anchor_{reg}_x':str(xy[1])})
            parent.instances[domain,'test_same',iid,b+'_K8_CH1']=ref
    blob,h=packed(raw,meta);record=dict(domain=domain,sample_id='test_same',file=f'raw_cache/{domain}/test_same.npz',sha256=h)
    parent.records={(domain,'test_same'):record}
    return parent,sample,raw,meta,blob,record,gt_blob


class RawAdmission(unittest.TestCase):
    def setUp(self):self.p,self.s,self.a,self.m,self.blob,self.r,self.gt=fixture()
    def test_original_identity_and_support_order_without_current_code_identity(self):
        a,m=ip.decode(self.blob,self.p,self.s,self.r)
        self.assertEqual(m['identity'],self.p.identity)
        np.testing.assert_array_equal(a['target_distance'],self.a['target_distance'])
    def test_modified_capture_cannot_be_reenrolled(self):
        a=copy.deepcopy(self.a);a['target_distance'][7,3,2]+=.01
        blob,_=packed(a,self.m)
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):ip.decode(blob,self.p,self.s,self.r)
    def test_union_only_is_not_eight_support_capture(self):
        a=dict(self.a);a['union_distance']=a.pop('target_distance').min(0)
        blob,h=packed(a,self.m)
        with self.assertRaisesRegex(ValueError,'target_distance'):ip.decode(blob,self.p,self.s,dict(self.r,sha256=h))
    def test_support_permutation_rejected(self):
        m=copy.deepcopy(self.m);m['supports'][0],m['supports'][1]=m['supports'][1],m['supports'][0]
        blob,h=packed(self.a,m)
        with self.assertRaisesRegex(ValueError,'support identity/order'):ip.decode(blob,self.p,self.s,dict(self.r,sha256=h))
    def test_wrong_domain_layer_unit_and_identity(self):
        for field,value in [('domain','view'),('layers',[31,23,15,7]),('distance_unit','raw_euclidean'),('identity','new_program_hash')]:
            with self.subTest(field=field):
                m=dict(self.m,**{field:value});blob,h=packed(self.a,m)
                with self.assertRaises(ValueError):ip.decode(blob,self.p,self.s,dict(self.r,sha256=h))
    def test_positions_must_be_physical_bijection(self):
        a=copy.deepcopy(self.a);a['positions'][0]=a['positions'][1];blob,h=packed(a,self.m)
        with self.assertRaisesRegex(ValueError,'positions'):ip.decode(blob,self.p,self.s,dict(self.r,sha256=h))
    def test_exact_shape_four_layers_eight_supports(self):
        for key,value in [('target_distance',self.a['target_distance'][:7]),('source_distance',self.a['source_distance'].astype(float))]:
            a=dict(self.a,**{key:value});blob,h=packed(a,self.m)
            with self.assertRaisesRegex(ValueError,'distances'):ip.decode(blob,self.p,self.s,dict(self.r,sha256=h))


class CPUAdapter(unittest.TestCase):
    def execute(self,label=1,derive=False,domain='illumination'):
        p,s,a,m,blob,r,gt=fixture(domain,label)
        if derive:
            for ref in p.instances.values():
                for field in list(ref):
                    if field.startswith('union_anchor_'):del ref[field]
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as folder:
            root=Path(folder);path=root/s['relative_path'].replace('test/','ground_truth/',1)
            path.parent.mkdir(parents=True);path.write_bytes(gt)
            result=an.one_image(p,s,r,blob,root,0)
        return result,(p,s,a,m,blob,r)
    def test_actual_map_score_spatial_adapter_on_small_capture(self):
        result,_=self.execute()
        self.assertFalse(result['failed']);self.assertEqual(len(result['instances']),1)
        row=result['instances'][0];self.assertEqual(row['UNION_instance_coordinate_origin'],'parent_CPU_saved_coordinate')
        self.assertTrue(all(c['verified'] for c in result['checks']))
        self.assertAlmostEqual(row['fixed_margin_change'],row['decomposition_ring_minus_inside'])
        self.assertTrue(all(r['domain']=='illumination' for r in result['selected']))
    def test_derive_coordinates_is_explicit_and_same_scalars(self):
        a,_=self.execute();b,_=self.execute(derive=True)
        self.assertFalse(b['failed']);self.assertTrue(b['instances'][0]['UNION_instance_coordinate_origin'].startswith('derived_this_analysis'))
        self.assertEqual(a['instances'][0]['UNION_fixed_margin'],b['instances'][0]['UNION_fixed_margin'])
    def test_strict_new_check_separate_from_old_compatibility(self):
        r,_=self.execute();checks=r['checks']
        earlier=[x for x in checks if x.get('historical_score_check')]
        self.assertEqual(len(earlier),2);self.assertTrue(all(x['verified'] and not x['strict_verified'] for x in earlier))
        self.assertTrue(all(x['atol']<=1e-5 for x in checks if not x.get('historical_score_check')))
    def test_same_cache_score_change_stops(self):
        p,s,a,m,blob,r,gt=fixture();p.source_scores[s['domain'],s['sample_id']]+=.001
        with mock.patch.object(ip,'Data',side_effect=AssertionError('must fail before GT read')):
            out=an.one_image(p,s,r,blob,Path('unused'),0)
        self.assertTrue(out['failed'])
    def test_normal_top_percent_and_same_physical_positions_all_layers(self):
        r,(p,s,a,m,*_)=self.execute(label=0)
        self.assertFalse(r['failed']);self.assertEqual(len(r['selected']),4)
        self.assertEqual(len({x['physical_patch'] for x in r['selected']}),1)
        off=r['selected'][0]['compact_offset']
        expected=bp.native(a['source_distance'][0],a['target_distance'][:,0],a['segment_distance'][0],p.medians[0],p.scales[0])
        self.assertEqual(r['selected'][0]['gap8'],expected['gap8'][off])
        self.assertEqual(r['selected'][0]['physical_patch'],a['positions'][off])
    def test_cross_domain_same_id_parallel_never_overwrites(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda d:self.execute(domain=d)[0],ip.DOMAINS))
        keyed={(r['domain'],r['sample_id']):r for r in results}
        self.assertEqual(len(keyed),2);self.assertTrue(all(not r['failed'] for r in keyed.values()))
    def test_quartile_export_not_background_three(self):
        r,_=self.execute();rows=[an.scoped(x,'illumination') for x in r['instances']]
        self.assertNotIn('small_original_three',rows[0]);self.assertTrue(rows[0]['smallest_area_quartile'])
        summary=an.summarize_instances(rows,'illumination')
        self.assertTrue(all(x['small_definition']=='original_domain_smallest_area_quartile' for x in summary))


class Statistics(unittest.TestCase):
    def test_exact_background_kernel_reuse_and_independent_domain_edges(self):
        self.assertIs(run.kernel.describe,co.describe);self.assertIs(an.bp.native,bp.native)
        a=constructed_data();b={k:v.copy() for k,v in a.items()};b['b']+=100;b['delta']*=3
        ra=run.kernel.describe(a,7,bootstrap_count=5);rb=run.kernel.describe(b,7,bootstrap_count=5)
        np.testing.assert_allclose(np.array(ra[3]['b_edges'])+100,rb[3]['b_edges'])
        np.testing.assert_allclose(np.array(ra[3]['delta_edges'])*3,rb[3]['delta_edges'])
        self.assertEqual(co.BOOTSTRAPS,200)
        self.assertEqual(run.protocol('illumination')['images'],265);self.assertEqual(run.protocol('view')['images'],364)
    def test_NA_and_no_common_support_retained(self):
        a={k:np.arange(10,dtype=float) for k in bp.FEATURES}
        a.update(image_id=np.arange(10),group=np.array([0]*5+[1]*5),b=np.arange(10,dtype=float),
                 delta=np.arange(10,dtype=float),effective_reduction=np.ones(10,bool))
        a['gap8'][:]=np.nan
        result=co.describe(a,31,bootstrap_count=3)
        rows=[r for r in result[1] if r.get('mode')=='matched_b_delta']
        self.assertEqual(len(rows),len(bp.FEATURES))
        self.assertTrue(all(r['status']=='insufficient_common_support_or_feature_coverage' for r in rows))
        self.assertTrue(all(r['diagnostic_AUROC'] is None and r['diagnostic_AUROC_valid_bootstraps']==0 for r in rows))
        self.assertEqual(len([c for c in result[2] if c['row_kind']=='matched_group']),2)
    def test_missing_effective_group_preserves_reason(self):
        a=constructed_data();a['effective_reduction']=a['group']==0
        result=co.describe(a,15,bootstrap_count=3)
        self.assertEqual(len(result[1]),len(bp.FEATURES))
        self.assertTrue(all(r['status']=='one_group_has_no_effective_reduction' for r in result[1]))
    def test_AUC_direction_not_flipped(self):
        a=constructed_data()
        for k in bp.FEATURES:a[k]=-a['group'].astype(float)
        rows=co.describe(a,23,bootstrap_count=3)[1]
        self.assertTrue(all(r['diagnostic_AUROC']==0 for r in rows))


class DiscoveryAndSummary(unittest.TestCase):
    def test_small_synthetic_runner_writes_domain_results_and_pending_reference(self):
        # Only one 4x4-grid anomaly per domain; replace admission counts for this
        # fixture, not production mathematics. No real image/domain experiment.
        fixtures=[fixture(d) for d in ip.DOMAINS];parent=fixtures[0][0]
        for name in ('scores','source_scores','instances','records'):
            getattr(parent,name).update(getattr(fixtures[1][0],name))
        parent.catalog['domains'].update(fixtures[1][0].catalog['domains'])
        parent.catalog['formal_sha256']={'source_mad_state.npz':'test-only'}
        parent.historical_audit={'fixture':'synthetic only'}
        parent.required=lambda:[dict(domain=d,sample_id='test_same',relative_path=f'raw_cache/{d}/test_same.npz') for d in ip.DOMAINS]
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);out=root/'analysis';out.mkdir();data=root/'data';found=[]
            for p,s,a,m,blob,record,gt in fixtures:
                gt_path=data/s['relative_path'].replace('test/','ground_truth/',1);gt_path.parent.mkdir(parents=True);gt_path.write_bytes(gt)
                raw=root/record['file'];raw.parent.mkdir(parents=True);raw.write_bytes(blob)
                found.append(dict(record,storage='file',path=str(raw),member=''))
            args=SimpleNamespace(background=root/'absent_reference',data=data,cpu_workers=2)
            with mock.patch.dict(ip.COUNTS,{d:(0,1) for d in ip.DOMAINS}),mock.patch.object(ip.base,'available_ram',return_value=3*2**30):
                run.analyze(args,out,parent,found)
            self.assertEqual(ip.old.read_json(out/'status.json')['status'],'partial_evidence')
            for d in ip.DOMAINS:
                status=ip.old.read_json(out/d/'status.json')
                self.assertEqual(status['status'],'diagnostic_complete');self.assertFalse(status['torch_imported'])
                self.assertEqual(status['numerical_failed_checks'],0)
                self.assertEqual(len(list((out/d/'figures').glob('*.png'))),3)
                self.assertEqual(len(reporting.csv_file(out/d/'conditional_comparisons.csv')),64)

    def test_dirs_and_archive_keep_domain_and_do_not_extract(self):
        req=[dict(relative_path=f'raw_cache/{d}/test_same.npz',domain=d,sample_id='test_same',sha256='f'*64) for d in ip.DOMAINS]
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);archive=root/'aebad_raw.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                for r in req:
                    data=r['domain'].encode();m=tarfile.TarInfo('cache/'+r['relative_path']);m.size=len(data);t.addfile(m,io.BytesIO(data))
            found,missing,index=discovery.discover(req,[root])
            self.assertFalse(missing);self.assertEqual(len(found),2);self.assertFalse((root/'cache').exists())
            blobs=list(discovery.blobs(found));self.assertEqual({r['domain']:v.decode() for r,v in blobs},{d:d for d in ip.DOMAINS})
            again=discovery.discover(req,[root],previous_index=index)
            self.assertTrue(again[2]['archives'][0]['index_reused'])
    def test_valid_copy_can_be_used_while_modified_copy_rejected(self):
        p,s,a,m,blob,r,gt=fixture();required=dict(relative_path=r['file'],domain=s['domain'],sample_id=s['sample_id'],sha256=r['sha256'])
        p.catalog['domains']['view']=dict(samples=[],supports=[])
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f)
            for name,value in [('bad',b'changed'),('good',blob)]:
                file=root/name/r['file'];file.parent.mkdir(parents=True);file.write_bytes(value)
            found,missing,_=discovery.discover([required],[root]);valid,invalid=run.verified_sources(found,p)
            self.assertFalse(missing);self.assertFalse(invalid);self.assertEqual(len(valid),1);self.assertIn('good',valid[0]['path'])
            (root/'good'/r['file']).unlink();found,_,_=discovery.discover([required],[root]);valid,invalid=run.verified_sources(found,p)
            self.assertFalse(valid);self.assertEqual(invalid[0]['status'],'untrusted_or_incomplete_capture')
    def test_summary_does_not_complete_from_background_alone(self):
        background=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_s_support_distance_diagnostic_20261021_054307'
        if not background.exists():self.skipTest('local discovery reference absent')
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);rows=reporting.summarize(background,{},root)
            status=ip.old.read_json(root/'status.json')
            self.assertEqual(len(rows),36);self.assertEqual(status['status'],'partial_evidence')
            self.assertEqual(status['pending'],list(ip.DOMAINS));self.assertFalse(status['aggregate_AUROC_computed'])
            self.assertEqual(sum(r['status']=='pending' for r in rows),24)
    def test_incomplete_result_cannot_be_reported_complete(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);ip.write_json(root/'status.json',dict(status='running',images=265))
            with self.assertRaisesRegex(ValueError,'incomplete'):reporting.verified_result(root,'illumination')
    def test_summary_signed_margin_association_and_direction(self):
        result=dict(comparisons=[],coverage=[],instance_summary=[])
        for l in ip.LAYERS:
            for k in ip.MAIN_FEATURES:
                result['comparisons'].append(dict(layer=l,mode='matched_b_delta',feature=k,status='available',
                    diagnostic_AUROC='.3',diagnostic_AUROC_CI_low='.2',diagnostic_AUROC_CI_high='.4'))
                for mode in ('own','fixed_UNION'):
                    for group,value in [('decreased',-3),('not_decreased',1)]:
                        result['instance_summary'].append(dict(layer=l,mode=mode,feature=k,margin_group=group,image_balanced_mean=str(value),valid_images='2'))
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f,mock.patch.object(reporting,'verified_result',return_value=result):
            root=Path(f);(root/'status.json').write_text('{}')
            out=root/'out';out.mkdir();rows=reporting.summarize(root,{d:root for d in ip.DOMAINS},out)
            self.assertTrue(all(r['same_direction_as_background'] for r in rows))
            self.assertTrue(all(r['fixed_margin_group_mean_difference']==-4 for r in rows))
            self.assertEqual(ip.old.read_json(out/'status.json')['status'],'cross_domain_summary_complete')


class RealMetadataAndImportBoundary(unittest.TestCase):
    def test_real_parent_fixed_629_population_and_historical_audit(self):
        path=ip.ROOT/'downloads'/ip.SUMMARY_NAME
        if not path.exists():self.skipTest('approved parent summary absent')
        p=ip.Parent(path);self.assertEqual(len(p.required()),629)
        self.assertEqual([len(p.catalog['domains'][d]['samples']) for d in ip.DOMAINS],[265,364])
        self.assertEqual(p.historical_audit['parent_capture']['strict_failed_checks'],24)
        self.assertEqual(p.historical_audit['parent_capture']['rejected_checks'],0)
        self.assertEqual(p.historical_audit['parent_CPU']['strict_failed_checks'],19)
    def test_default_preflight_denies_torch_and_model_imports(self):
        # Actual CLI default stage + real parent metadata; an empty search root is
        # deliberate so this test is lightweight and cannot execute diagnostics.
        if not (ip.ROOT/'downloads'/ip.SUMMARY_NAME).exists():self.skipTest('parent summary absent')
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);(root/'empty').mkdir()
            code='''
import sys,importlib.abc
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('torch','torchvision','industrial','dinov3'):
   raise AssertionError('forbidden import '+fullname)
sys.meta_path.insert(0,Deny())
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain.runner import main
rc=main(sys.argv[1:]);assert rc==2
assert 'torch' not in sys.modules
'''
            p=subprocess.run([sys.executable,'-B','-c',code,'--search-root',str(root/'empty'),'--data',str(root/'missing_data'),
                '--output-dir',str(root/'inventory')],cwd=ip.ROOT,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            status=ip.old.read_json(root/'inventory/status.json')
            self.assertEqual(status['missing_cache_files'],629);self.assertFalse(status['formal_analysis_executed'])
            self.assertEqual(len(reporting.csv_file(root/'inventory/required_cache_files.csv')),629)


if __name__=='__main__':unittest.main()
