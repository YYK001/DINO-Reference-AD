"""Reuse original stable_support_key on FULL confirmed-normal candidate pools."""
from pathlib import Path
from types import SimpleNamespace
from DINOv3.MADEqual.aebad_s_k8_support_sensitivity_p0.selection import original_key
from .protocol import DOMAINS,SEEDS


def select(parents,history,root):
    domains={}
    for d in DOMAINS:
        p=parents[d];full=sorted(p.samples+p.supports,key=lambda r:r['sample_id'])
        if len({r['sample_id'] for r in full})!=len(full) or len({r['relative_path'] for r in full})!=len(full):raise ValueError('duplicate full-test image identity')
        t42=[dict(r) for r in p.supports];s42={r['relative_path'] for r in t42}
        if [r['sample_id'] for r in full if r['relative_path'] not in s42]!=[r['sample_id'] for r in p.samples]:raise ValueError('cannot restore original full-test order')
        normals=[r for r in full if r['label']==0]
        t43=sorted(normals,key=lambda r:original_key()(SimpleNamespace(path=Path(root)/r['relative_path']),dataset_root=Path(root),domain=d,seed=43))[:8]
        groups={'42':t42,'43':t43}
        for seed in SEEDS:
            if len(groups[str(seed)])!=8 or any(r['label']!=0 for r in groups[str(seed)]):raise ValueError('K8 confirmed normals required')
            old=sorted([r for r in history['records'] if r['domain']==d and int(r['support_seed'])==seed],key=lambda r:int(r['support_index']))
            if [(r['relative_path'],r['sample_id'],r['image_sha256']) for r in old]!=[(r['relative_path'],r['sample_id'],r['image_sha256']) for r in groups[str(seed)]]:raise ValueError('historical support selection/data identity differs')
        s43={r['relative_path'] for r in t43};excluded=s42|s43;common=[r for r in full if r['relative_path'] not in excluded]
        seed44={r['relative_path'] for r in history['records'] if r['domain']==d and int(r['support_seed'])==44}
        domains[d]=dict(supports=groups,common=common,excluded=[dict(image=r,seeds=[s for s in SEEDS if any(x['relative_path']==r['relative_path'] for x in groups[str(s)])]) for r in full if r['relative_path'] in excluded],
            overlap=len(s42&s43),normal_images=sum(r['label']==0 for r in common),anomaly_images=sum(r['label']==1 for r in common),
            full_images=len(full),common_images=len(common),seed44_normals_retained=sorted(seed44-excluded),historical_supports_match=True)
    return dict(domains=domains,total_common_images=sum(x['common_images'] for x in domains.values()),support_seeds=list(SEEDS),Source_seed=42)
