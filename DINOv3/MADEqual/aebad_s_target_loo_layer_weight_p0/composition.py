"""Fixed output references only: no new map, inference, or AUPRO job."""
from .protocol import NEW,READOUTS,U,S,COMBINED,SOURCE,TARGET,COMBINATIONS,SOURCE_COMBO,TARGET_COMBO
METRICS=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')


def fixed_combinations(rows,domain,development=False):
    index={}
    for r in rows:
        if r.get('domain',domain)!=domain:raise ValueError('mixed domains in single-domain metrics')
        key=(r['branch'],r['scoring'])
        if key in index:raise ValueError('duplicate metric key')
        index[key]=r
    wanted={(b,sc) for b in (U,S,COMBINED) for sc in ('CH1','CURRENT')}|{(b,sc) for b in NEW for sc in READOUTS}
    if not wanted.issubset(index):raise ValueError('required baseline/weighted reference missing')
    result=[dict(index[k],domain=domain) for k in sorted(wanted)]
    for b,reference in COMBINATIONS.items():
        first=index[reference,READOUTS[0]];second=index[reference,READOUTS[1]]
        if any(first.get(k)!=second.get(k) for k in METRICS[2:]):raise ValueError('two readouts must share localization metrics')
        for sc in ('CH1','CURRENT'):
            image=index[S,sc]
            result.append(dict(first,domain=domain,branch=b,scoring=sc,image_AUROC=image['image_AUROC'],image_AP=image['image_AP'],
                image_reference=S,map_reference=reference,metric_origin='fixed_task_division_reference_only',
                proposal_role='background_development_derived_combination' if development else 'frozen_before_new_domain_evaluation'))
    return result


def comparisons(rows,domain):
    index={(r['branch'],r['scoring']):r for r in rows};specs=[]
    for rule,old in zip(READOUTS,('CH1','CURRENT')):
        specs.extend([(TARGET,rule,SOURCE,rule),(SOURCE,rule,U,old),(TARGET,rule,U,old)])
    for rule in ('CH1','CURRENT'):
        specs.extend([(TARGET_COMBO,rule,COMBINED,rule),(TARGET_COMBO,rule,SOURCE_COMBO,rule),(SOURCE_COMBO,rule,COMBINED,rule)])
    result=[]
    for b,rule,ref,rr in specs:
        if (b,rule) not in index or (ref,rr) not in index:raise ValueError('required comparison reference missing')
        for key in METRICS:
            x,y=index[b,rule].get(key),index[ref,rr].get(key)
            result.append(dict(domain=domain,branch=b,scoring=rule,reference=ref,reference_scoring=rr,metric=key,
                delta_pp=None if x in ('',None) or y in ('',None) else 100*(float(x)-float(y))))
    return result


def image_scores(segment_scores):return {b:dict(segment_scores) for b in COMBINATIONS}
