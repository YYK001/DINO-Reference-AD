"""GT is consumed only after global no-GT crossed response freeze."""
import numpy as np
from contextlib import ExitStack
from . import inputs as ip,numerics as nm,prediction as pred,evaluation as ev
from .inputs import st,GTData
from .protocol import *
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics


def area_plot(path,rows):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,kind in zip(axes,('fixed','own')):
        for new,old in COMPARISONS:
            selected=[r for r in rows if r['comparison']==new+'_minus_'+old and r[kind+'_margin_change'] is not None]
            ax.scatter([r['area_original_pixels'] for r in selected],[r[kind+'_margin_change'] for r in selected],s=8,alpha=.4,label=new+' - '+old)
        if any(r[kind+'_margin_change'] is not None for r in rows):ax.set_xscale('log')
        else:ax.text(.5,.5,'No computable instances (NA retained)',ha='center',transform=ax.transAxes)
        ax.axhline(0,color='gray',linewidth=.7);ax.set_xlabel('Original defect area (pixels, log axis when available)');ax.set_ylabel(kind+' margin change');ax.legend(fontsize=7)
    fig.savefig(path,dpi=110);plt.close(fig)


def analyze(a):
    frozen=pred.global_freeze(a);p=ip.parent(a);done=[]
    with ExitStack() as stack:
        data=None
        for d in a.domains:
            for m in a.memory_seeds:
                if ev.completed(a,d,m,'analysis'):done.append(dict(domain=d,memory_seed=m,status='analysis_reused'));continue
                if data is None:data=stack.enter_context(GTData(a.data))
                sig=ip.signature(a,d,m);out=ip.unit(a,d,m)/'analysis';out.mkdir(parents=True,exist_ok=True);samples=p.samples(d);weights={w:p.weights[d,w] for w in WEIGHTS}
                scores=[];stats=[];tails=[];instances=[];checks=[]
                for i,row in enumerate(samples,1):
                    images,sc,stat,z,pos,ref=pred.load_maps(a,d,m,row);record=st.sealed_json(ip.record_path(a,d,m,row))
                    for w in WEIGHTS:
                        if ev.backend.shared.array_sha(images[w])!=record['map_hashes'][w]:raise ValueError('analysis map differs from frozen cross response')
                    scores.append(record['scores']);stats.append(record['layer_statistics'])
                    prefix=dict(domain=d,memory_seed=m,sample_id=row['sample_id'])
                    if not row['label']:tails.extend(dict(**prefix,**r) for r in nm.normal_tail(images))
                    else:
                        original,mask=ev.mask_record(data,row,ref['output_shape'])
                        if ev.backend.shared.array_sha(mask)!=p.maps[d,m][row['sample_id'],SOURCE]['mask_uint8_sha256']:raise ValueError('diagnosis GT differs from original')
                        lm=nm.per_layer_maps(z,pos,ref['grid'],ref['output_shape']);rs,cs=nm.instance_changes(images,lm,original,weights)
                        instances.extend(dict(**prefix,**r) for r in rs);checks.extend(dict(**prefix,**r) for r in cs)
                    if i%25==0 or i==len(samples):print(f'[cross separability CPU] {d}/Z{m} {i}/{len(samples)}',flush=True)
                    pred.budget(a,i,len(samples))
                stats=np.asarray(stats);transitions,contrib,compact,cs=nm.pair_analysis(samples,scores,stats,weights);checks.extend(dict(domain=d,memory_seed=m,**r) for r in cs)
                for w in ('WS',diagonal(m)):
                    actual=image_metrics([r['label'] for r in samples],[r[w][READOUTS[0]] for r in scores]);known=next(r for r in p.known_metrics(d,m,w) if r['scoring']==READOUTS[0])
                    for key,value in actual.items():
                        error=abs(value-float(known[key]))
                        if error>SCORE_ATOL:raise ValueError('pair/scalar ranking does not reproduce original diagonal metric')
                        checks.append(dict(domain=d,memory_seed=m,weight_source=w,check='frozen_parent_'+key,max_abs_error=error,atol=SCORE_ATOL,verified=True))
                # Compact records preserve EVERY normal-anomaly pair, actual gaps,
                # all p/q contributions and rounding residuals without millions of CSV rows.
                for comparison,arrays in compact.items():
                    record=st.put(out/'pair_records'/comparison,arrays,dict(**sig,comparison=comparison,sample_ids=[r['sample_id'] for r in samples],pair_order='anomaly_major_normal_minor',exact_ties=True))
                    for r in transitions:
                        if r['comparison']==comparison:r['compact_bundle']='pair_records/'+comparison;r['compact_seal']=record['seal']
                proxy=[];y=np.asarray([r['label'] for r in samples]);normal=stats[y==0]
                for w in WEIGHTS:
                    seed=m if w=='WS' else int(w[1:]);saved=p.settings[d,seed]['weight_rows'];b=SOURCE if w=='WS' else TARGET
                    rows={(int(r['layer']),r['statistic']):r for r in saved if r['branch']==b}
                    for j,l in enumerate(LAYERS):
                        r=dict(domain=d,memory_seed=m,weight_source=w,weight_seed=None if w=='WS' else seed,layer=l,diagnostic_only=w==crossed(m))
                        for k,statistic in enumerate(('u','p','q')):
                            old=rows[l,statistic];r.update({statistic+'_Source_scale':float(old['Source_scale']),statistic+'_Target_LOO_scale':float(old['Target_LOO_scale']),statistic+'_mixed_scale':float(old['mixed_scale']),statistic+'_WS_weight':p.weights[d,'WS'][j][k],statistic+'_weight':weights[w][j][k],statistic+'_weight_change':float(np.float32(weights[w][j][k]))-float(np.float32(weights['WS'][j][k]))})
                            r.update(nm.distribution(normal[:,j,k],statistic+'_evaluation_normal'))
                        for k,name in ((1,'p'),(2,'q')):
                            r[name+'_single_layer_image_AUROC']=image_metrics(y,stats[:,j,k])['image_AUROC']
                            for group in ('all','repaired','damaged'):
                                entry=next((x for x in contrib if x['comparison']==w+'_minus_WS' and x['group']==group and x['layer']==l and x['component']==name),None)
                                r[f'{name}_{group}_pair_gap_contribution_mean']=entry['contribution_mean'] if entry else 0. if w=='WS' and group=='all' else None
                                r[f'{name}_{group}_pairs']=entry['pairs'] if entry else int((y==0).sum()*(y==1).sum()) if w=='WS' and group=='all' else 0
                        reference=('W42' if w=='WS' else w)+'_minus_WS'
                        selected=[x for x in instances if x['comparison']==reference and x['status']=='available']
                        r['valid_margin_instances']=len(selected);r['NA_margin_instances']=sum(x['comparison']==reference and x['status']!='available' for x in instances)
                        r['fixed_margin_layer_contribution_mean']=0. if w=='WS' and selected else float(np.mean([x[f'contribution_L{l}'] for x in selected])) if selected else None
                        r['interpretation']='descriptive_layer_evidence_no_significance_or_layer_selection';proxy.append(r)
                margin_summary=[]
                normal_summary=[]
                for comparison in (x+'_minus_'+y for x,y in COMPARISONS):
                    rs=[r for r in tails if r['comparison']==comparison]
                    normal_summary.append(dict(domain=d,memory_seed=m,comparison=comparison,images=len(rs),aggregation='equal_image',max_single_image_weight=1/len(rs) if rs else None,
                        **{key:float(np.mean([r[key] for r in rs])) if rs else None for key in ('fixed_peak_change','fixed_top1_mean_change','own_peak_change')},peak_moved_images=sum(r['own_peak_moved'] for r in rs),interpretation='normal_high_response_not_FPR'))
                for comparison in (x+'_minus_'+y for x,y in COMPARISONS):
                    rs=[r for r in instances if r['comparison']==comparison]
                    for kind in ('fixed','own'):
                        margin_summary.append(dict(domain=d,memory_seed=m,comparison=comparison,margin_kind=kind,total_instances=len(rs),**{name:sum(r[kind+'_margin_group']==name for r in rs) for name in ('improved','decreased','unchanged','NA')},mean_change=float(np.mean([r[kind+'_margin_change'] for r in rs if r[kind+'_margin_change'] is not None])) if any(r[kind+'_margin_change'] is not None for r in rs) else None,margin_atol=MARGIN_ATOL))
                tables={'image_pair_transitions.csv':transitions,'layer_pair_contributions.csv':[dict(domain=d,memory_seed=m,**r) for r in contrib],
                    'normal_tail_changes.csv':tails,'normal_tail_summary.csv':normal_summary,'defect_instance_changes.csv':instances,'normal_proxy_vs_discrimination.csv':proxy,'verification.csv':checks,'margin_summary.csv':margin_summary}
                for name,rs in tables.items():st.write_csv(out/name,rs)
                area_plot(out/'area_margin.png',instances)
                # Check compact bundle seals on resume as well as the small CSVs.
                bundle_seals={c:st.sealed_json(out/'pair_records'/c/'manifest.json')['seal'] for c in compact}
                done.append(st.sealed_json(out/'status.json',dict(**sig,status='descriptive_analysis_complete',response_freeze=frozen['seal'],files={n:st.sha(out/n) for n in (*tables,'area_margin.png')},compact_pair_bundles=bundle_seals,images=len(samples),GT_used_for_analysis_only=True)))
    return dict(status='requested_analysis_complete',results=done)
