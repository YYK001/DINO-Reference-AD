"""Image-balanced distributions, frozen 5x5 matching and image-cluster intervals."""
from __future__ import annotations
import hashlib
import numpy as np
from .profiles import FEATURES

BOOTSTRAPS=200
BOOTSTRAP_SEED=42
VERSION='support_distance_conditional_v1'


def weighted_quantiles(values,weights,quantiles):
    a,w=np.asarray(values,np.float64),np.asarray(weights,np.float64)
    valid=np.isfinite(a)&np.isfinite(w)&(w>0)
    if not valid.any():return np.full(len(quantiles),np.nan)
    a,w=a[valid],w[valid];order=np.argsort(a,kind='stable');a,w=a[order],w[order]
    c=np.cumsum(w);c/=c[-1]
    return a[np.minimum(np.searchsorted(c,quantiles,side='left'),len(a)-1)]


class ImageWeights:
    def __init__(self,ids,labels):
        ids,y=np.asarray(ids),np.asarray(labels)
        if ids.ndim!=1 or y.shape!=ids.shape or not set(y.tolist()).issubset({0,1}):raise ValueError('binary diagnostic groups required')
        self.ids,self.inverse,self.counts=np.unique(ids,return_inverse=True,return_counts=True)
        self.image_labels=np.empty(len(self.ids),int)
        for j in range(len(self.ids)):
            ys=np.unique(y[self.inverse==j])
            if len(ys)!=1:raise ValueError('one image cannot belong to both diagnostic groups')
            self.image_labels[j]=ys[0]
        self.labels=y
        self.groups=[np.flatnonzero(self.image_labels==g) for g in (0,1)]

    def weights(self,multiplicities=None):
        factor=np.ones(len(self.ids)) if multiplicities is None else np.asarray(multiplicities,float)
        for group in self.groups:
            if len(group):factor[group]/=len(group)
        return factor[self.inverse]/self.counts[self.inverse]

    def draw(self,rng):
        mult=np.zeros(len(self.ids))
        for g in self.groups:
            if len(g):mult+=np.bincount(rng.choice(g,size=len(g),replace=True),minlength=len(self.ids))
        return self.weights(mult)


class Match:
    def __init__(self,b,delta,labels,weights):
        self.labels=np.asarray(labels)
        if any(not np.isfinite(x).all() for x in (b,delta,weights)):raise ValueError('finite matching controls required')
        # Each group has mass 1; pooling gives the two groups equal prior mass.
        self.edges=[np.unique(weighted_quantiles(v,weights,np.linspace(0,1,6))) for v in (b,delta)]
        bins=[]
        for values,edge in zip((b,delta),self.edges):
            bins.append(np.searchsorted(edge[1:-1],values,side='right'))
        self.width=max(1,len(self.edges[1])-1)
        self.cells=bins[0]*self.width+bins[1]
        self.ncells=max(1,len(self.edges[0])-1)*self.width

    def weights(self,base):
        masses=np.asarray([np.bincount(self.cells,weights=base*(self.labels==g),minlength=self.ncells) for g in (0,1)])
        common=np.minimum(masses[0],masses[1]);total=common.sum()
        result=np.zeros(len(base))
        if total>0:
            q=common/total
            for g in (0,1):
                factors=np.divide(q,masses[g],out=np.zeros(self.ncells),where=masses[g]>0)
                selected=self.labels==g
                result[selected]=base[selected]*factors[self.cells[selected]]
        return result,masses,common


class Statistic:
    """Cached sort/tie blocks: O(patches) per bootstrap, no patch pair matrix."""
    def __init__(self,values,labels):
        self.values=np.asarray(values,np.float64);self.labels=np.asarray(labels)
        valid=np.flatnonzero(np.isfinite(self.values))
        self.order=valid[np.argsort(self.values[valid],kind='stable')]
        a=self.values[self.order]
        self.starts=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1] if len(a) else np.array([],int)

    def evaluate(self,weights):
        o=self.order;a=self.values[o];y=self.labels[o];w=np.asarray(weights)[o]
        result={};means=[];vars=[]
        for g in (0,1):
            wg=w*(y==g);mass=wg.sum()
            result[f'group{g}_valid_mass']=float(mass)
            if mass<=0:
                result.update({f'group{g}_{k}':None for k in ('mean','median','p25','p75')})
                means.append(None);vars.append(None);continue
            wg=wg/mass;mean=float(np.dot(wg,a));means.append(mean)
            vars.append(float(np.dot(wg,(a-mean)**2)))
            c=np.cumsum(wg)
            qs=a[np.minimum(np.searchsorted(c,[.25,.5,.75],side='left'),len(a)-1)]
            result.update({f'group{g}_mean':mean,f'group{g}_p25':float(qs[0]),
                           f'group{g}_median':float(qs[1]),f'group{g}_p75':float(qs[2])})
        if None in means:
            result.update(mean_difference_GT_minus_normal=None,standardized_difference=None,diagnostic_AUROC=None)
            return result
        difference=means[1]-means[0];denom=np.sqrt((vars[0]+vars[1])/2)
        result['mean_difference_GT_minus_normal']=difference
        result['standardized_difference']=float(difference/denom) if denom>0 else 0. if difference==0 else None
        wn=w*(y==0);wp=w*(y==1)
        n=np.add.reduceat(wn,self.starts);p=np.add.reduceat(wp,self.starts)
        result['diagnostic_AUROC']=float(np.dot(p,np.cumsum(n)-.5*n)/(wn.sum()*wp.sum()))
        return result


def describe(data,layer,*,bootstrap_count=BOOTSTRAPS,progress=None):
    """Production uses 200; the argument exists only for small numerical tests."""
    y=np.asarray(data['group'],int);ids=np.asarray(data['image_id'])
    eligible=np.asarray(data['effective_reduction'],bool)
    all_weight=ImageWeights(ids,y);base=all_weight.weights()
    summary=[];comparisons=[];coverage=[];bins={}
    for key in FEATURES:
        stat=Statistic(data[key],y)
        summary.append(dict(layer=layer,scope='all_selected',feature=key,**stat.evaluate(base)))
        for name,select in (('effective_reduction',eligible),('no_effective_reduction',~eligible)):
            if select.any():
                weight=ImageWeights(ids[select],y[select]).weights()
                summary.append(dict(layer=layer,scope=name,feature=key,**Statistic(np.asarray(data[key])[select],y[select]).evaluate(weight)))
    selected={k:np.asarray(v)[eligible] for k,v in data.items()}
    sy=selected['group'].astype(int);sids=selected['image_id']
    weights=ImageWeights(sids,sy);sw=weights.weights()
    for g in (0,1):
        take=y==g;keep=take&eligible
        coverage.append(dict(layer=layer,row_kind='selection',group=g,
            selected_patches=int(take.sum()),effective_patches=int(keep.sum()),excluded_patches=int((take&~eligible).sum()),
            images_all=len(np.unique(ids[take])),images_effective=len(np.unique(ids[keep])),
            no_reduction_patch_fraction=float((~eligible[take]).mean()) if take.any() else None,
            no_reduction_image_balanced_mass=float(base[take&~eligible].sum()),
            group_definition='normal_high_response' if g==0 else 'GT_covered'))
    if not len(sy) or set(sy.tolist())!={0,1}:
        for k in FEATURES:
            comparisons.append(dict(layer=layer,feature=k,status='one_group_has_no_effective_reduction'))
        return summary,comparisons,coverage,bins,None
    match=Match(selected['b'],selected['delta'],sy,sw)
    matched,masses,common=match.weights(sw)
    bins=dict(b_edges=match.edges[0].tolist(),delta_edges=match.edges[1].tolist(),
              repeated_edges_merged=True,boundary_weight='equal_group_prior_then_equal_image_then_equal_patch',
              intervals='internal cuts side=right; outer bounds inclusive; constant axis=one cell')
    for c in range(match.ncells):
        for g in (0,1):
            take=(match.cells==c)&(sy==g)
            coverage.append(dict(layer=layer,row_kind='cell',cell=c,b_bin=c//match.width,delta_bin=c%match.width,group=g,
                patches=int(take.sum()),images=len(np.unique(sids[take])),base_probability_mass=float(masses[g,c]),
                common_unnormalized_mass=float(common[c]),matched_probability_mass=float(matched[take].sum())))
    for g in (0,1):
        take=sy==g
        coverage.append(dict(layer=layer,row_kind='matched_group',group=g,
            images_effective=len(np.unique(sids[take])),images_covered=len(np.unique(sids[take&(matched>0)])),
            uncovered_probability_mass=float(sw[take&(matched==0)].sum()),
            effective_overlap_coefficient=float(common.sum()),
            max_matched_image_share=max((float(matched[(sids==i)&take].sum()) for i in np.unique(sids[take])),default=0)))
        for sid in np.unique(sids[take]):
            w=float(matched[(sids==sid)&take].sum())
            coverage.append(dict(layer=layer,row_kind='image_weight',group=g,image_id=int(sid),
                                 base_image_share=float(sw[(sids==sid)&take].sum()),matched_image_share=w))
    stats={k:Statistic(selected[k],sy) for k in FEATURES}
    controls={k:Statistic(selected[k],sy) for k in ('b','delta')}
    interval_keys=('mean_difference_GT_minus_normal','standardized_difference','diagnostic_AUROC',
                   'group0_mean','group1_mean','group0_median','group1_median')
    replicates={(mode,k):[] for mode in ('uncontrolled_effective','matched_b_delta') for k in FEATURES}
    seed=int.from_bytes(hashlib.sha256(f'{VERSION}|{BOOTSTRAP_SEED}|{layer}'.encode()).digest()[:16],'little')
    rng=np.random.Generator(np.random.PCG64(seed))
    for replicate in range(bootstrap_count):
        draw=weights.draw(rng);mw,_,_=match.weights(draw)
        for mode,w in (('uncontrolled_effective',draw),('matched_b_delta',mw)):
            for key,stat in stats.items():replicates[mode,key].append(stat.evaluate(w))
        if progress is not None and (replicate+1)%50==0:progress(replicate+1)
    for mode,w in (('uncontrolled_effective',sw),('matched_b_delta',matched)):
        control_values={key:stat.evaluate(w) for key,stat in controls.items()}
        for key,stat in stats.items():
            value=stat.evaluate(w)
            row=dict(layer=layer,mode=mode,feature=key,**value,
                status='available' if value['diagnostic_AUROC'] is not None else 'insufficient_common_support_or_feature_coverage',
                direction='GT_covered_minus_normal_high_response; AUROC_GT_positive_no_flip',
                bootstrap_unit='whole_image_within_group',bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=bootstrap_count)
            for g in (0,1):
                finite=(sy==g)&np.isfinite(selected[key])&(w>0)
                covered_images=np.unique(sids[finite]);mass=float(w[finite].sum())
                row[f'group{g}_finite_patches']=int(finite.sum())
                row[f'group{g}_finite_images']=len(covered_images)
                row[f'group{g}_max_finite_image_share']=max((float(w[finite&(sids==sid)].sum())/mass
                    for sid in covered_images),default=None)
            for control,cv in control_values.items():
                row[control+'_mean_difference']=cv['mean_difference_GT_minus_normal']
                row[control+'_standardized_difference']=cv['standardized_difference']
                feature_cv=controls[control].evaluate(w*np.isfinite(selected[key]))
                row['feature_valid_'+control+'_standardized_difference']=feature_cv['standardized_difference']
            row['residual_confounding_flag']=any(row['feature_valid_'+c+'_standardized_difference'] is not None
                and abs(row['feature_valid_'+c+'_standardized_difference'])>.1 for c in controls)
            for field in interval_keys:
                finite=[r[field] for r in replicates[mode,key] if r[field] is not None and np.isfinite(r[field])]
                ci=np.quantile(finite,[.025,.975]) if finite else [None,None]
                row[field+'_CI_low'],row[field+'_CI_high']=ci
                row[field+'_valid_bootstraps']=len(finite)
            comparisons.append(row)
    # Return only small selected arrays/weights for fixed diagnostic CDF figures.
    plot=dict(data=selected,base_weights=sw,matched_weights=matched)
    return summary,comparisons,coverage,bins,plot
