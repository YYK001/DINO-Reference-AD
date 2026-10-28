"""Intervention on processed input, then the SAME global zero mask and actual host."""
import numpy as np
from . import inputs as ip,context as cx,runtime as rt
from .inputs import st
from .protocol import LAYERS


def check_interventions(value,position,forward,synthesize):
    """CPU-testable complete information path; never use a context cache here.

    forward receives the globally masked input and returns [N,128] descriptors;
    synthesize accepts descriptors only, returns candidates, weights, reference.
    """
    h,w=value.shape[-2:];grid=(h//16,w//16);y,x=divmod(int(position),grid[1]);phase=int(cx.phases(grid)[y,x])
    y0,x0=(y//5)*80,(x//5)*80;y1,x1=min(y0+80,h),min(x0+80,w)
    base=cx.masked_input(value,phase);context=forward(base);candidate,weight,reference=synthesize(context)
    records=[]
    for kind in ('constant','noise'):
        changed=value.copy();shape=changed[...,y0:y1,x0:x1].shape
        changed[...,y0:y1,x0:x1]=7 if kind=='constant' else st.rng('blind_isolation',position).standard_normal(shape).astype(np.float32)
        masked=cx.masked_input(changed,phase)
        if not np.array_equal(base,masked):raise ValueError('masked backbone inputs differ')
        cc=forward(masked);ci,ww,rr=synthesize(cc)
        if not np.allclose(cc,context,atol=1e-7,rtol=1e-6):raise ValueError('context depends on hidden original contents')
        if not np.array_equal(ci,candidate) or not np.allclose(ww,weight,atol=1e-7,rtol=1e-6) or not np.allclose(rr,reference,atol=1e-7,rtol=1e-6):raise ValueError('candidate/weight/reference leaks hidden content')
        records.append(dict(position=int(position),phase=phase,intervention=kind,range=[y0,y1,x0,x1],
                            masked_input_equal=True,context_equal=True,candidates_equal=True,weights_equal=True,reference_equal=True))
    return records


def check(a):
    st.sealed_json(a.run_root/'context_ready.json')
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    if (a.run_root/'trained.json').exists():
        from .training import trained
        heads,record=trained(a);modelseal=record['seal']
    else:heads={l:cx.initial_head(l) for l in LAYERS};modelseal='fixed_initial_heads'
    rows=ip.dictionary_rows(a);selected=[rows[0],rows[64]];records=[]
    import torch
    from .prediction import Synthesizer
    # Use the SAME resident FP32 selector/attention/convex summation as predict.
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reliable_precision
    reliable_precision(a.device);generator=Synthesizer(a,heads)
    with Data(a.data) as data:
        for row in selected:
            value=rt.processed(a,data.image(row)).detach().cpu().numpy();gh,gw=value.shape[-2]//16,value.shape[-1]//16
            # Image edge, an actual block edge, and interior of a non-edge block.
            probes=list(dict.fromkeys([0,min(4,gh-1)*gw+min(4,gw-1),min(7,gh-1)*gw+min(7,gw-1)]))
            for pos in probes:
                def forward(masked):
                    q,m=rt.forward_processed(a,torch.tensor(masked,device=a.device),np.array([pos],np.int64),context_only=True)
                    return np.concatenate([q[l] for l in LAYERS])
                # ONE actual host forward simultaneously supplies all four layers.
                image_id=rows.index(row)
                def synthesize(c):
                    candidates=[];weights=[];references=[]
                    for j,l in enumerate(LAYERS):
                        ref,w,ci=generator.reference(l,c[j:j+1],exclude_image=image_id,return_details=True)
                        candidates.append(ci);weights.append(w);references.append(ref)
                    return np.concatenate(candidates),np.concatenate(weights),np.concatenate(references)
                for r in check_interventions(value,pos,forward,synthesize):records.append(dict(image=row['relative_path'],layers=list(LAYERS),**r))
    st.write_csv(a.run_root/'isolation_checks.csv',records)
    # Deliberately mutable stage result: pre-training and FINAL trained-head checks.
    value=dict(identity=ip.identity(a),passed=all(r['masked_input_equal'] and r['context_equal'] and r['reference_equal'] for r in records),
               model_seal=modelseal,checks=len(records),atol=1e-7,rtol=1e-6,uses_context_cache=False,
               evidence='two real normal images; global processed-input interventions; actual host, projected context and synthesizer',
               does_not_prove_detection_or_defect_preservation=True)
    value['seal']=st.digest(value);st.write_json(a.run_root/'isolation.json',value);return value
