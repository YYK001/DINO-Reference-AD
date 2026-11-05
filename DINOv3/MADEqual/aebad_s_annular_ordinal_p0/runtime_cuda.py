"""GPU batch implementation plugged into the original image iterator and pipeline."""
import time
import json
from collections import OrderedDict
import numpy as np
import torch
from . import runtime as base,pixels_cuda as px,geometry_cuda as geo,evidence_cuda as ev
from . import compact_storage as codec
from .index_cuda import Store
st=base.st

class Engine(base.Engine):
    def __init__(self,a):
        tick=time.perf_counter();torch.set_num_threads(a.cpu_workers);super().__init__(a,Store(a));self.writer=codec.Writer(a.writer_depth);self.triples=OrderedDict()
        a.gpu_startup_seconds=time.perf_counter()-tick
        self.event_start=0;self.wall_start={};self.query=None;self.checked=False
    def prepare_image(self,row):
        self.event_start=len(self.store.timers.events);self.wall_start=dict(self.store.timers.wall)
        self.metrics_start=dict(self.store.metrics);self.uploads_start=self.store.images.uploads;self.hits_start=self.store.images.hits
        torch.cuda.reset_peak_memory_stats(self.store.device)
        self.query=self.store.image(row)
        return self.store.gray(row)  # readonly host view for dimensions and bounded fixed-case rendering
    def normal_image(self,index):return self.store.image(self.store.rows[index])
    def compute_batch(self,gray,batch,row,case,start,boxes):
        arrays=[];summaries=[]
        for lo in range(0,len(batch),self.a.window_batch):
            positions=batch[lo:lo+self.a.window_batch];timers=self.store.timers;device=self.store.device
            with timers.measure('anchors_and_descriptor'):
                outer=px.outer_batch(self.query,positions)
            with timers.measure('D2H'):infos=px.outer_to_host(outer)
            bits=outer['bits'][outer['anchor_valid']]
            lookups=self.store.lookup(bits) if len(bits) else []
            matches=[];flags=[];cursor=0
            for info in infos:
                count=len(info['points']);m,flag=self.store.matches(info,lookups[cursor:cursor+count]);cursor+=count
                matches.append(m);flags.append(flag)
            def get_image(index):return self.normal_image(index)
            def prefetch(indices):
                for index in indices[:self.a.prefetch_depth]:self.normal_image(index)
            get_image.prefetch=prefetch
            with timers.measure('geometry'):
                frozen=geo.freeze_batch(infos,matches,self.store.references,positions,get_image,device,
                    group_batch=self.a.geometry_groups,hypothesis_batch=self.a.hypothesis_batch,triple_cache=self.triples)
            with timers.measure('normal_center'):
                normal=ev.normal_expectations(frozen,self.store.references,positions,get_image,device,candidate_batch=self.a.hypothesis_batch)
            with timers.measure('query_evidence_and_pack'):
                result=ev.query_evidence(self.query,normal,positions)
            ids=np.full((len(positions),3,12),-1,np.int16);trans=np.zeros((len(positions),3,12,2,3),np.float64);usage=[]
            for j,item in enumerate(frozen):
                candidates=[]
                for k,name in enumerate(base.SETS):
                    for h,hypothesis in enumerate(item[name]['candidates']):ids[j,k,h]=hypothesis['ref'];trans[j,k,h]=hypothesis['T'];candidates.append(hypothesis)
                usage.append([any(h['kind']==kind for h in candidates) for kind in ('Source','Target')])
            dense=dict(boxes=torch.as_tensor(positions,device=device,dtype=torch.int32),
                candidate_references=torch.as_tensor(ids,device=device),transforms=torch.as_tensor(trans,device=device),
                **{name:result[name] for name in ('scores','evidence_present','missing_mask_packed','flip_counts','agreement_fraction','disagreement_fraction','distinct_references')})
            # Only small per-window control/status reductions return to the CPU.
            with timers.measure('D2H'):
                small={k:result[k].cpu().numpy() for k in ('scores','evidence_present','agreement_fraction','disagreement_fraction','distinct_references')}
            stats=[]
            for j,item in enumerate(frozen):
                evidence=[dict(disagreement_fraction=small['disagreement_fraction'][j,k],distinct_references=small['distinct_references'][j,k]) for k in range(3)]
                stats.append(base.diagnostics(item,evidence,matches[j],flags[j]))
                if case and start+lo+j in (0,len(boxes)//2):
                    with timers.measure('fixed_case_CPU_render'):
                        norm=dict(disagreement=normal['disagreement'][j,2].cpu().numpy())
                        preview=dict(evidence=result['evidence'][j,2].cpu().numpy(),missing_mask=(~normal['agreed'][j,2]).cpu().numpy())
                        base.visualize(self.a,row,gray,positions[j],item,norm,preview,self.store.references,start+lo+j)
            arrays.append(dense);summaries.append(dict(E=small['scores'].max(0),valid_windows=small['evidence_present'].sum(0),
                agreement_sum=small['agreement_fraction'].sum(0),disagreement_sum=small['disagreement_fraction'].sum(0),
                distinct_reference_sum=small['distinct_references'].sum(0),reference_used_windows=np.sum(usage,0),
                counters={key:int(sum(s[key] for s in stats)) for key in stats[0]}))
        dense={key:torch.cat([a[key] for a in arrays]) for key in arrays[0]}
        summary={key:(np.max([s[key] for s in summaries],0) if key=='E' else np.sum([s[key] for s in summaries],0)).tolist()
            for key in ('E','valid_windows','agreement_sum','disagreement_sum','distinct_reference_sum','reference_used_windows')}
        summary['counters']={key:sum(s['counters'][key] for s in summaries) for key in summaries[0]['counters']}
        with self.store.timers.measure('sparse_encode'):compact=codec.encode_tensors(dense)
        return compact,summary
    def commit_batch(self,dest,name,batch,index,arrays,summary,began):
        summary=dict(summary,identity=base.ip.identity(self.a),index=index['seal'],boxes=[list(b) for b in batch],
            mask_layout='sparse_lossless_v1 reconstructs original (window,branch,24,64,64) masks/counts/candidates')
        with self.store.timers.measure('D2H_and_writer_queue'):
            return self.writer.submit(self.a,dest,name,arrays,summary,began)
    def finish_image(self,record,tick,row):
        writes=[f.result() for f in self.writer.futures];self.writer.flush();elapsed=time.perf_counter()-tick
        record.update(seconds=elapsed if not record['reused_chunk_count'] else record['seconds'],
            full_image_wall_seconds=elapsed,whole_image_timing_complete=not bool(record['reused_chunk_count']),
            timings=self.store.timers.collect(self.event_start,self.wall_start),retrieval={k:v-self.metrics_start.get(k,0) for k,v in self.store.metrics.items()},
            index_residency=self.store.residency,image_uploads=self.store.images.uploads-self.uploads_start,image_cache_hits=self.store.images.hits-self.hits_start,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.store.device),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(self.store.device))
        record['writer']=dict(current_session_write_seconds=sum(r['write_seconds'] for r in writes),
            compression='ZIP_STORED; no deflate',overlaps_next_GPU_batch=True)
        # Keep only compact measurements in the review output, not the masks.
        path=self.a.run_root/'gpu_image_timings.json';table=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        table[row['sample_id']]={k:record[k] for k in ('sample_id','windows','full_image_wall_seconds','whole_image_timing_complete',
            'timings','retrieval','index_residency','image_uploads','image_cache_hits','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','writer')}
        st.write_json(path,table);self.store.timers.events.clear()
        return super().finish_image(record,tick,row)
    def verify(self):
        if not self.checked:
            from .verify_cuda import verify
            result=verify(self.a,self.store.device,self.store);self.checked=True;return result
    def close(self):
        try:self.writer.close()
        finally:super().close()
