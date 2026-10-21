"""Read a newly completed original pipeline; never relabel it as old capture."""
from pathlib import Path
import hashlib
from . import inputs as ip


class LiveParent(ip.Parent):
    def __init__(self,root):
        self.path=Path(root).resolve();self.content={};self.origin='new_complete_pipeline_rerun'
        self.source_files={}
        # This logical namespace reuses the metadata reader. No file is copied,
        # renamed or rewritten, and the new capture identity remains its own.
        pairs=[('cache','source_provenance/cache'),('prediction','source_provenance/prediction'),
               ('frozen_inputs','source_provenance/frozen_inputs'),('cpu','cpu')]
        for source,destination in pairs:
            directory=self.path/source
            if not directory.is_dir():raise FileNotFoundError('missing completed pipeline directory: '+str(directory))
            for p in directory.rglob('*'):
                if p.is_file() and p.suffix in ('.json','.csv','.npz','.md') and not any(
                    part in ('raw_cache','support_features','predictions') for part in p.relative_to(directory).parts):
                    rel=p.relative_to(self.path).as_posix();blob=p.read_bytes()
                    self.source_files[rel]=hashlib.sha256(blob).hexdigest()
                    self.content['live/'+destination+'/'+p.relative_to(directory).as_posix()]=blob
        config=self.path/'run_config.json';blob=config.read_bytes()
        self.source_files['run_config.json']=hashlib.sha256(blob).hexdigest()
        self.content['live/source_provenance/run_config.json']=blob
        self.source_sha256=ip.digest(self.source_files)
        self.validate()
        if self.protocol!=ip.old.PROTOCOL:raise ValueError('rerun changes the frozen crossdomain protocol')
        extraction=self.frozen['frozen']['extraction']
        for key,expected in dict(query=ip.old.QUERY,precision=ip.old.PRECISION,layers=list(ip.LAYERS),dimension=1280,
            memory_sha256=self.catalog['memory_sha256'],checkpoint_sha256=self.catalog['checkpoint_sha256']).items():
            if extraction.get(key)!=expected:raise ValueError('rerun changes original extraction: '+key)
        self.cache_root=self.path/'cache'


def unchanged(parent):
    if isinstance(parent,LiveParent):
        if LiveParent(parent.path).source_sha256!=parent.source_sha256:
            raise ValueError('live parent metadata changed during analysis')
