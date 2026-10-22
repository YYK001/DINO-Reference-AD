"""Explicit transfer of frozen CPU parents. Never publishes assets to GitHub."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
from . import inputs as ip,storage as st

MANIFEST='witness_transfer_manifest.json'


def selected_files(root):
    root=Path(root).resolve();paths=ip.parent.defaults();background=ip.parent.bg.defaults(root);files=set()
    for k in ('raw_archive','review_archive'):files.add(background[k])
    for k in ('formal_dir','ch1_dir'):
        files.update(p for p in background[k].rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.npz','.md'))
    for key in ('background_cpu','background_aupro'):
        files.update(p for p in paths[key].rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.npz','.md'))
    cross=paths['parent_run'];files.add(cross/'run_config.json')
    for folder in ('cache','prediction','frozen_inputs','cpu','aupro'):
        for p in (cross/folder).rglob('*'):
            if (p.is_file() and p.suffix in ('.json','.csv','.npz','.md')
                and not any(part in ('raw_cache','support_features','predictions') for part in p.relative_to(cross/folder).parts)):
                files.add(p)
    # Precisely the original629 capture records, no model/Source bank/Q arrays.
    for row in st.read_json(cross/'cache/cache_status.json')['records']:
        files.add(cross/'cache'/ip.parent.bg.safe(row['file']))
    for p in files:
        if not p.is_file():raise FileNotFoundError(p)
        if not p.resolve().is_relative_to(root):raise ValueError('transfer input outside project root')
    return sorted(files)


def pack(root,output):
    root=Path(root).resolve();output=Path(output).resolve();files=selected_files(root)
    if output.exists():raise FileExistsError('existing transfer package preserved: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True);records=[]
    # Streaming, no recompression of the already compressed raw NPZs/tars.
    # Each source is read once; the SHA is produced while tar consumes it.
    class Reader:
        def __init__(self,f):self.f=f;self.h=hashlib.sha256()
        def read(self,n=-1):
            value=self.f.read(n);self.h.update(value);return value
    with tarfile.open(output,'x') as archive:
        for i,p in enumerate(files):
            name=p.relative_to(root).as_posix();info=archive.gettarinfo(str(p),arcname=name)
            with p.open('rb') as f:
                reader=Reader(f);archive.addfile(info,reader)
            records.append(dict(file=name,bytes=info.size,sha256=reader.h.hexdigest()))
            if (i+1)%200==0:print(f'[witness transfer] packed {i+1}/{len(files)} files',flush=True)
        blob=(json.dumps(dict(version=1,files=records,code=False,images=False,weights=False,Source_memory=False),indent=2)+'\n').encode()
        info=tarfile.TarInfo(MANIFEST);info.size=len(blob);archive.addfile(info,io.BytesIO(blob))
    return dict(archive=str(output),bytes=output.stat().st_size,files=len(files),status='transfer_package_complete')


def restore(archive_path,root):
    root=Path(root).resolve();count=0
    with tarfile.open(archive_path,'r:') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or names.count(MANIFEST)!=1:raise ValueError('duplicate/missing transfer manifest')
        manifest=json.load(archive.extractfile(MANIFEST));records=manifest['files']
        if set(names)!={r['file'] for r in records}|{MANIFEST} or len(records)!=len(names)-1:raise ValueError('transfer manifest coverage differs')
        for row in records:
            rel=ip.parent.bg.safe(row['file']);member=archive.getmember(rel);target=root/rel
            if not member.isfile() or member.size!=row['bytes'] or not target.resolve().is_relative_to(root):raise ValueError('unsafe/invalid transfer member')
            if target.exists():
                if not target.is_file() or st.sha(target)!=row['sha256']:raise ValueError('existing input differs; preserved '+str(target))
                continue
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.transfer_pending')
            with archive.extractfile(member) as src,temp.open('xb') as dest:shutil.copyfileobj(src,dest,4<<20)
            if st.sha(temp)!=row['sha256']:raise ValueError('transfer member corrupt; preserved '+str(temp))
            temp.replace(target);count+=1
    return dict(status='transfer_restored_without_overwriting_inputs',new_files=count)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('pack','restore'))
    p.add_argument('--archive',required=True,type=Path);a=p.parse_args(argv)
    result=pack(ip.parent.ROOT,a.archive) if a.action=='pack' else restore(a.archive,ip.parent.ROOT)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
