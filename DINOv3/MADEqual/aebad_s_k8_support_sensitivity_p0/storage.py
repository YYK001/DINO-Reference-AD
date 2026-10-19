"""Append-only result ledger: verify old hashes BEFORE admitting any new work."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import (
    sha, digest, read_json, write_json, write_csv, csv_file, safe, save_npz, load_npz,
)


class Ledger:
    def __init__(self, root, identity):
        self.root = Path(root); self.identity = identity
        self.path = self.root/'ledger.json'
        if self.root.exists():
            self.state = read_json(self.path)  # missing ledger is never a new run
            self.verify()
        else:
            self.root.mkdir(parents=True)
            self.state = dict(identity=identity, records={})
            write_json(self.path, self.state)

    def verify(self):
        if self.state['identity'] != self.identity:
            raise ValueError('frozen inputs/code identity changed; new output required')
        files = set()
        for key, row in self.state['records'].items():
            for name, checksum in row['files'].items():
                if name in files or sha(self.root/safe(name)) != checksum:
                    raise ValueError('recorded file modified/missing or duplicate: '+name)
                files.add(name)
        present = {p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()}
        if present != files | {'ledger.json'}:
            raise ValueError('orphan/temporary file not recorded; preserve and inspect, never re-enroll')

    def get(self, key):
        return self.state['records'].get(key)

    def commit(self, key, payload, data):
        """data: relative filename -> bytes. Immutable key, atomic ledger update."""
        if key in self.state['records']:
            raise ValueError('completed unit cannot be overwritten')
        checksums = {}
        for name, content in data.items():
            path = self.root/safe(name)
            if path.exists():
                raise ValueError('unrecorded destination exists; no re-approval')
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name+'.tmp')
            with temp.open('xb') as f:
                f.write(content)
            temp.replace(path); checksums[name] = sha(path)
        self.state['records'][key] = dict(payload, files=checksums)
        write_json(self.path, self.state)


def npz_bytes(arrays, metadata):
    import io
    stream = io.BytesIO()
    np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
    return stream.getvalue()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                      default=lambda v: v.item() if isinstance(v, np.generic) else str(v)).encode('utf-8')


def csv_bytes(rows):
    import csv, io
    rows = list(rows); stream = io.StringIO(newline='')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    w = csv.DictWriter(stream, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return stream.getvalue().encode('utf-8')
