"""Bounded, standard-library GGUF inventory; runtime support comes from C++."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from gufo.qualify.record import _expand_shards

# Physical GGUF packing geometry, not a runtime capability list. Unsupported
# engine types can still be inventoried; only the native probe grants support.
BLOCKS = {0:(1,4), 1:(1,2), 2:(32,18), 3:(32,20), 6:(32,22), 7:(32,24),
          8:(32,34), 10:(256,84), 11:(256,110), 12:(256,144), 13:(256,176),
          14:(256,210), 15:(256,292), 16:(256,66), 17:(256,74), 18:(256,98),
          19:(256,50), 20:(32,18), 21:(256,110), 22:(256,82), 23:(256,136),
          24:(1,1), 25:(1,2), 26:(1,4), 27:(1,8), 28:(1,8), 29:(256,56),
          30:(1,2), 39:(32,17), 40:(64,36), 42:(64,18)}

class InvalidGGUF(ValueError):
    pass


class Header:
    def __init__(self, stream):
        self.stream = stream
        self.size = stream.seek(0, 2)
        stream.seek(0)

    def read(self, count):
        if count < 0 or count > self.size - self.stream.tell():
            raise InvalidGGUF("truncated GGUF header")
        data = self.stream.read(count)
        if len(data) != count:
            raise InvalidGGUF("truncated GGUF header")
        return data

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self):
        count = self.number('Q')
        if count > 64 << 20:
            raise InvalidGGUF("GGUF string exceeds inspection bound")
        return self.read(count).decode('utf-8', 'strict')

    def value(self, tag, depth=0):
        formats = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f',
                   7:'?', 10:'Q', 11:'q', 12:'d'}
        if tag in formats:
            return self.number(formats[tag])
        if tag == 8:
            return self.string()
        if tag == 9 and depth < 2:
            subtype, count = self.number('I'), self.number('Q')
            if count > 10_000_000 or count > self.size - self.stream.tell():
                raise InvalidGGUF("invalid GGUF array length")
            digest = hashlib.sha256()
            for _ in range(count):
                value = self.value(subtype, depth + 1)
                digest.update(json.dumps(value, ensure_ascii=False,
                                         allow_nan=False).encode() + b'\0')
            return {'count':count, 'type':subtype, 'sha256':digest.hexdigest()}
        raise InvalidGGUF(f"unsupported metadata tag {tag}")


def inventory(path: Path) -> dict:
    paths = _expand_shards([path])
    tensors, metadata, shards = {}, {}, []
    tokenizer = {}
    for index, shard in enumerate(paths):
        with shard.open('rb') as stream:
            r = Header(stream)
            if r.read(4) != b'GGUF' or r.number('I') != 3:
                raise InvalidGGUF('expected GGUF version 3')
            count, kv = r.number('Q'), r.number('Q')
            if count > 100_000 or kv > 100_000:
                raise InvalidGGUF('GGUF header count exceeds inspection bound')
            values = {}
            for _ in range(kv):
                key = r.string()
                if key in values:
                    raise InvalidGGUF('duplicate metadata key: ' + key)
                values[key] = r.value(r.number('I'))
            if len(paths) > 1:
                if values.get('split.no') != index or values.get('split.count') != len(paths):
                    raise InvalidGGUF('inconsistent shard metadata')
            if index == 0:
                metadata = values
                tokenizer = {k:v for k,v in values.items() if k.startswith('tokenizer.')}
                if not tokenizer:
                    raise InvalidGGUF('missing tokenizer metadata')
            else:
                for key, value in values.items():
                    if key.startswith('tokenizer.') and tokenizer.get(key) != value:
                        raise InvalidGGUF('shard tokenizer mismatch')
            for _ in range(count):
                name = r.string()
                dimensions = r.number('I')
                if not 1 <= dimensions <= 4:
                    raise InvalidGGUF('invalid tensor rank: ' + name)
                shape = [r.number('Q') for _ in range(dimensions)]
                kind, offset = r.number('I'), r.number('Q')
                if name in tensors or any(n == 0 for n in shape):
                    raise InvalidGGUF('duplicate tensor or zero shape: ' + name)
                tensors[name] = {'shape':shape, 'type_id':kind, 'offset':offset,
                                 'shard':index}
            alignment = values.get('general.alignment', 32)
            if not isinstance(alignment, int) or alignment < 1 or alignment & (alignment - 1):
                raise InvalidGGUF('invalid GGUF alignment')
            data_offset = (stream.tell() + alignment - 1) // alignment * alignment
            ranges = []
            for name, tensor in tensors.items():
                if tensor['shard'] != index:
                    continue
                if tensor['type_id'] not in BLOCKS:
                    raise InvalidGGUF(f'unknown storage layout {tensor["type_id"]}: {name}')
                block, size = BLOCKS[tensor['type_id']]
                if tensor['shape'][0] % block:
                    raise InvalidGGUF('tensor row is not block aligned: ' + name)
                elements = 1
                for dimension in tensor['shape']:
                    elements *= dimension
                tensor['encoded_bytes'] = elements // block * size
                start = data_offset + tensor['offset']
                end = start + tensor['encoded_bytes']
                if tensor['offset'] % alignment or end > r.size:
                    raise InvalidGGUF('tensor payload truncated/misaligned: ' + name)
                ranges.append((start, end))
            ranges.sort()
            if any(previous[1] > following[0] for previous, following in zip(ranges, ranges[1:])):
                raise InvalidGGUF('overlapping tensor payloads')
            shards.append({'name':shard.name, 'bytes':r.size, 'data_offset':data_offset})
    if len(paths) > 1 and metadata.get('split.tensors.count') != len(tensors):
        raise InvalidGGUF('shard tensor count mismatch')
    architecture = metadata.get('general.architecture')
    config = {k:v for k,v in metadata.items() if k.startswith(str(architecture) + '.')}
    signature = hashlib.sha256(json.dumps(tokenizer, sort_keys=True,
                                         allow_nan=False).encode()).hexdigest()
    return {'schema':'gufo-gguf-inventory/1', 'runtime_support':'UNVERIFIED',
            'architecture':architecture, 'config':config, 'tokenizer_sha256':signature,
            'shards':shards, 'tensors':tensors}
