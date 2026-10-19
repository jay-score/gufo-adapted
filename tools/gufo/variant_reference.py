"""Owned llama.cpp server, full-vocabulary reference on explicit token prefixes."""
from __future__ import annotations

from array import array
import copy
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def post(url, body):
    request = urllib.request.Request(url, json.dumps(body).encode(),
                                     {'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def full_logprobs(response, vocabulary):
    groups = response.get('completion_probabilities', response.get('probs', []))
    if len(groups) != 1:
        raise ValueError('llama.cpp did not return one probability row')
    entries = groups[0].get('top_logprobs', [])
    if len(entries) != vocabulary:
        raise ValueError('llama.cpp did not return the complete vocabulary')
    values, seen = [0.0] * vocabulary, set()
    for entry in entries:
        token, value = entry['id'], entry['logprob']
        if (not isinstance(token, int) or not 0 <= token < vocabulary or token in seen
                or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError('invalid/duplicate/nonfinite llama.cpp probability')
        seen.add(token)
        values[token] = value
    # A full finite log-probability vector defines its distribution up to an
    # additive constant. HIP's reported FP32 softmax can sum to 1.000122; use
    # stable FP64 normalization, just as for Gufo's raw logits. This changes no
    # KL/TV/NLL acceptance budget and never fills missing vocabulary entries.
    maximum = max(values)
    log_z = maximum + math.log(sum(math.exp(v - maximum) for v in values))
    return [value - log_z for value in values]


def capture(binary: Path, model: Path, rows: list, out: Path) -> tuple[Path, list, dict]:
    # Never reuse a user's server or its loaded model/cache.
    with socket.socket() as endpoint:
        endpoint.bind(('127.0.0.1', 0))
        port = endpoint.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    argv = [str(binary), '-m', str(model), '-c', '4096', '-ngl', '99',
            '--parallel', '1', '--host', '127.0.0.1', '--port', str(port),
            '--no-warmup']
    ref_rows = copy.deepcopy(rows)
    destination = out / 'llama-logprobs.bin'
    with (out / 'llama-server.log').open('wb') as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError('llama.cpp exited during loading; see llama-server.log')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('llama.cpp model load timed out')
                time.sleep(0.25)
            with destination.open('wb') as stream:
                for i, row in enumerate(ref_rows):
                    body = {'prompt':row['prefix'], 'n_predict':1,
                            'n_probs':row['vocab'], 'temperature':1,
                            'samplers':[], 'post_sampling_probs':False,
                            'cache_prompt':False, 'return_tokens':True,
                            'seed':459}
                    response = post(url + '/completion', body)
                    (out / f'llama-row-{i}.json').write_text(
                        json.dumps({'request':body, 'response':response}), encoding='utf-8')
                    values = array('f', full_logprobs(response, row['vocab']))
                    if sys.byteorder != 'little':
                        values.byteswap()
                    data = values.tobytes()
                    row['sha256'] = hashlib.sha256(data).hexdigest()
                    stream.write(data)
                    print(f'llama.cpp reference row {i+1}/{len(rows)}', flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
    return destination, ref_rows, {'argv':argv, 'rows':len(ref_rows),
                                  'purpose':'independent execution of identical Q5 weights'}
