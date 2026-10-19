"""Compare complete FP32 captures on identical teacher-forced histories."""
from __future__ import annotations

from array import array
import hashlib
import math
from pathlib import Path
import re
import sys


def capture_rows(log: str) -> list[dict]:
    cases, rows = {}, []
    for line in log.splitlines():
        match = re.fullmatch(r'capture case=(\d+) prompt=(\d+) tokens=([\d,]+)', line)
        if match:
            case, prompt, tokens = match.groups()
            if int(case) in cases:
                raise ValueError('duplicate capture case')
            cases[int(case)] = (int(prompt), [int(t) for t in tokens.strip(',').split(',')])
        match = re.fullmatch(r'capture case=(\d+) row=(\d+) vocab=(\d+) sha256=([0-9a-f]{64})', line)
        if match:
            case, row, vocab, digest = match.groups()
            prompt, tokens = cases[int(case)]
            index = prompt + int(row)
            if not 0 < prompt <= len(tokens) or int(vocab) <= 1 or index > len(tokens):
                raise ValueError('invalid capture history')
            rows.append({'case':int(case), 'row':int(row), 'vocab':int(vocab),
                         'prefix':tokens[:index],
                         'label':tokens[index] if index < len(tokens) else None,
                         'sha256':digest})
    if not rows or len({(r['case'], r['row']) for r in rows}) != len(rows):
        raise ValueError('missing or duplicate capture rows')
    expected = {}
    for row in rows:
        if row['row'] != expected.get(row['case'], 0):
            raise ValueError('noncontiguous capture rows')
        expected[row['case']] = row['row'] + 1
    return rows


def read_row(stream, row):
    data = stream.read(row['vocab'] * 4)
    if len(data) != row['vocab'] * 4 or hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('truncated or corrupted raw-logit row')
    values = array('f')
    values.frombytes(data)
    if sys.byteorder != 'little':
        values.byteswap()
    if not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite logits')
    return values


def distribution(values):
    maximum = max(values)
    log_z = maximum + math.log(sum(math.exp(v - maximum) for v in values))
    return [math.exp(v - log_z) for v in values], log_z


def compare(reference: Path, candidate: Path, ref_rows: list, cand_rows: list) -> dict:
    if len(ref_rows) != len(cand_rows):
        raise ValueError('capture row counts differ')
    kl_sum = tv_sum = nll_ref = nll_cand = 0.0
    max_tv = 0.0
    labels = top1 = 0
    with reference.open('rb') as p_stream, candidate.open('rb') as q_stream:
        for r, c in zip(ref_rows, cand_rows):
            for key in ('case', 'row', 'vocab', 'prefix', 'label'):
                if r[key] != c[key]:
                    raise ValueError('capture histories/settings differ: ' + key)
            p, q = read_row(p_stream, r), read_row(q_stream, c)
            pp, pz = distribution(p)
            qp, qz = distribution(q)
            kl = sum(prob * ((pv - pz) - (qv - qz)) for prob, pv, qv in zip(pp, p, q))
            tv = 0.5 * sum(abs(a-b) for a,b in zip(pp, qp))
            kl_sum += max(0.0, kl)
            tv_sum += tv
            max_tv = max(tv, max_tv)
            top1 += p.index(max(p)) == q.index(max(q))
            if r['label'] is not None:
                label = r['label']
                if not 0 <= label < len(p):
                    raise ValueError('continuation label outside vocabulary')
                nll_ref += pz - p[label]
                nll_cand += qz - q[label]
                labels += 1
        if p_stream.read(1) or q_stream.read(1) or labels == 0:
            raise ValueError('extra capture bytes or no scored labels')
    delta = (nll_cand - nll_ref) / labels
    return {'rows':len(ref_rows), 'labels':labels, 'top1_matches':top1,
            'mean_kl_nats':kl_sum/len(ref_rows), 'mean_tv':tv_sum/len(ref_rows),
            'maximum_row_tv':max_tv, 'reference_mean_nll':nll_ref/labels,
            'candidate_mean_nll':nll_cand/labels,
            'reference_perplexity':math.exp(nll_ref/labels),
            'candidate_perplexity':math.exp(nll_cand/labels),
            'perplexity_ratio':math.exp(delta), 'mean_nll_delta':delta}


def enforce(metrics: dict, limits: dict) -> list[str]:
    required = {'mean_kl_nats', 'mean_tv', 'maximum_row_tv', 'perplexity_ratio'}
    if not required <= limits.keys():
        raise ValueError('missing required quality limits')
    errors = []
    for metric, maximum in limits.items():
        value = metrics.get(metric)
        if (not isinstance(maximum, (int, float)) or not math.isfinite(maximum)
                or maximum < 0 or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError('missing/nonfinite metric or invalid limit: ' + metric)
        if value > maximum:
            errors.append(f'{metric}={value} exceeds {maximum}')
    return errors
