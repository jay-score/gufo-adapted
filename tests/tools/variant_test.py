"""CPU fixtures for the quantization variant gate; no model or GPU required."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from gufo.variant import compatible, identities, validate, verdict, write_report
from gufo.variant_gguf import InvalidGGUF, inventory
from gufo.variant_metrics import assess_quality, capture_rows, compare, enforce
from gufo.variant_reference import full_logprobs


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def gguf(path, *, tensors=None, metadata=None, truncate=0):
    values = [('general.architecture', 8, 'qwen35'),
              ('tokenizer.ggml.model', 8, 'fixture'),
              ('qwen35.embedding_length', 4, 2)]
    values += metadata or []
    tensors = [('weight', [2], 0, 0)] if tensors is None else tensors
    data = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(values))
    for key, kind, value in values:
        data += string(key) + struct.pack('<I', kind)
        data += string(value) if kind == 8 else struct.pack('<I', value)
    for name, shape, kind, offset in tensors:
        data += string(name) + struct.pack('<I', len(shape))
        data += b''.join(struct.pack('<Q', d) for d in shape)
        data += struct.pack('<IQ', kind, offset)
    data += b'\0' * (-len(data) % 32) + b'\0' * 8
    path.write_bytes(data[:len(data)-truncate] if truncate else data)


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gufo-variant-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'model.gguf'

    def test_complete_inventory(self):
        gguf(self.model)
        result = inventory(self.model)
        self.assertEqual(result['runtime_support'], 'UNVERIFIED')
        self.assertEqual(result['tensors']['weight']['encoded_bytes'], 8)
        self.assertEqual(result['architecture'], 'qwen35')
        self.assertEqual(result['tensors']['weight']['shape'], [2])

    def test_truncated_header_and_payload(self):
        for count in (1, 12, 100, 200):
            with self.subTest(count=count):
                gguf(self.model, truncate=count)
                with self.assertRaises((InvalidGGUF, UnicodeError)):
                    inventory(self.model)

    def test_duplicate_tensors_and_metadata(self):
        for kwargs in ({'tensors':[('weight', [2], 0, 0)] * 2},
                       {'metadata':[('general.architecture', 8, 'duplicate')]}):
            with self.subTest(kwargs=kwargs):
                gguf(self.model, **kwargs)
                with self.assertRaises(InvalidGGUF):
                    inventory(self.model)

    def test_bad_shape_offset_storage_and_overlap(self):
        for tensors in ([('x', [0], 0, 0)], [('x', [3], 12, 0)],
                        [('x', [2], 0, 1)], [('x', [2], 100, 0)],
                        [('x', [2], 0, 0), ('y', [2], 0, 0)]):
            with self.subTest(tensors=tensors):
                gguf(self.model, tensors=tensors)
                with self.assertRaises(InvalidGGUF):
                    inventory(self.model)

    def test_missing_or_inconsistent_shard(self):
        first = self.root / 'part-00001-of-00002.gguf'
        gguf(first)
        with self.assertRaises(RuntimeError):
            inventory(first)
        second = self.root / 'part-00002-of-00002.gguf'
        gguf(second)
        with self.assertRaises(InvalidGGUF):
            inventory(first)

    def test_complete_shards_and_duplicate_cross_shard_tensor(self):
        paths = [self.root / f'part-{i+1:05d}-of-00002.gguf' for i in range(2)]
        for i, path in enumerate(paths):
            gguf(path, tensors=[(f'w{i}', [2], 0, 0)], metadata=[
                ('split.no', 4, i), ('split.count', 4, 2), ('split.tensors.count', 4, 2)])
        self.assertEqual(len(inventory(paths[0])['tensors']), 2)
        gguf(paths[1], tensors=[('w0', [2], 0, 0)], metadata=[
            ('split.no', 4, 1), ('split.count', 4, 2)])
        with self.assertRaises(InvalidGGUF):
            inventory(paths[0])

    def test_checkpoint_tokenizer_and_geometry_mismatch(self):
        gguf(self.model)
        candidate = inventory(self.model)
        for key, value in (('architecture','other'), ('tokenizer_sha256','wrong'),
                           ('config',{}), ('tensors',{})):
            altered = copy.deepcopy(candidate)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compatible(altered, candidate)
        compatible(candidate, candidate)

    def test_pinned_identity_rejects_different_artifact(self):
        self.model.write_bytes(b'fixture')
        expected = {self.model.name:hashlib.sha256(b'fixture').hexdigest()}
        self.assertEqual(len(identities(self.model, expected)), 1)
        self.model.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            identities(self.model, expected)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gufo-variant-metrics-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'capture.bin'

    def fixture(self, values=(0.0,1.0)):
        data = struct.pack('<2f', *values)
        self.path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        return capture_rows('capture case=0 prompt=1 tokens=0,1,\n'
            f'capture case=0 row=0 vocab=2 sha256={digest}\n')

    def test_exact_rows_measure_zero_drift(self):
        rows = self.fixture()
        result = compare(self.path, self.path, rows, rows)
        self.assertEqual(result['mean_kl_nats'], 0)
        self.assertEqual(result['mean_tv'], 0)
        self.assertEqual(result['perplexity_ratio'], 1)
        self.assertEqual(result['labels'], 1)
        self.assertEqual(result['row_tv_percentiles'], {str(p):0 for p in (50,90,95,99)})

    def test_reference_direction_and_nll_ratio_are_analytic(self):
        reference_rows = self.fixture((0.0, 0.0))
        reference = self.path.with_name('reference.bin')
        reference.write_bytes(self.path.read_bytes())
        candidate_rows = self.fixture((0.0, math.log(3.0)))
        result = compare(reference, self.path, reference_rows, candidate_rows,
                         tv_thresholds=(0.15, 0.30))
        self.assertAlmostEqual(result['mean_kl_nats'], 0.5 * math.log(4.0/3.0), places=7)
        self.assertAlmostEqual(result['mean_tv'], 0.25, places=7)
        self.assertAlmostEqual(result['perplexity_ratio'], 2.0/3.0, places=7)
        self.assertAlmostEqual(result['mean_nll_delta'], math.log(2.0/3.0), places=7)
        self.assertEqual(result['row_tv_exceedances'],
                         [{'threshold':0.15,'rows':1}, {'threshold':0.30,'rows':0}])
        self.assertAlmostEqual(result['row_tv_percentiles']['99'], 0.25, places=7)

    def test_histories_vocabulary_and_labels_must_match(self):
        rows = self.fixture()
        for key, value in (('prefix',[1]), ('vocab',3), ('label',0)):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(self.path, self.path, rows, changed)

    def test_tail_summary_counts_rare_shifts(self):
        reference = self.path.with_name('reference.bin')
        reference.write_bytes(struct.pack('<4f', 0, 0, 0, 0))
        self.path.write_bytes(struct.pack('<4f', 0, 0, 0, math.log(3)))
        rows = [{'case':0, 'row':i, 'vocab':2, 'prefix':[0]*(i+1), 'label':1,
                 'sha256':hashlib.sha256(reference.read_bytes()[i*8:i*8+8]).hexdigest()}
                for i in range(2)]
        candidate_rows = copy.deepcopy(rows)
        for i, row in enumerate(candidate_rows):
            row['sha256'] = hashlib.sha256(self.path.read_bytes()[i*8:i*8+8]).hexdigest()
        metrics = compare(reference, self.path, rows, candidate_rows, tv_thresholds=(0.15,))
        self.assertEqual(metrics['row_tv_percentiles']['50'], 0)
        self.assertAlmostEqual(metrics['row_tv_percentiles']['99'], 0.25, places=7)
        self.assertEqual(metrics['row_tv_exceedances'], [{'threshold':0.15,'rows':1}])

    def test_nonfinite_corrupt_truncated_or_extra_raw_rows_fail(self):
        for values in ((float('nan'),1), (0,float('inf'))):
            rows = self.fixture(values)
            with self.assertRaises(ValueError):
                compare(self.path, self.path, rows, rows)
        rows = self.fixture()
        for data in (b'bad', self.path.read_bytes() + b'extra'):
            self.path.write_bytes(data)
            with self.assertRaises(ValueError):
                compare(self.path, self.path, rows, rows)

    def test_missing_duplicate_and_noncontiguous_capture_rows_fail(self):
        rows = self.fixture()
        self.assertEqual(len(rows),1)
        line = 'capture case=0 row=0 vocab=2 sha256=' + '0'*64 + '\n'
        for log in ('', 'capture case=0 prompt=1 tokens=0,1,\n' + line*2,
                    'capture case=0 prompt=1 tokens=0,1,\n' + line.replace('row=0','row=1')):
            with self.subTest(log=log), self.assertRaises(ValueError):
                capture_rows(log)

    def test_limits_enforced_and_missing_or_nonfinite_rejected(self):
        limits = {'mean_kl_nats':0.01, 'mean_tv':0.05,
                  'maximum_row_tv':0.15, 'perplexity_ratio':1.05}
        metrics = {k:0.0 for k in limits}
        self.assertEqual(enforce(metrics,limits),[])
        for key in limits:
            changed = {**metrics,key:limits[key]+0.1}
            self.assertEqual(len(enforce(changed,limits)),1)
            for value in (float('nan'),float('inf'),None):
                with self.assertRaises(ValueError):
                    enforce({**metrics,key:value},limits)
        with self.assertRaises(ValueError):
            enforce(metrics,{})

    def test_reference_requires_every_unique_finite_token(self):
        response = {'probs':[{'top_logprobs':[
            {'id':0,'logprob':-0.6931471805599453},
            {'id':1,'logprob':-0.6931471805599453}]}]}
        self.assertEqual(len(full_logprobs(response,2)),2)
        shifted = copy.deepcopy(response)
        for entry in shifted['probs'][0]['top_logprobs']:
            entry['logprob'] += 0.000122
        for a,b in zip(full_logprobs(response,2), full_logprobs(shifted,2)):
            self.assertAlmostEqual(a,b,places=12)
        for entries in ([{'id':0,'logprob':0}],
                        [{'id':0,'logprob':0}]*2,
                        [{'id':0,'logprob':float('nan')},{'id':1,'logprob':0}]):
            with self.assertRaises(ValueError):
                full_logprobs({'probs':[{'top_logprobs':entries}]},2)


class PolicyTest(unittest.TestCase):
    def v2(self):
        declaration = json.loads((ROOT / 'docs/models/qwen3.8-27b/artifacts/variant-qualify.json').read_text())
        declaration['schema'] = 'gufo-quant-variant/2'
        limits = declaration.pop('limits')
        declaration['quality_limits'] = {'perplexity_ratio':limits.pop('perplexity_ratio')}
        declaration['investigation_triggers'] = limits
        declaration['policy_rationale'] = 'Synthetic fixture policy; not a new Q5 qualification.'
        declaration['heldout_evaluation_plan'] = 'Fixture only; real campaigns require fresh held-out inputs.'
        return declaration

    def test_v2_outlier_alert_does_not_replace_quality_or_execution_failure(self):
        declaration = self.v2()
        validate(declaration)
        metrics = {'mean_kl_nats':0.002, 'mean_tv':0.017,
                   'maximum_row_tv':0.31, 'perplexity_ratio':1.002}
        failures, alerts = assess_quality(metrics, declaration)
        self.assertEqual(failures, [])
        self.assertEqual(len(alerts), 1)
        self.assertIn('maximum_row_tv', alerts[0])
        self.assertEqual(len(enforce(metrics, declaration['llama_limits'])), 3)
        failures, alerts = assess_quality({**metrics,'perplexity_ratio':1.06}, declaration)
        self.assertEqual(len(failures), 1)
        self.assertEqual(len(alerts), 1)
        for key in metrics:
            with self.subTest(key=key), self.assertRaises(ValueError):
                assess_quality({**metrics,key:float('nan')}, declaration)

    def test_v1_outlier_remains_a_failure(self):
        declaration = json.loads((ROOT / 'docs/models/qwen3.8-27b/artifacts/variant-qualify.json').read_text())
        failures, alerts = assess_quality({'mean_kl_nats':0.002, 'mean_tv':0.017,
            'maximum_row_tv':0.31, 'perplexity_ratio':1.002}, declaration)
        self.assertEqual(len(failures), 1)
        self.assertEqual(alerts, [])

    def test_v2_refuses_ambiguous_missing_or_invalid_policy(self):
        for key in ('quality_limits', 'investigation_triggers'):
            declaration = self.v2()
            del declaration[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate(declaration)
        for key, value in (('quality_limits',{}), ('investigation_triggers',{}),
                           ('quality_limits',{'maximum_row_tv':0.15,'perplexity_ratio':1.05}),
                           ('policy_rationale',''), ('heldout_evaluation_plan',''),
                           ('investigation_triggers',{'mean_kl_nats':0.01,'mean_tv':0.05,
                                                     'maximum_row_tv':float('inf')}),
                           ('limits',{})):
            declaration = self.v2()
            declaration[key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                validate(declaration)

    def test_report_keeps_distribution_alerts_visible(self):
        with tempfile.TemporaryDirectory(prefix='gufo-variant-report-') as temp:
            out = Path(temp)
            result = {'status':'PASS', 'scope':'Synthetic fixture',
                      'declaration':self.v2(), 'alerts':['corpus: maximum_row_tv outlier']}
            write_report(out, result)
            self.assertEqual(json.loads((out / 'qualification.json').read_text())['alerts'],
                             result['alerts'])
            self.assertIn(result['alerts'][0], (out / 'qualification.md').read_text())

    def test_failed_skipped_or_missing_gates_never_pass(self):
        commands = ('inspect-candidate', 'fast', 'kernels', 'state',
                    'capability-candidate', 'inspect-teacher', 'capability-teacher',
                    'candidate-short', 'candidate-prefill', 'candidate-corpus',
                    'teacher-short', 'teacher-prefill', 'teacher-corpus')
        result = {'errors':[], 'unverified':[],
                  'commands':{name:{'status':'pass'} for name in commands},
                  'comparisons':{k:{} for k in ('short','prefill','corpus','llama_same_artifact')}}
        self.assertEqual(verdict(result),'PASS')
        result['alerts'] = ['distribution outlier; investigate']
        self.assertEqual(verdict(result),'PASS')
        for status in ('fail','skipped','not-applicable','timeout'):
            changed = copy.deepcopy(result)
            changed['commands']['kernels']['status'] = status
            self.assertEqual(verdict(changed),'FAIL')
        for key in ('short','prefill','corpus','llama_same_artifact'):
            changed = copy.deepcopy(result)
            del changed['comparisons'][key]
            self.assertEqual(verdict(changed),'UNVERIFIED')
        for key in commands:
            changed = copy.deepcopy(result)
            del changed['commands'][key]
            self.assertEqual(verdict(changed),'UNVERIFIED')
        result['unverified'] = ['missing teacher']
        self.assertEqual(verdict(result),'UNVERIFIED')

    def test_pilot_declaration_and_missing_required_evidence(self):
        declaration = json.loads((ROOT / 'docs/models/qwen3.8-27b/artifacts/variant-qualify.json').read_text())
        validate(declaration)
        for key in ('captures','checks','limits','candidate','teacher'):
            altered = copy.deepcopy(declaration)
            altered[key] = {}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(altered)


if __name__ == '__main__':
    unittest.main()
