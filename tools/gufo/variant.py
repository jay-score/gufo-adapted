"""Bounded Qwen27B variant gate. Reuses qualification provenance/command logging."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from gufo.qualify.config import Command, config_for
from gufo.qualify.record import _context, _expand_shards, _run_command
from gufo.qualify.util import atomic_json, git_identity, hash_file, utc_now
from gufo.variant_gguf import inventory
from gufo.variant_metrics import capture_rows, compare, enforce
from gufo import variant_reference

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / 'docs/models/qwen3.8-27b/artifacts/variant-qualify.json'


def validate(declaration):
    if declaration.get('schema') != 'gufo-quant-variant/1' or declaration.get('model') != 'qwen3.8-27b':
        raise ValueError('unsupported variant declaration')
    expected = {'short':('--capture-logits', 27, 24),
                'prefill':('--capture-prefill-logits', 189, 168),
                'corpus':('--capture-corpus-logits', 512, 512)}
    actual = {k:(v['flag'], v['rows'], v['labels'])
              for k,v in declaration['captures'].items()}
    if actual != expected:
        raise ValueError('declaration does not cover the native capture contract')
    if (set(declaration['checks']) != {'fast', 'kernels'} or
            any(not names or len(set(names)) != len(names)
                for names in declaration['checks'].values())):
        raise ValueError('missing/duplicate required checks')
    if not declaration['candidate'] or not declaration['teacher']:
        raise ValueError('missing artifact identities')
    if declaration['candidate'] == declaration['teacher']:
        raise ValueError('candidate cannot be its own BF16 teacher')
    for policy in ('limits', 'llama_limits'):
        enforce({k:0.0 for k in declaration[policy]}, declaration[policy])


def compatible(reference, candidate):
    for key in ('architecture', 'config', 'tokenizer_sha256'):
        if reference[key] != candidate[key]:
            raise ValueError('teacher/candidate identity mismatch: ' + key)
    geometry = lambda item: {k:v['shape'] for k,v in item['tensors'].items()}
    if geometry(reference) != geometry(candidate):
        raise ValueError('teacher/candidate tensor names or shapes differ')


def identities(path, expected):
    actual = [hash_file(p) for p in _expand_shards([path])]
    if {p['name']:p['sha256'] for p in actual} != expected:
        raise ValueError('artifact differs from pinned source/checkpoint declaration')
    return actual


def verdict(result):
    if result['errors'] or any(command['status'] != 'pass'
                               for command in result['commands'].values()):
        return 'FAIL'
    required = {'short', 'prefill', 'corpus', 'llama_same_artifact'}
    commands = {'inspect-candidate', 'fast', 'kernels', 'state',
                'capability-candidate', 'inspect-teacher', 'capability-teacher',
                *(f'{side}-{capture}' for side in ('candidate', 'teacher')
                  for capture in ('short', 'prefill', 'corpus'))}
    if (result['unverified'] or not required <= result['comparisons'].keys()
            or not commands <= result['commands'].keys()):
        return 'UNVERIFIED'
    return 'PASS'


def write_report(out, result):
    atomic_json(out / 'qualification.json', result)
    text = ['# Quantization variant qualification', '',
            '**Status: ' + result['status'] + '.**', '', result['scope'], '',
            'Source and identities, commands, support inventory, complete comparison metrics',
            'and limitations are retained in `qualification.json` and the linked raw logs.', '',
            '## Comparisons', '']
    for name, metrics in result.get('comparisons', {}).items():
        text += [f'- {name}: {metrics["rows"]} complete rows, {metrics["labels"]} labels; '
                 f'KL {metrics["mean_kl_nats"]:.8g}, TV {metrics["mean_tv"]:.8g}, '
                 f'perplexity ratio {metrics["perplexity_ratio"]:.8g}.']
    text += ['', '## Gaps and failures', '']
    text += ['- ' + gap for gap in result.get('errors', []) + result.get('unverified', [])]
    text += ['', '## Limits of the claim', '',
             '- BF16 is the same converted model in Gufo, not original-checkpoint/conversion parity.',
             '- Corpus results cover 512 labels; short and repeated-prefill fixtures are reported separately.',
             '- No vision, speculative draft, HTTP API, Linux execution or performance claim.', '']
    (out / 'qualification.md').write_text('\n'.join(text), encoding='utf-8')


def run(args):
    # Model paths belong to the invoking directory, not the build checkout.
    for name in ('candidate', 'teacher', 'corpus', 'llama_server', 'declaration'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    declaration = json.loads(args.declaration.read_text(encoding='utf-8'))
    # Validate policy before looking at any candidate measurements.
    validate(declaration)
    out, repo = args.out.resolve(), args.repo.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory; existing evidence is never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    config = config_for(ROOT, 'qwen3.8-27b')
    tool, probe = args.target_test.resolve(), args.inspect_tool.resolve()
    cache = repo / 'build/gpu-test/CMakeCache.txt'
    cache_text = cache.read_text(encoding='utf-8')
    if f'CMAKE_HOME_DIRECTORY:INTERNAL={repo.as_posix()}' not in cache_text.replace('\\', '/'):
        raise ValueError('gpu-test build belongs to a different checkout')
    for binary in (tool, probe):
        if not binary.is_relative_to(repo / 'build/gpu-test') or not binary.is_file():
            raise ValueError('tools must be from the selected checkout gpu-test build')
    ctx = _context(ROOT, out, tool, repo, config, {}, {}, {}, args.timeout, set())
    result = {'schema':'gufo-quant-variant-result/1', 'status':'UNVERIFIED',
              'started':utc_now(), 'scope':declaration['scope'],
              'declaration':declaration, 'declaration_identity':hash_file(args.declaration),
              'source':git_identity(repo), 'tools':{'target':hash_file(tool), 'inspector':hash_file(probe)},
              'build_cache':hash_file(cache), 'commands':{}, 'comparisons':{},
              'errors':[], 'unverified':[], 'stage':'intake'}
    result['build_inputs'] = {name:hash_file(repo / name) for name in (
        'vcpkg.json', 'CMakePresets.json', 'src/models/qwen/weights.cpp',
        'src/models/qwen/gemm_route.hpp', 'tests/models/qwen27b/target_test.cpp',
        'tests/models/qwen27b/inspect.cpp')}
    result['harness'] = {str(path.relative_to(ROOT)):hash_file(path)
                         for path in [Path(__file__),
                             *Path(__file__).parent.glob('variant_*.py')]}
    result['runtime'] = [hash_file(path) for path in sorted(tool.parent.glob('*.dll'))]
    (out / 'source.patch').write_bytes(subprocess.check_output(
        ['git', 'diff', '--binary', 'HEAD'], cwd=repo))
    (out / 'inspector-source.cpp').write_bytes(
        (repo / 'tests/models/qwen27b/inspect.cpp').read_bytes())
    import re
    compiler = re.search(r'^CMAKE_CXX_COMPILER:[^=]*=(.+)$', cache_text, re.MULTILINE)
    if not compiler:
        raise ValueError('missing actual CMake compiler identity')
    result['compiler'] = hash_file(Path(compiler[1]))
    result['compiler']['version'] = subprocess.check_output(
        [compiler[1], '--version'], text=True, stderr=subprocess.STDOUT)
    def command(name, argv, *, collect_failure=False):
        print('variant stage: ' + name, flush=True)
        result['stage'] = name
        write_report(out, result)
        entry = _run_command(Command(tuple(str(x) for x in argv)), ctx,
                             kind='check', item_id=name, index=0, item_out=out)
        result['commands'][name] = entry
        write_report(out, result)
        if entry['status'] != 'pass':
            reason = name + ': ' + entry['status'] + ' ' + str(entry.get('reason'))
            if not collect_failure:
                raise RuntimeError(reason)
            result['errors'].append(reason)
            result['status'] = 'FAIL'
            write_report(out, result)
        return ((out / entry['stdout']).read_text(encoding='utf-8')
                if 'stdout' in entry else '')
    try:
        result['candidate'] = identities(args.candidate, declaration['candidate'])
        candidate = inventory(args.candidate)
        atomic_json(out / 'candidate-inventory.json', candidate)
        native = json.loads(command('inspect-candidate', [probe, args.candidate]))
        if native.get('schema') != 'gufo-qwen27b-inspection/1' or not native.get('tensors'):
            raise ValueError('invalid native support inventory')
        # Unbound tensors cannot silently become supported runtime weights.
        if any(not any(name.startswith(prefix) for prefix in declaration['excluded_tensor_prefixes'])
               for name in native['unclaimed_tensors']):
            raise ValueError('native inspector found undeclared unclaimed tensors')
        atomic_json(out / 'candidate-routes.json', native)
        result['support'] = {'bound_tensors':len(native['tensors']),
                             'unclaimed_tensors':native['unclaimed_tensors'],
                             'inventory':hash_file(out / 'candidate-inventory.json'),
                             'routes':hash_file(out / 'candidate-routes.json')}
        for group, names in declaration['checks'].items():
            command(group, ['ctest', '--test-dir', str(repo / 'build/gpu-test'),
                            '-R', '^(' + '|'.join(names) + ')$', '--output-on-failure',
                            '--no-tests=error', '--timeout', '300'], collect_failure=True)
            # Require every declared test, not only CTest's matching subset.
            import xml.etree.ElementTree as ET
            junit = out / f'logs/check-{group}-0.junit.xml'
            actual = {case.get('name') for case in ET.parse(junit).getroot().iter('testcase')}
            if actual != set(names):
                raise ValueError('CTest coverage differs from declaration: ' + group)
        command('state', [tool, args.candidate], collect_failure=True)
        command('capability-candidate', [tool, args.candidate, '--capability-check'], collect_failure=True)
        result['corpus'] = hash_file(args.corpus)
        if result['corpus']['sha256'] != declaration['corpus']['sha256']:
            raise ValueError('corpus differs from declaration')
        captures = {}
        for side, model in (('candidate', args.candidate), ('teacher', args.teacher)):
            if model is None:
                result['unverified'].append('BF16 teacher missing')
                continue
            if side == 'teacher':
                result['teacher'] = identities(model, declaration['teacher'])
                teacher = inventory(model)
                compatible(teacher, candidate)
                atomic_json(out / 'teacher-inventory.json', teacher)
                command('inspect-teacher', [probe, model])
                command('capability-teacher', [tool, model, '--capability-check'], collect_failure=True)
            captures[side] = {}
            for name, settings in declaration['captures'].items():
                destination = out / f'{side}-{name}.bin'
                argv = [tool, model, settings['flag']]
                if name == 'corpus':
                    argv.append(args.corpus)
                log = command(f'{side}-{name}', [*argv, destination])
                rows = capture_rows(log)
                if (len(rows) != settings['rows'] or
                        sum(row['label'] is not None for row in rows) != settings['labels']):
                    raise ValueError('incomplete capture: ' + name)
                captures[side][name] = (destination, rows)
                result.setdefault('captures', {})[f'{side}-{name}'] = hash_file(destination)
                atomic_json(out / f'{side}-{name}-rows.json', rows)
        if 'teacher' in captures:
            for name in declaration['captures']:
                p, p_rows = captures['teacher'][name]
                q, q_rows = captures['candidate'][name]
                metrics = compare(p, q, p_rows, q_rows)
                result['comparisons'][name] = metrics
                result['errors'] += [name + ': ' + e for e in enforce(metrics, declaration['limits'])]
                write_report(out, result)
        if args.llama_server:
            result['tools']['llama_server'] = hash_file(args.llama_server)
            result['llama_runtime'] = [hash_file(path)
                for path in sorted(args.llama_server.parent.glob('*.dll'))]
            result['llama_version'] = subprocess.check_output(
                [str(args.llama_server), '--version'], text=True, stderr=subprocess.STDOUT)
            p, p_rows, evidence = variant_reference.capture(args.llama_server,
                args.candidate, captures['candidate']['short'][1], out)
            atomic_json(out / 'llama-rows.json', p_rows)
            evidence['capture'] = hash_file(p)
            evidence['row_manifest'] = hash_file(out / 'llama-rows.json')
            q, q_rows = captures['candidate']['short']
            metrics = compare(p, q, p_rows, q_rows)
            result['comparisons']['llama_same_artifact'] = metrics
            result['llama'] = evidence
            result['errors'] += ['llama: ' + e for e in enforce(metrics, declaration['llama_limits'])]
        else:
            result['unverified'].append('independent same-artifact llama.cpp reference missing')
        result['status'] = verdict(result)
        result['stage'] = 'complete'
    except (OSError, ValueError, RuntimeError, KeyError, ArithmeticError) as error:
        result['errors'].append(str(error))
        result['status'] = 'FAIL'
    result['finished'] = utc_now()
    write_report(out, result)
    print('variant result: ' + result['status'], flush=True)
    return 0 if result['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='inventory only; cannot grant runtime support')
    inspect.add_argument('model', type=Path)
    inspect.add_argument('--out', type=Path, required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--candidate', type=Path, required=True)
    run_parser.add_argument('--teacher', type=Path)
    run_parser.add_argument('--corpus', type=Path, required=True)
    run_parser.add_argument('--repo', type=Path, required=True)
    run_parser.add_argument('--inspect-tool', type=Path, required=True)
    run_parser.add_argument('--target-test', type=Path, required=True)
    run_parser.add_argument('--llama-server', type=Path)
    run_parser.add_argument('--declaration', type=Path, default=DECLARATION)
    run_parser.add_argument('--out', type=Path, required=True)
    run_parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            atomic_json(args.out, inventory(args.model))
            print('Inventory only: runtime/quality UNVERIFIED')
            return 0
        return run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print('qualify-variant: ' + str(error), file=sys.stderr)
        return 1
