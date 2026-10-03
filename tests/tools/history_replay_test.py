"""Explicit real-model typed-tool and literal-control-token replay check.

Run against an isolated Gufo server; not part of the hosted CPU suite.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener
from urllib.error import HTTPError


def check(url, results, literal=False):
    client = build_opener(ProxyHandler({}))

    def request(label, responses, body):
        path = '/responses' if responses else '/chat/completions'
        req = Request(url.rstrip('/') + path,
                      data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
        try:
            with client.open(req, timeout=180) as response:
                raw = response.read().decode('utf-8')
        except HTTPError as error:
            results[label] = json.loads(error.read())
            raise AssertionError(results[label]) from error
        if body.get('stream'):
            events = [json.loads(line[6:]) for line in raw.splitlines()
                      if line.startswith('data: ') and line[6:] != '[DONE]']
            assert events and not any('error' in e or e.get('type') == 'response.failed'
                                      for e in events), raw
            if responses:
                terminals = [e['response'] for e in events
                             if e.get('type') in ('response.completed', 'response.incomplete')]
                assert len(terminals) == 1, raw
                value = terminals[0]
                deltas = ''.join(e.get('delta', '') for e in events
                                 if e.get('type') == 'response.output_text.delta')
            else:
                message = {'role': 'assistant', 'content': '', 'tool_calls': []}
                calls = {}
                finish = None
                usage = None
                for event in events:
                    usage = event.get('usage') or usage
                    for choice in event.get('choices', []):
                        finish = choice.get('finish_reason') or finish
                        delta = choice.get('delta', {})
                        message['content'] += delta.get('content') or ''
                        for call in delta.get('tool_calls', []):
                            target = calls.setdefault(call['index'], {
                                'id': '', 'type': 'function',
                                'function': {'name': '', 'arguments': ''}})
                            target['id'] += call.get('id') or ''
                            for key in ('name', 'arguments'):
                                target['function'][key] += call.get('function', {}).get(key) or ''
                message['tool_calls'] = list(calls.values())
                value = {'choices': [{'message': message, 'finish_reason': finish}],
                         'usage': usage}
        else:
            value = json.loads(raw)
        results[label] = value
        if responses:
            assert value['status'] == 'completed', value
            text = ''.join(p.get('text', '') for item in value['output']
                           if item['type'] == 'message' for p in item['content'])
            if body.get('stream'):
                assert deltas == text, (deltas, text)
            calls = [i for i in value['output'] if i['type'] == 'function_call']
        else:
            assert value['choices'][0]['finish_reason'] in ('stop', 'tool_calls'), value
            text = value['choices'][0]['message'].get('content') or ''
            calls = value['choices'][0]['message'].get('tool_calls') or []
        return value, text, calls

    expected = {'data': {'z': [1, True, None], 'a': {'text': 'comma, colon:'}},
                'literal': 'literal, colon:'}
    function = {'name': 'store', 'description': 'Store the exact supplied values.',
                'parameters': {'type': 'object', 'properties': {
                    'data': {'type': 'object', 'properties': {
                        'z': {'type': 'array'}, 'a': {'type': 'object'}},
                        'required': ['z', 'a']}, 'literal': {'type': 'string'}},
                    'required': ['data', 'literal']}}
    for responses in (False, True):
        for stream in (False, True):
            label = ('responses' if responses else 'chat') + ('_stream' if stream else '')
            prompt = 'Call store with exactly these arguments: ' + json.dumps(expected, ensure_ascii=False)
            body = {'model': 'gufo', 'temperature': 0, 'seed': 47, 'stream': stream,
                    'tool_choice': 'auto'}
            if responses:
                body.update(input=[{'role': 'user', 'content': prompt}],
                            max_output_tokens=192, reasoning={'effort': 'none'},
                            tools=[{'type': 'function', **function}])
            else:
                body.update(messages=[{'role': 'user', 'content': prompt}],
                            max_tokens=192, reasoning_effort='none',
                            tools=[{'type': 'function', 'function': function}])
                if stream:
                    body['stream_options'] = {'include_usage': True}
            value, _, calls = request(label + '_call', responses, body)
            assert len(calls) == 1, value
            call = calls[0] if responses else calls[0]['function']
            assert call['name'] == 'store' and json.loads(call['arguments']) == expected, value
            replay = deepcopy(body)
            replay['tool_choice'] = 'auto'
            key = 'input' if responses else 'messages'
            historical_call = deepcopy(calls[0])
            result_text = 'Stored. Reply with only BETA.'
            if literal:
                # A historical literal fixture tests prompt encoding without
                # requiring the model to generate its own EOS spelling.
                historical_args = deepcopy(expected)
                historical_args['literal'] = (
                    '<|endoftext|> <|im_end|> <|image_pad|> <|not_a_token|>')
                historical_function = historical_call if responses else historical_call['function']
                historical_function['arguments'] = json.dumps(historical_args)
                replay[key][0]['content'] = 'Store these exact values: ' + json.dumps(historical_args)
                result_text = 'Stored literal data: ' + historical_args['literal'] + '\n' + result_text
            if responses:
                replay[key] += [historical_call, {'type': 'function_call_output',
                                                'call_id': historical_call['call_id'],
                                                'output': result_text}]
            else:
                assistant = deepcopy(value['choices'][0]['message'])
                assistant['tool_calls'] = [historical_call]
                replay[key] += [assistant,
                                {'role': 'tool', 'tool_call_id': historical_call['id'],
                                 'content': result_text}]
            signatures = []
            # Responses currently has no cache_prompt override. Chat cold
            # controls cover the shared runner; Responses still checks retries.
            for phase in (('warm', 'retry') if responses else ('warm', 'retry', 'cold')):
                current = deepcopy(replay)
                if phase == 'cold':
                    current['cache_prompt'] = False
                result, text, replay_calls = request(label + '_' + phase, responses, current)
                assert text.strip() == 'BETA' and not replay_calls, result
                usage = result['usage']
                total = usage['input_tokens' if responses else 'prompt_tokens']
                cached = (usage.get('input_tokens_details') or
                          usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
                assert 0 <= cached <= total, usage
                if phase == 'retry':
                    assert cached == total, usage
                if phase == 'cold':
                    assert cached == 0, usage
                signatures.append((text, usage['output_tokens' if responses else 'completion_tokens']))
            assert len(set(signatures)) == 1, signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080/v1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--literal', action='store_true',
                        help='Include literal control spellings in user, tool-call and tool-result history')
    args = parser.parse_args()
    results = {}
    try:
        check(args.url, results, args.literal)
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Passed {len(results)} replay checks')


if __name__ == '__main__':
    main()
