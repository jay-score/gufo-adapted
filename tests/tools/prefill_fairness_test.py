"""Explicit C2 real-model prefill fairness check against an isolated server.

Requires two sessions, at least 16384 context tokens, and no other clients.
Uses the repository's standard-library streaming benchmark transport.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from gufo.serving_bench import run_request


def metrics(url):
    with urlopen(url.rstrip("/") + "/metrics", timeout=10) as response:
        fields = dict(line.split() for line in response.read().decode().splitlines()
                      if line and not line.startswith("#"))
    return (float(fields["llamacpp:prompt_tokens_total"]),
            float(fields["llamacpp:tokens_predicted_total"]))


def wait_for_work(url, pending, initial, counter):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        observed = metrics(url)
        if observed[counter] > initial[counter]:
            assert not pending.done(), "request ended before arrival could be injected"
            return observed
        if pending.done():
            pending.result()
            raise AssertionError("request produced no observable work")
        time.sleep(0.02)
    raise AssertionError("timed out waiting for live request work")


def check(url, model, results):
    long_prompt = ("Archive notes: memory pages preserve physical storage. " * 1200
                   + "\nReply with only ALPHA.")
    short_prompt = "Reply with only BETA."

    def request(prompt, tokens, label):
        return run_request(base_url=url, model=model, prompt=prompt,
                           max_tokens=tokens, temperature=0, timeout_seconds=1200,
                           client_id="prefill-fairness-" + label, concurrency=2,
                           repetition=0, request_index=0, cache_prompt=False,
                           extra_body={"reasoning_effort": "none", "seed": 470})

    control = request(short_prompt, 4, "control")
    with ThreadPoolExecutor(max_workers=2) as pool:
        initial = metrics(url)
        long_pending = pool.submit(request, long_prompt, 4, "long")
        arrival = wait_for_work(url, long_pending, initial, 0)
        short_pending = pool.submit(request, short_prompt, 4, "short")
        short, long = short_pending.result(), long_pending.result()
        results["short_arrival"] = dict(short=asdict(short), long=asdict(long),
                                        arrival_prompt_tokens=arrival[0] - initial[0])
        assert long.prefill_tokens > 4096 and long.cached_prompt_tokens == 0
        assert short.completion_sha256 == control.completion_sha256
        short_first = short.started_at + short.client_ttft_ms / 1000
        long_first = long.started_at + long.client_ttft_ms / 1000
        short_is_early = short_first < long_first
        results["short_arrival"]["short_preceded_long_prefill"] = short_is_early
        assert short.completion_tokens > 0 and long.completion_tokens > 0

        initial = metrics(url)
        decoder = pool.submit(request, "Count from 1 to 1000, one integer per line.",
                              256, "decoder")
        before_long = wait_for_work(url, decoder, initial, 1)
        pending = pool.submit(request, long_prompt, 4, "decode-peer")
        samples = []
        while not pending.done():
            prompt, generated = metrics(url)
            samples.append((time.perf_counter(), prompt, generated))
            time.sleep(0.02)
        peer, ongoing = pending.result(), decoder.result()
        results["ongoing_decode"] = dict(decoder=asdict(ongoing), peer=asdict(peer),
                                         samples=samples)
        peer_first = peer.started_at + peer.client_ttft_ms / 1000
        assert peer.prefill_tokens > 4096 and peer.cached_prompt_tokens == 0
        assert ongoing.completion_tokens >= 32, "decode fixture ended too early"
        assert any(stamp < peer_first and prompt > before_long[0]
                   and generated > before_long[1]
                   for stamp, prompt, generated in samples), "decode stalled throughout prefill"
        assert len({prompt for stamp, prompt, generated in samples
                    if stamp < peer_first}) >= 2, "long peer made no incremental progress"
        assert short_is_early, "short output waited for long prefill completion"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="gufo")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    try:
        check(args.url, args.model, results)
        results["passed"] = True
    except Exception as error:
        results.update(passed=False, error=str(error))
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
