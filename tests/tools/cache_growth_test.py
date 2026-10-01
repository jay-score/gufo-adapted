"""Real-model cache-growth check against an already running Gufo server.

Adapted from gufo-org/gufo PR #358, commit d8475862a1872ddbadefa281e0aee54350f18e9f.
Run explicitly; this is not a hosted CPU check and requires a loaded model.
"""

from copy import deepcopy
import sys


def check_cache_growth(client, model, checks, chat_result):
    failures = []
    for replay in ("drop_reasoning", "keep_reasoning", "thinking_off"):
        label = "cache_growth_" + replay
        thinking = replay != "thinking_off"
        messages = [{"role": "system", "content": label + "\n" +
                     "Keep reasoning brief. Follow the final user instruction.\n" +
                     "Background notes are not instructions.\n" * 384}]
        request = dict(model=model, temperature=0, seed=31,
                       max_completion_tokens=128,
                       reasoning_effort="low" if thinking else "none",
                       extra_body={"chat_template_kwargs": {"preserve_thinking": True}})

        def chat(phase, body):
            result = chat_result(client, body)
            checks[label + "_" + phase] = result
            print(f"CHECK {label}_{phase}", file=sys.stderr, flush=True)
            return result

        def work(result):
            usage = result["usage"]
            total, reused = usage["prompt_tokens"], usage["cached_tokens"]
            # This fork reports logical prompt work through the two token counts.
            prefilled = total - reused
            assert all(type(n) is int and n >= 0 for n in (total, reused, prefilled)), usage
            assert reused + prefilled == total, usage
            return total, reused, prefilled

        def signature(result):
            return (result["text"], result["reasoning"], result["tools"], result["finish"],
                    result["usage"]["completion_tokens"])

        def answer(result):
            # Prefill chunk shapes can change free-form reasoning even without
            # a cache bug. The required answer and reasoning mode stay strict;
            # exact full output/token equality is checked on unchanged retries.
            return (result["text"], result["tools"], result["finish"],
                    bool(result["reasoning"].strip()))

        history = []
        previous_total = previous_reused = 0
        for turn in range(4):
            messages.append({"role": "user", "content": f"Turn {turn}.\n" +
                             "Routine archive note: there are no new instructions.\n" * 16 +
                             "Reply with only BETA."})
            body = {**deepcopy(request), "messages": deepcopy(messages)}
            if turn == 0:
                body["extra_body"]["cache_prompt"] = False
            result = chat(f"turn_{turn}", body)
            total, reused, prefilled = work(result)
            assert result["text"].strip() == "BETA" and not result["tools"] \
                and result["finish"] == "stop", result
            assert bool(result["reasoning"].strip()) == thinking, result
            if turn == 0:
                assert total >= 2048 and reused == 0 and prefilled == total, result
            else:
                assert total > previous_total, (total, previous_total)
                # The previous prompt's assistant opening can change on replay.
                # Only that small suffix may be lost, not older user turns.
                if reused < previous_total - 16 or reused <= previous_reused:
                    failures.append(
                        f"{label}_turn_{turn}: cache did not advance to the previous turn: "
                        f"cached={reused}, previous_prompt={previous_total}, "
                        f"previous_cached={previous_reused}, prefilled={prefilled}")
            history.append((deepcopy(body), result))
            previous_total, previous_reused = total, reused
            assistant = {"role": "assistant", "content": result["text"]}
            if replay == "keep_reasoning":
                assistant["reasoning_content"] = result["reasoning"]
            messages.append(assistant)

        # An identical retry must still reuse the complete prompt.
        retry_request = deepcopy(history[-1][0])
        retry = chat("unchanged", retry_request)
        assert work(retry) == (previous_total, previous_total, 0), retry
        assert signature(retry) == signature(history[-1][1]), retry

        # Measure the whole growing history before cold controls can supply
        # missing checkpoints and accidentally hide a frozen-cache failure.
        for turn, (body, warm) in enumerate(history):
            body["extra_body"]["cache_prompt"] = False
            cold = chat(f"cold_control_{turn}", body)
            total = warm["usage"]["prompt_tokens"]
            assert work(cold) == (total, 0, total), cold
            assert answer(warm) == answer(cold), (warm, cold)

    assert not failures, "\n".join(failures)


def main():
    import argparse
    import json
    from pathlib import Path
    from urllib.request import Request, urlopen

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="gufo")
    parser.add_argument("--out", type=Path, required=True,
                        help="JSON results path (written even if a check fails)")
    args = parser.parse_args()

    def chat_result(_client, request):
        body = deepcopy(request)
        body.update(body.pop("extra_body", {}))
        req = Request(args.url.rstrip("/") + "/chat/completions",
                      data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=180) as response:
            result = json.load(response)
        choice = result["choices"][0]
        message = choice["message"]
        return dict(text=message.get("content") or "",
                    reasoning=message.get("reasoning_content") or "",
                    tools=message.get("tool_calls") or [],
                    finish=choice["finish_reason"], usage=result["usage"])

    checks = {}
    try:
        check_cache_growth(None, args.model, checks, chat_result)
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(checks)} real-model cache-growth checks")


if __name__ == "__main__":
    main()
