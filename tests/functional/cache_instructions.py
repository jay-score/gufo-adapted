"""Check a new question under long instructions reuses all of them.

The first request sends a long system prompt and a question; the second keeps
the system prompt and asks another question. Nothing has learned the
divergence point yet, so the second request restores all of the instructions
only if the first one kept a checkpoint where they end; the nearest grid
checkpoint leaves hundreds of instruction tokens to prefill again. Both
answers must match uncached controls.
"""

from copy import deepcopy
import sys

from cache_concurrency import background, system_prompt

# The grid checkpoint interval (TextRunnerPool::Request::Impl and
# cache::PlanCaptures): grid checkpoints sit at multiples of it.
GRID_TOKENS = 2048
# Tokens after the instructions in the uncached stem probe below: the user
# turn framing, its one-word message and the generation prompt.
STEM_ALLOWANCE = 32
# About 5,000 tokens of instructions: more than two grid steps and far from
# the next one, so a grid-only restore misses hundreds of tokens.
INSTRUCTION_LINES = 215
# A short mail-like question of about a hundred tokens.
QUESTION_LINES = 4


def check_cache_instructions(client, model, checks, chat_result):
    request = dict(model=model, temperature=0, seed=31, max_completion_tokens=16,
                   reasoning_effort="none")
    # A unique head keeps other suites' system prompts from sharing this prefix.
    system = {"role": "system", "content": system_prompt("cache_instructions", INSTRUCTION_LINES)}

    def question(code):
        return {"role": "user", "content": background(f"mail {code}", QUESTION_LINES) +
                f"Reply with only the code {code}."}

    def chat(label, messages, cold=False):
        body = {**deepcopy(request), "messages": deepcopy(messages)}
        if cold:
            body["extra_body"] = {"cache_prompt": False}
        result = chat_result(client, body)
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        usage = result["usage"]
        total, reused, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                    usage["gufo"]["prefill_tokens"])
        assert all(type(n) is int and n >= 0 for n in (total, reused, prefilled)), usage
        assert reused + prefilled == total, usage
        if cold:
            assert reused == 0, result
        return result

    def answer(result, code):
        assert result["text"].strip() == code and not result["tools"] \
            and result["finish"] == "stop", result

    conversations = [("first", "ALPHA"), ("second", "BETA")]
    results = {}
    for label, code in conversations:
        results[label] = chat(f"instructions_{label}", [system, question(code)])
        answer(results[label], code)

    # An uncached probe with a one-word question measures the instructions.
    stem = chat("instructions_stem", [system, {"role": "user", "content": "Z"}], cold=True)
    shared = stem["usage"]["prompt_tokens"] - STEM_ALLOWANCE
    grid = shared // GRID_TOKENS * GRID_TOKENS
    assert shared - grid > 2 * STEM_ALLOWANCE, (
        "the instructions end too close to a grid checkpoint to tell them apart; "
        "adjust INSTRUCTION_LINES", shared, grid)
    second = results["second"]["usage"]
    assert second["cached_tokens"] >= shared, (
        f"the second question reused {second['cached_tokens']} tokens, expected at least "
        f"{shared} of the instructions (the nearest grid checkpoint is {grid})", second)
    assert second["gufo"]["prefill_tokens"] <= second["prompt_tokens"] - shared, second

    for label, code in conversations:
        cold = chat(f"instructions_{label}_cold", [system, question(code)], cold=True)
        warm = results[label]
        answer(cold, code)
        assert cold["usage"]["prompt_tokens"] == warm["usage"]["prompt_tokens"], (warm, cold)
        assert cold["text"].strip() == warm["text"].strip(), (warm, cold)
