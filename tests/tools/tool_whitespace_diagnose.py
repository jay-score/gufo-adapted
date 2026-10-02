"""Compare exact-edit failures with controlled tool-result and prompt changes.

Run against an isolated server. The original case remains the control; alternate
cases are diagnostics, not replacements for its byte-for-byte acceptance check.
Each case retains requests, responses, actions and actual fixture contents.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from urllib.request import urlopen

from tool_parser_agent_check import Check, NEW, function, normalized


CASES = ("raw", "plain_raw", "json", "description", "plain_description_json",
         "escaped_description_json", "plain_recovery_json", "escaped_recovery_json")
DESCRIPTION = (
    "Apply exact substring replacements to the fixture. Each old_text must "
    "include every original space and newline being replaced and match exactly "
    "once. Characters outside the matching substring are preserved. To replace "
    "the entire file, use its complete contents as old_text, including ALL "
    "trailing newlines. Never trim old_text or new_text.")


class DiagnosticCheck(Check):
    def __init__(self, url, model, out, case):
        super().__init__(url, model, out)
        self.case = case
        out.mkdir(parents=True, exist_ok=True)

    def post(self, label, api, body):
        body = deepcopy(body)
        messages = body["messages" if api == "chat" else "input"]
        if "recovery" in self.case:
            replacement = "  after\n" if self.case.startswith("plain") else NEW
            messages[0]["content"] = (
                "Read the fixture, use edit_fixture to make its entire contents exactly " +
                json.dumps(replacement) + ". Read again to verify every character, "
                "including leading spaces and ALL trailing newlines. If different, "
                "correct the edit and read again. Call finish only after the contents "
                "match exactly. Do not repeat an action without progress.")
        if self.case.startswith("escaped"):
            messages[0]["content"] = (messages[0]["content"]
                                      .replace("<", "\\u003c")
                                      .replace(">", "\\u003e"))
        if self.case.endswith("json"):
            reads = set()
            for message in messages:
                if api == "chat":
                    for call in message.get("tool_calls", []):
                        if call["function"]["name"] == "read_fixture":
                            reads.add(call["id"])
                    if (message.get("role") == "tool" and
                            message["tool_call_id"] in reads):
                        message["content"] = json.dumps({"content": message["content"]})
                else:
                    if (message.get("type") == "function_call" and
                            message["name"] == "read_fixture"):
                        reads.add(message["call_id"])
                    if (message.get("type") == "function_call_output" and
                            message["call_id"] in reads):
                        message["output"] = json.dumps({"content": message["output"]})
        return super().post(label, api, body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES,
                        help="Repeat to select cases; defaults to all cases.")
    parser.add_argument("--api", choices=("chat", "responses", "both"), default="both")
    args = parser.parse_args()
    with urlopen(args.url.rstrip("/") + "/v1/models", timeout=10) as response:
        models = json.load(response)
    tool = function("edit_fixture", {"type": "object", "properties": {
        "updates": {"type": "array", "items": {"type": "object", "properties": {
            "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            "required": ["old_text", "new_text"]}}}, "required": ["updates"]})
    apis = ("chat", "responses") if args.api == "both" else (args.api,)
    summary = {"model": models, "results": {}}
    for case in args.case or CASES:
        edits = deepcopy(tool)
        if "description" in case or "recovery" in case:
            edits["function"]["description"] = DESCRIPTION
        check = DiagnosticCheck(args.url, models["data"][0]["id"], args.out / case, case)
        for api in apis:
            replacement = "  after\n" if case.startswith("plain") else NEW
            result = check.agent(api, edits, replacement=replacement)
            if "recovery" in case:
                actions = result["actions"]
                _, final_calls = normalized(check.records[-1]["response"], api)
                result["passed"] = (result["file_matches"] and
                                    actions[:1] == ["read_fixture"] and
                                    actions[-2:] == ["read_fixture", "finish"] and
                                    [c["name"] for c in final_calls] == ["finish"] and
                                    "edit_fixture" in actions and
                                    result["repeated_without_progress"] == 0)
            result["expected_contents"] = replacement
            summary["results"][case + "-" + api] = result
            check.save()
            (args.out / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(case, api, "PASS" if result["passed"] else "FAIL", flush=True)
    return 0 if all(r["passed"] for r in summary["results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
