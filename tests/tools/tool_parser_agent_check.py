"""Bounded real-model tool/parser checks through both stateless APIs.

Run against an isolated Gufo server. Retains exact requests, responses, latency,
usage, cache history and independently checked agent file contents under --out.
Model copying failures remain failures; deterministic C++ fixtures establish
the parser contract independently of model behavior.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


LITERAL = "Example: <tool_call> <｜DSML｜tool_calls> π <tool_"
OLD = "  before\n\n"
NEW = "  after\nExample: <tool_call> </parameter> </tool_call>\n"


def function(name, parameters):
    return {"type": "function", "function": {
        "name": name, "description": name.replace("_", " "),
        "parameters": parameters}}


def request_body(model, api, messages, tools=(), choice="auto"):
    body = {"model": model, "temperature": 0, "seed": 31,
            "tool_choice": choice}
    if api == "chat":
        body.update(messages=deepcopy(messages), max_completion_tokens=256,
                    chat_template_kwargs={"enable_thinking": False})
        body["tools"] = deepcopy(list(tools))
    else:
        body.update(input=deepcopy(messages), max_output_tokens=256,
                    reasoning={"effort": "none"})
        body["tools"] = [dict(type="function", **tool["function"])
                         for tool in tools]
    return body


def normalized(value, api):
    if "error" in value and value["error"]:
        return "", []
    if api == "chat":
        message = value["choices"][0]["message"]
        return message.get("content") or "", [
            {"call_id": call["id"], **call["function"]}
            for call in message.get("tool_calls", [])]
    text, calls = "", []
    for item in value.get("output", []):
        if item["type"] == "function_call":
            calls.append(item)
        elif item["type"] == "message":
            text += "".join(part.get("text", "")
                            for part in item["content"])
    return text, calls


class Check:
    def __init__(self, url, model, out):
        self.url, self.model, self.out = url.rstrip("/"), model, out
        self.records = []

    def post(self, label, api, body):
        path = "/v1/chat/completions" if api == "chat" else "/v1/responses"
        request = Request(self.url + path, json.dumps(body).encode(),
                          {"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=90) as response:
                status, raw = response.status, response.read()
        except HTTPError as error:
            status, raw = error.code, error.read()
        record = {"label": label, "api": api, "request": body,
                  "status": status, "response": json.loads(raw),
                  "elapsed_ms": (time.perf_counter() - started) * 1000}
        self.records.append(record)
        print(f"{api} {label}: HTTP {status}", flush=True)
        self.save()
        return record

    def save(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "requests.json").write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")

    def probes(self, api):
        wildcard = function("f", {"type": "object",
            "properties": {"value": {"type": "string"}}, "required": ["value"],
            "patternProperties": {"^x_": {"type": "integer"}}})
        wildcard["function"]["strict"] = False
        edits = function("edit_fixture", {"type": "object", "properties": {
            "updates": {"type": "array", "items": {"type": "object",
                "properties": {"old_text": {"type": "string"},
                               "new_text": {"type": "string"}},
                "required": ["old_text", "new_text"]}}}, "required": ["updates"]})
        desired = {"updates": [{"old_text": OLD, "new_text": NEW}]}
        cases = [
            ("no_tools", [], "none", "Copy this text exactly, with no fences: " + LITERAL,
             lambda text, calls: text == LITERAL and not calls),
            ("disabled", [wildcard], "none", "Copy this text exactly, with no fences: " + LITERAL,
             lambda text, calls: text == LITERAL and not calls),
            ("wildcard", [wildcard], "required",
             "Call f with value set to the string 1 and x_count set to integer 1. No other arguments.",
             lambda text, calls: len(calls) == 1 and
             json.loads(calls[0]["arguments"]) == {"value": "1", "x_count": 1}),
            ("multiline", [edits], "required", "Call edit_fixture with exactly these arguments: " +
             json.dumps(desired, ensure_ascii=False),
             lambda text, calls: len(calls) == 1 and json.loads(calls[0]["arguments"]) == desired),
        ]
        for label, tools, choice, content, predicate in cases:
            messages = [{"role": "system", "content": "Tool parser probe: " + label},
                        {"role": "user", "content": content}]
            body = request_body(self.model, api, messages, tools, choice)
            for phase in ("fresh", "repeat"):
                record = self.post(label + "_" + phase, api, body)
                text, calls = normalized(record["response"], api)
                try:
                    passed = record["status"] == 200 and predicate(text, calls)
                except (ValueError, KeyError, TypeError):
                    passed = False
                record["passed"] = passed
            if label == "wildcard":
                interrupted = deepcopy(body)
                interrupted["max_completion_tokens" if api == "chat" else "max_output_tokens"] = 1
                record = self.post("interrupted_required", api, interrupted)
                _, calls = normalized(record["response"], api)
                record["passed"] = record["status"] == 200 and not calls
                record = self.post("resumed_required", api, body)
                text, calls = normalized(record["response"], api)
                try:
                    record["passed"] = record["status"] == 200 and predicate(text, calls)
                except (ValueError, KeyError, TypeError):
                    record["passed"] = False
        # A past name is a history record, not a current declaration.
        call = {"name": "retired_工具", "arguments": "{}"}
        if api == "chat":
            history = [{"role": "assistant", "content": None, "tool_calls": [
                {"id": "past", "type": "function", "function": call}]},
                {"role": "tool", "tool_call_id": "past", "content": "ALPHA"}]
        else:
            history = [{"type": "function_call", "call_id": "past", **call},
                       {"type": "function_call_output", "call_id": "past", "output": "ALPHA"}]
        messages = [{"role": "user", "content": "Read fixture status."}, *history,
                    {"role": "user", "content": "Reply with only the status ALPHA."}]
        record = self.post("historical", api,
                           request_body(self.model, api, messages, [wildcard], "none"))
        text, calls = normalized(record["response"], api)
        record["passed"] = record["status"] == 200 and text.strip() == "ALPHA" and not calls
        return edits

    def agent(self, api, edits):
        fixture = self.out / (api + "-fixture.txt")
        fixture.write_text(OLD, encoding="utf-8", newline="")
        tools = [function("read_fixture", {"type": "object", "properties": {}}),
                 edits, function("finish", {"type": "object", "properties": {}})]
        messages = [{"role": "system", "content":
                     "Use tools in this order: read_fixture, edit_fixture, read_fixture to verify, finish. "
                     "Edit the fixture by replacing its old text with exactly " + json.dumps(NEW) +
                     ". Preserve indentation and protocol literals. Do not repeat an action without progress."},
                    {"role": "user", "content": "Read, edit, verify and finish the fixture."}]
        actions, seen, repeats = [], set(), 0
        for turn in range(6):
            record = self.post(f"agent_{turn}", api,
                               request_body(self.model, api, messages, tools))
            if record["status"] != 200:
                break
            _, calls = normalized(record["response"], api)
            if api == "chat":
                messages.append(record["response"]["choices"][0]["message"])
            else:
                messages.extend(record["response"]["output"])
            if not calls:
                break
            finished = False
            for call in calls:
                before = fixture.read_text(encoding="utf-8")
                identity = (call["name"], call["arguments"], before)
                repeats += identity in seen
                seen.add(identity)
                actions.append(call["name"])
                try:
                    args = json.loads(call["arguments"])
                    if call["name"] == "read_fixture":
                        result = before
                    elif call["name"] == "edit_fixture":
                        text = before
                        for update in args["updates"]:
                            if text.count(update["old_text"]) != 1:
                                raise ValueError("old_text must match exactly once")
                            text = text.replace(update["old_text"], update["new_text"], 1)
                        fixture.write_text(text, encoding="utf-8", newline="")
                        result = "Edit applied. Read the fixture to verify."
                    elif call["name"] == "finish":
                        finished = True
                        result = "Finished."
                    else:
                        raise ValueError("undeclared function")
                except (ValueError, KeyError, TypeError) as error:
                    result = "Tool error: " + str(error)
                if api == "chat":
                    messages.append({"role": "tool", "tool_call_id": call["call_id"], "content": result})
                else:
                    messages.append({"type": "function_call_output", "call_id": call["call_id"], "output": result})
            if finished:
                break
        return {"actions": actions, "repeated_without_progress": repeats,
                "file_contents": fixture.read_text(encoding="utf-8"),
                "passed": fixture.read_bytes() == NEW.encode() and
                actions == ["read_fixture", "edit_fixture", "read_fixture", "finish"] and repeats == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with urlopen(args.url.rstrip("/") + "/v1/models", timeout=10) as response:
        models = json.load(response)
    model = models["data"][0]["id"]
    check = Check(args.url, model, args.out)
    agents = {}
    for api in ("chat", "responses"):
        agents[api] = check.agent(api, check.probes(api))
    check.save()
    summary = {"model": models, "agents": agents,
               "checks": [{k: v for k, v in record.items()
                           if k in ("label", "api", "status", "elapsed_ms", "passed")} |
                          {"usage": record["response"].get("usage")}
                          for record in check.records]}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = all(r.get("passed", True) for r in check.records) and all(a["passed"] for a in agents.values())
    print("PASS" if passed else "FAIL (inspect retained requests and file contents)", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
