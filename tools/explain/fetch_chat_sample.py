"""Downloads Hugging Face's everyday-conversations chat set for model diffing (FEAT-8):

    python3 tools/explain/fetch_chat_sample.py ~/.cache/pulsatrix/chat

writes conversations.jsonl into the directory: one {"text": ...} line per conversation, in the
ChatML format SmolLM2-Instruct was trained on (its chat template, with its default system
prompt), so a base model and its chat fine-tune can be read on the same text. The set
(HuggingFaceTB/everyday-conversations-llama3.1-2k) is part of SmolLM2-Instruct's own training
data; both splits are taken, about 2,300 conversations. Standard library only: rows come from the
datasets-server API, 100 at a time.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

DATASET = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
API = "https://datasets-server.huggingface.co/rows"
SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def chatml(messages):
    text = ""
    if not messages or messages[0]["role"] != "system":
        text += "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
    for m in messages:
        text += "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
    return text


def rows(split):
    offset = 0
    while True:
        query = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": split, "offset": offset, "length": 100})
        with urllib.request.urlopen(API + "?" + query, timeout=60) as r:
            page = json.load(r)
        batch = page.get("rows", [])
        if not batch:
            return
        for row in batch:
            yield row["row"]
        offset += len(batch)
        if offset >= page.get("num_rows_total", offset):
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "conversations.jsonl")
    count = 0
    with open(path, "w", encoding="utf-8") as out:
        for split in ("train_sft", "test_sft"):
            for row in rows(split):
                out.write(json.dumps({"text": chatml(row["messages"])}, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} conversations to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
