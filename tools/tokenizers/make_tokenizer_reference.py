"""Reference tokenizations for TOK-2, from Hugging Face `tokenizers` (tests/tokenizer_json_test.cpp,
pulsatrix_tokenizer_parity). Runs in the pulsatrix-lrpref image (tools/lrp_reference.Dockerfile):

    docker run --rm -v "$PWD":/w -v ~/.cache/pulsatrix:/c -w /w pulsatrix-lrpref:latest \
        python3 tools/tokenizers/make_tokenizer_reference.py tiny \
            smollm2=/c/golden/SmolLM2-135M/tokenizer.json qwen2.5=/c/golden/Qwen2.5-0.5B/tokenizer.json ...

tiny NAME=TOKENIZER_JSON ...
    For each real tokenizer.json, writes tests/fixtures/tokenizers/NAME/tokenizer.json: the same
    pipeline (normalizer, pre-tokenizer, post-processor, decoder, added tokens, BPE options and
    merge format), with a small BPE trained on the CI corpus in place of the real vocabulary, plus
    reference.jsonl for the CI corpus. Small enough to commit; the pipelines are the real ones.

reference TOKENIZER_JSON CORPUS_JSONL OUT_JSONL
    Reference tokenizations of a corpus by a tokenizer.json (the full corpus test, run locally).

corpus OUT_JSONL [--lines N]
    The full multilingual corpus: the CI lines, then lines of the Universal Declaration of Human
    Rights in every language of the UDHR in XML project (downloaded; not committed), then
    seeded mixtures of both, up to N lines (default 10000).

Every corpus and reference file is JSON lines. A reference line holds the text, whether the
normalizer changed it, ids (add_special_tokens=True), plain_ids (False), offsets (in characters,
as Hugging Face reports them), and decode(ids) with and without skip_special_tokens.
"""

import io
import json
import os
import random
import sys
import urllib.request
import zipfile

import tokenizers
from tokenizers import Tokenizer, pre_tokenizers, trainers

sys.path.insert(0, os.path.dirname(__file__))
from corpus_lines import LINES  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures", "tokenizers")
TINY_VOCAB = 1000
# Byte-level pieces of CI corpus words ("Ġ" is a space), added to the vocabularies of pipelines with
# ignore_merges.
UNREACHABLE_WORDS = ["Ġpneumonoultramicroscopicsilicovolcanoconiosis", "Ġantidisestablishmentarianism", "Ġquick"]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reference(tok, texts):
    rows = []
    for text in texts:
        enc = tok.encode(text)
        normalized = tok.normalizer is not None and tok.normalizer.normalize_str(text) != text
        rows.append({"text": text, "normalized": normalized,
                     "ids": enc.ids, "plain_ids": tok.encode(text, add_special_tokens=False).ids,
                     "offsets": [list(o) for o in enc.offsets],
                     "decoded": tok.decode(enc.ids, skip_special_tokens=False),
                     "decoded_skip": tok.decode(enc.ids, skip_special_tokens=True)})
    return rows


def tiny(name, template_path):
    template = json.load(open(template_path, encoding="utf-8"))
    real_model = template["model"]
    # Train a small BPE through the real normalizer and pre-tokenizer.
    skeleton = dict(template)
    skeleton["added_tokens"] = []
    skeleton["post_processor"] = None
    skeleton["model"] = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                         "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "vocab": {}, "merges": []}
    tok = Tokenizer.from_str(json.dumps(skeleton))
    trainer = trainers.BpeTrainer(vocab_size=TINY_VOCAB, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(LINES * 3, trainer)
    trained = json.loads(tok.to_str())

    out = dict(template)
    model = trained["model"]
    for key in ("ignore_merges", "continuing_subword_prefix", "end_of_word_suffix"):
        if key in real_model:
            model[key] = real_model[key]
    if real_model["merges"] and isinstance(real_model["merges"][0], str):
        model["merges"] = [" ".join(m) for m in model["merges"]]  # the real file's "a b" format
    if real_model.get("ignore_merges"):
        # Whole words no merge sequence reaches, as real Llama 3 and gpt-oss vocabularies hold:
        # only ignore_merges turns them into one token. A trained vocabulary has none.
        for word in UNREACHABLE_WORDS:
            if word not in model["vocab"]:
                model["vocab"][word] = max(model["vocab"].values()) + 1
    out["model"] = model
    # The real added tokens, renumbered after the small vocabulary, and the template's ids with them.
    remap = {}
    next_id = max(model["vocab"].values()) + 1
    added = []
    for t in template["added_tokens"]:
        remap[t["id"]] = next_id
        added.append(dict(t, id=next_id))
        next_id += 1
    out["added_tokens"] = added

    def renumber(processor):
        if processor is None:
            return
        if processor.get("type") == "Sequence":
            for p in processor["processors"]:
                renumber(p)
        if processor.get("type") == "TemplateProcessing":
            for special in processor["special_tokens"].values():
                special["ids"] = [remap[i] for i in special["ids"]]
    renumber(out.get("post_processor"))

    directory = os.path.join(FIXTURES, name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "tokenizer.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    final = Tokenizer.from_file(path)
    write_jsonl(os.path.join(directory, "reference.jsonl"), reference(final, LINES))
    print("%s: %d tokens, %d merges, %d added; reference for %d lines" %
          (name, len(model["vocab"]), len(model["merges"]), len(added), len(LINES)))


# The UDHR in XML project (Unicode's UDHR collection), as its GitHub repository archive.
UDHR = "https://github.com/eric-muller/udhr/archive/refs/heads/master.zip"


def corpus(out_path, n):
    import xml.etree.ElementTree as ET
    lines = list(LINES)
    with urllib.request.urlopen(UDHR, timeout=300) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))
    udhr = []
    for member in sorted(archive.namelist()):
        name = member.rsplit("/", 1)[-1]
        if "/data/udhr/" not in member or not name.startswith("udhr_") or not name.endswith(".xml"):
            continue
        root = ET.fromstring(archive.read(member))
        texts = [" ".join("".join(e.itertext()).split()) for e in root.iter() if e.tag.split("}")[-1] in ("title", "para")]
        udhr.extend([t for t in texts if t][:12])  # the title, preamble and first articles of each language
    rng = random.Random(0)
    rng.shuffle(udhr)
    lines.extend(udhr[: max(0, n - len(lines))])
    pool = lines[:]
    while len(lines) < n:  # mixtures: two lines joined by a space, a newline or nothing
        a, b = rng.choice(pool), rng.choice(pool)
        lines.append(a + rng.choice([" ", "\n", "", "\t", "  "]) + b)
    write_jsonl(out_path, [{"text": t} for t in lines[:n]])
    print("wrote %d lines (%d from the UDHR)" % (n, min(len(udhr), n - len(LINES))))


def main():
    print("tokenizers", tokenizers.__version__)
    command = sys.argv[1]
    if command == "tiny":
        for arg in sys.argv[2:]:
            name, path = arg.split("=", 1)
            tiny(name, path)
    elif command == "reference":
        tok = Tokenizer.from_file(sys.argv[2])
        rows = reference(tok, [row["text"] for row in read_jsonl(sys.argv[3])])
        write_jsonl(sys.argv[4], rows)
        print("wrote %d reference lines" % len(rows))
    elif command == "corpus":
        n = int(sys.argv[sys.argv.index("--lines") + 1]) if "--lines" in sys.argv else 10000
        corpus(sys.argv[2], n)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
