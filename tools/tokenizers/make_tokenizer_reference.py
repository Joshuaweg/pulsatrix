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

Every corpus and reference file is JSON lines. A reference line holds the text, whether Unicode
normalization (NFC and the like) changed it, ids (add_special_tokens=True), plain_ids (False), offsets (in characters,
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


UNICODE_FORMS = ("NFC", "NFD", "NFKC", "NFKD")


def uses_unicode_normalization(tok):
    """Whether the normalizer includes NFC, NFD, NFKC or NFKD, the forms whose offsets Hugging Face
    aligns by position (Replace and Prepend align exactly and are checked exactly)."""
    def walk(n):
        if not n:
            return False
        return n.get("type") in UNICODE_FORMS or any(walk(m) for m in n.get("normalizers", []))
    return walk(json.loads(tok.to_str()).get("normalizer"))


def reference(tok, texts):
    rows = []
    unicode_forms = uses_unicode_normalization(tok)
    for text in texts:
        enc = tok.encode(text)
        normalized = unicode_forms and tok.normalizer.normalize_str(text) != text
        rows.append({"text": text, "normalized": normalized,
                     "ids": enc.ids, "plain_ids": tok.encode(text, add_special_tokens=False).ids,
                     "offsets": [list(o) for o in enc.offsets],
                     "decoded": tok.decode(enc.ids, skip_special_tokens=False),
                     "decoded_skip": tok.decode(enc.ids, skip_special_tokens=True)})
    return rows


MAX_TINY_ADDED = 64


def _metaspace(scheme, split):
    def patch(t):
        t["normalizer"] = None
        t["pre_tokenizer"] = {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": scheme, "split": split}
        t["decoder"] = {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": scheme, "split": split}
    return patch


def _no_byte_fallback(t):
    t["model"]["byte_fallback"] = False


# Named variants of a real pipeline, for paths no target model takes: SentencePiece's unknown
# token without byte fallback (fuse_unk), and the newer Metaspace form of Llama 2 style files.
VARIANTS = {"-unk": _no_byte_fallback, "metaspace-first": _metaspace("first", False),
            "metaspace-always": _metaspace("always", True)}


def tiny(name, template_path):
    template = json.load(open(template_path, encoding="utf-8"))
    for key, patch in VARIANTS.items():
        if name.endswith(key) or name == key:
            patch(template)
    real_model = template["model"]
    sentencepiece = bool(real_model.get("byte_fallback") or real_model.get("unk_token"))
    # Train a small BPE through the real normalizer and pre-tokenizer.
    skeleton = dict(template)
    skeleton["added_tokens"] = []
    skeleton["post_processor"] = None
    skeleton["model"] = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                         "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "vocab": {}, "merges": []}
    tok = Tokenizer.from_str(json.dumps(skeleton))
    if sentencepiece:
        # A small alphabet, so many characters go through byte fallback (or the unknown token).
        trainer = trainers.BpeTrainer(vocab_size=TINY_VOCAB, limit_alphabet=120, show_progress=False,
                                      special_tokens=[real_model["unk_token"]] if real_model.get("unk_token") else [])
    else:
        trainer = trainers.BpeTrainer(vocab_size=TINY_VOCAB, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=False)
    tok.train_from_iterator(LINES * 3, trainer)
    trained = json.loads(tok.to_str())

    out = dict(template)
    model = trained["model"]
    for key in ("ignore_merges", "continuing_subword_prefix", "end_of_word_suffix", "unk_token", "fuse_unk",
                "byte_fallback"):
        if key in real_model:
            model[key] = real_model[key]
    if real_model.get("byte_fallback"):
        for b in range(256):  # SentencePiece's byte tokens
            token = "<0x%02X>" % b
            if token not in model["vocab"]:
                model["vocab"][token] = max(model["vocab"].values()) + 1
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
    keep = template["added_tokens"]
    if len(keep) > MAX_TINY_ADDED:  # Gemma has thousands: keep the first ones and the template's
        used = set()
        for p in [out.get("post_processor")] + list((out.get("post_processor") or {}).get("processors", [])):
            for special in ((p or {}).get("special_tokens") or {}).values():
                used.update(special["ids"])
        keep = [t for i, t in enumerate(keep) if i < MAX_TINY_ADDED or t["id"] in used]
    for t in keep:
        if t["content"] in model["vocab"]:  # already a vocabulary token (<unk>): keep its id, as real files do
            remap[t["id"]] = model["vocab"][t["content"]]
        else:
            remap[t["id"]] = next_id
            next_id += 1
        added.append(dict(t, id=remap[t["id"]]))
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
