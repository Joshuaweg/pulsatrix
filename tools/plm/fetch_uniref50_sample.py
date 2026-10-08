"""Downloads a sample of UniRef50 cluster representatives for masked-LM training (PLM-7):

    python3 tools/plm/fetch_uniref50_sample.py ~/.cache/pulsatrix/uniref

writes train.fasta and valid.fasta (a seeded 95/5 split) into the directory.

UniProt's FASTA file is sorted longest first, and its REST stream by accession, which clumps
sequences by organism, so neither file's start is a fair sample. This takes up to --per-prefix
representatives of 50 to 1022 residues (ESM-2's crop) from each of several hundred accession
prefixes across the namespace, then shuffles them. It is a broad sample, not a uniform one: the
prefixes are equally weighted however many clusters they hold. Standard library only.
"""

import argparse
import itertools
import os
import random
import string
import sys
import urllib.parse
import urllib.request

QUERY = "(identity:0.5) AND (length:[50 TO 1022]) AND (id:UniRef50_{prefix}*)"


def prefixes():
    # TrEMBL accessions A0A + 7 characters: the 4th and 5th vary.
    for a, b in itertools.product(string.digits, string.digits + string.ascii_uppercase):
        yield "A0A" + a + b
    # Swiss-Prot style accessions: O, P and Q followed by a digit and a character.
    for a, b, c in itertools.product("OPQ", string.digits, string.digits + "ABCDEFGHIJ"):
        yield a + b + c


def fetch(prefix, limit):
    url = "https://rest.uniprot.org/uniref/stream?format=fasta&query=" + urllib.parse.quote(QUERY.format(prefix=prefix))
    records, header, seq = [], None, []
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            for raw in response:
                line = raw.decode().strip()
                if line.startswith(">"):
                    if header:
                        records.append((header, "".join(seq)))
                        if len(records) >= limit:
                            break
                    header, seq = line[1:], []
                elif line:
                    seq.append(line)
            else:
                if header:
                    records.append((header, "".join(seq)))
    except Exception as e:  # a prefix with no entries, or a dropped connection: skip it
        print("  %s: %s" % (prefix, e), file=sys.stderr)
    return records[:limit]


def write(path, records):
    with open(path, "w") as f:
        for header, seq in records:
            f.write(">" + header + "\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--per-prefix", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seen, records = set(), []
    for i, prefix in enumerate(prefixes()):
        for header, seq in fetch(prefix, a.per_prefix):
            key = header.split()[0]
            if key not in seen:
                seen.add(key)
                records.append((header, seq))
        if i % 25 == 0:
            print("%d prefixes, %d sequences" % (i + 1, len(records)), file=sys.stderr)
    random.Random(a.seed).shuffle(records)
    split = len(records) * 95 // 100
    write(os.path.join(a.out, "train.fasta"), records[:split])
    write(os.path.join(a.out, "valid.fasta"), records[split:])
    print("wrote %d training and %d validation sequences" % (split, len(records) - split))


if __name__ == "__main__":
    main()
