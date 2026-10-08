"""Downloads Swiss-Prot proteins with per-residue annotations for probes and SAE features (PLM-8):

    python3 tools/plm/fetch_swissprot_annotations.py ~/.cache/pulsatrix/swissprot

writes annotated.jsonl into the directory: one protein per line,
    {"accession": "P62593", "sequence": "MSIQ...", "features": [{"type": "Helix", "start": 27, "end": 40}, ...]}
with positions numbered from 1, inclusive. Proteins are reviewed (Swiss-Prot), 50 to 500 residues
long, and have a 3D structure, so their helix, strand and turn features come from structures.
The feature types kept are the per-residue ones: secondary structure, binding and active sites,
metal binding, disulfide bonds, modified residues, signal peptides, transmembrane spans, domains,
motifs, zinc fingers, coiled coils and compositional bias.

UniProt's stream is ordered by accession, which clumps organisms, so this lists every matching
accession, draws a seeded random sample of --count (default 3000), and fetches those. Proteins
with a nonstandard residue are dropped. Standard library only.
"""

import argparse
import concurrent.futures
import json
import os
import random
import sys
import urllib.parse
import urllib.request

FIELDS = ("accession,sequence,ft_helix,ft_strand,ft_turn,ft_binding,ft_act_site,ft_site,ft_disulfid,ft_mod_res,ft_signal,"
          "ft_transmem,ft_domain,ft_motif,ft_zn_fing,ft_coiled,ft_compbias")
QUERY = "(reviewed:true) AND (structure_3d:true) AND (length:[50 TO 500])"


def accessions():
    """Every matching accession, streamed as a plain list."""
    url = "https://rest.uniprot.org/uniprotkb/stream?format=list&query=" + urllib.parse.quote(QUERY)
    with urllib.request.urlopen(url, timeout=600) as response:
        return [line.decode().strip() for line in response if line.strip()]


def fetch(batch):
    url = ("https://rest.uniprot.org/uniprotkb/accessions?format=json&fields=%s&accessions=%s"
           % (FIELDS, ",".join(batch)))
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = json.load(response)
    except Exception as e:  # a dropped connection: skip the batch
        print("  batch at %s: %s" % (batch[0], e), file=sys.stderr)
        return []
    out = []
    for r in data.get("results", []):
        features = []
        for f in r.get("features", []):
            start, end = f["location"]["start"].get("value"), f["location"]["end"].get("value")
            if start is None or end is None:
                continue  # an uncertain position
            features.append({"type": f["type"], "start": start, "end": end})
        out.append({"accession": r["primaryAccession"], "sequence": r["sequence"]["value"], "features": features})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--count", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    every = accessions()
    print("%d matching proteins" % len(every), file=sys.stderr)
    chosen = random.Random(a.seed).sample(every, min(a.count, len(every)))
    batches = [chosen[i:i + 100] for i in range(0, len(chosen), 100)]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(fetch, batches):
            records.extend(r for r in result if set(r["sequence"]) <= set("ACDEFGHIKLMNPQRSTVWY"))
    with open(os.path.join(a.out, "annotated.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print("wrote %d proteins" % len(records))


if __name__ == "__main__":
    main()
