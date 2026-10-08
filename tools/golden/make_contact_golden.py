"""Writes golden values for PLM-4: residue distances read by biotite, and an ESM-2 model's
predicted contacts and their precision, to compare pulsatrix's structure reader, ContactPredictor
and ContactPrecision against (tests/protein_structure_test.cpp, tests/protein_contacts_test.cpp).

    python3 tools/golden/make_contact_golden.py --structures ~/.cache/pulsatrix/golden/structures
    python3 tools/golden/make_contact_golden.py --structures DIR \
        --model ~/.cache/pulsatrix/golden/esm2_t6_8M_UR50D --proteins 1UBQ:A 1BTL:A 2LZM:A

The first writes structures_golden.safetensors into DIR, for every <id>.cif there (each has an
<id>.pdb beside it), for each protein chain c of the first model:
    beta.<id>.<c>      F32 [L, L]   Cβ distances (Cα for glycine), NaN where an atom is missing
    alpha.<id>.<c>     F32 [L, L]   Cα distances
    virtual.<id>.<c>   F32 [L, L]   distances between Cβ placed from the backbone (ESM's `extend`)
with metadata "sequences": {"<id>.<c>": sequence}. Residues are biotite's amino acids that have a
Cα, with author chain ids and numbering, and the first of each atom's alternate locations.

The second also writes contacts_golden.safetensors into the model directory, for each protein:
    contacts.<id>.<c>  F32 [L, L]   EsmForMaskedLM.predict_contacts on the chain's sequence
    precision.<id>.<c> F64 [3, 3]   rows short, medium, long; columns P@L, P@L/2, P@L/5, against
                                    Cβ contacts under 8 Å
with metadata "proteins" (the list), "esm_precisions" (the same from ESM's own
compute_precisions, for comparison), the model and the versions. Not run by CI. Needs torch,
transformers, safetensors, numpy and biotite.
"""

import argparse
import glob
import json
import os

import biotite
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import torch
import transformers
from safetensors.numpy import save_file
from transformers import AutoTokenizer, EsmForMaskedLM

RANGES = [(6, 12), (12, 24), (24, None)]


def extend(a, b, c, length, angle, dihedral):
    """ESM's (and trRosetta's) placement of a fourth atom from three."""
    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [length * np.cos(angle), length * np.sin(angle) * np.cos(dihedral), -length * np.sin(angle) * np.sin(dihedral)]
    return c + sum(mi * di for mi, di in zip(m, d))


def chains(path):
    """{chain: (sequence, {atom name: [L, 3] coordinates, NaN where missing})} for the first model."""
    atoms = pdbx.get_structure(pdbx.CIFFile.read(path), model=1, use_author_fields=True, altloc="first")
    atoms = atoms[struc.filter_amino_acids(atoms) & ((atoms.hetero == False) | (atoms.res_name == "MSE"))]  # noqa: E712
    out = {}
    for chain_id in dict.fromkeys(atoms.chain_id):
        chain = atoms[atoms.chain_id == chain_id]
        starts = struc.get_residue_starts(chain)
        residues = [chain[s:e] for s, e in zip(starts, list(starts[1:]) + [len(chain)])]
        residues = [r for r in residues if "CA" in r.atom_name]
        if not residues:
            continue
        sequence = "".join(code(r.res_name[0]) for r in residues)
        coords = {}
        for name in ("N", "CA", "C", "CB"):
            coords[name] = np.array([r.coord[r.atom_name == name][0] if name in r.atom_name else [np.nan] * 3 for r in residues],
                                    dtype=np.float64)
        coords["GLY"] = np.array([r.res_name[0] == "GLY" for r in residues])
        out[chain_id] = (sequence, coords)
    return out


ONE_LETTER = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
              "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
              "MSE": "M", "SEC": "U", "PYL": "O"}


def code(name):
    return ONE_LETTER.get(name, "X")


def distances(points):
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    return d.astype(np.float32)


def residue_points(coords):
    beta = np.where(coords["GLY"][:, None], coords["CA"], coords["CB"])
    virtual = extend(coords["C"], coords["N"], coords["CA"], 1.522, 1.927, -2.143)
    return {"beta": beta, "alpha": coords["CA"], "virtual": virtual}


def precision(pred, truth, minsep, maxsep, top):
    """pulsatrix's definition: the top `top` known pairs i < j in range, ties in (i, j) order, and
    missing pairs counted as wrong."""
    L = pred.shape[0]
    i, j = np.triu_indices(L, minsep)
    keep = ~np.isnan(truth[i, j])
    if maxsep is not None:
        keep &= (j - i) < maxsep
    i, j = i[keep], j[keep]
    order = np.argsort(-pred[i, j], kind="stable")[:top]
    return float(truth[i[order], j[order]].sum() / top) if top > 0 else float("nan")


def esm_compute_precisions(predictions, targets, minsep=6, maxsep=None):
    """ESM's compute_precisions (esm/examples/contact_prediction.ipynb), for comparison."""
    predictions, targets = predictions.unsqueeze(0), targets.unsqueeze(0)
    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen)
    sep = (seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)).unsqueeze(0)
    valid_mask = (sep >= minsep) & (targets >= 0)
    if maxsep is not None:
        valid_mask &= sep < maxsep
    src_lengths = torch.full([batch_size], seqlen, dtype=torch.long)
    predictions = predictions.masked_fill(~valid_mask, float("-inf"))
    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper, targets_upper = predictions[:, x_ind, y_ind], targets[:, x_ind, y_ind]
    topk = seqlen
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = torch.nn.functional.pad(topk_targets, [0, topk - topk_targets.size(1)])
    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)
    gather_indices = (torch.arange(0.1, 1.1, 0.1).unsqueeze(0) * src_lengths.unsqueeze(1)).type(torch.long) - 1
    binned = cumulative_dist.gather(1, gather_indices) / (gather_indices + 1).type_as(cumulative_dist)
    return [binned[0, 9].item(), binned[0, 4].item(), binned[0, 1].item()]


def write_structures(directory):
    out, sequences = {}, {}
    for path in sorted(glob.glob(os.path.join(directory, "*.cif"))):
        entry = os.path.splitext(os.path.basename(path))[0]
        for chain_id, (sequence, coords) in chains(path).items():
            key = f"{entry}.{chain_id}"
            sequences[key] = sequence
            for kind, points in residue_points(coords).items():
                out[f"{kind}.{key}"] = distances(points)
    save_file(out, os.path.join(directory, "structures_golden.safetensors"),
              metadata={"sequences": json.dumps(sequences), "biotite": biotite.__version__})
    print(directory, len(sequences), "chains")
    return sequences


def write_contacts(model_dir, structures, proteins):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = EsmForMaskedLM.from_pretrained(model_dir, attn_implementation="eager", torch_dtype=torch.float32).eval()
    out, esm = {}, {}
    for protein in proteins:
        entry, chain_id = protein.split(":")
        sequence, coords = chains(os.path.join(structures, entry + ".cif"))[chain_id]
        ids = tok(sequence, return_tensors="pt")
        with torch.no_grad():
            contacts = model.predict_contacts(ids["input_ids"], ids["attention_mask"])[0].float().numpy()
        d = distances(residue_points(coords)["beta"])
        truth = np.where(np.isnan(d), np.nan, (d < 8.0).astype(np.float32))
        L = len(sequence)
        key = f"{entry}.{chain_id}"
        out[f"contacts.{key}"] = np.ascontiguousarray(contacts, dtype=np.float32)
        out[f"precision.{key}"] = np.array([[precision(contacts, truth, lo, hi, top) for top in (L, L // 2, L // 5)] for lo, hi in RANGES])
        targets = torch.from_numpy(np.where(np.isnan(truth), -1, truth).astype(np.int64))
        esm[key] = [esm_compute_precisions(torch.from_numpy(contacts), targets, lo, hi) for lo, hi in RANGES]
        print(key, L, "long P@L", out[f"precision.{key}"][2][0], "ESM's", esm[key][2][0])
    metadata = {"model": os.path.basename(os.path.normpath(model_dir)), "proteins": json.dumps(proteins), "esm_precisions": json.dumps(esm),
                "torch": torch.__version__, "transformers": transformers.__version__, "biotite": biotite.__version__}
    save_file(out, os.path.join(model_dir, "contacts_golden.safetensors"), metadata=metadata)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structures", required=True)
    p.add_argument("--model")
    p.add_argument("--proteins", nargs="*", default=[])
    a = p.parse_args()
    write_structures(a.structures)
    if a.model:
        write_contacts(a.model, a.structures, a.proteins)


if __name__ == "__main__":
    main()
