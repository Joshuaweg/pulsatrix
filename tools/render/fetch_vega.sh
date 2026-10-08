#!/usr/bin/env bash
# Downloads the Vega, Vega-Lite and vega-embed builds the HTML views load (VIZ-3), and 3Dmol.js for
# the structure page (PLM-5), at the pinned versions in include/pulsatrix/viz/html.hpp and
# protein_views.hpp, and checks their SHA-384 hashes. With them, pages
# can carry the scripts inline and work offline:
#
#   tools/render/fetch_vega.sh ~/.cache/pulsatrix/vega
#   pulsatrix_svg x.json -o x.html --inline-js ~/.cache/pulsatrix/vega
#
# Default directory: build/vega (pulsatrix_svg --offline looks for vega/ next to itself).
set -euo pipefail
dir="${1:-build/vega}"
mkdir -p "$dir"
fetch() {  # package version file sha384-hex
    local url="https://cdn.jsdelivr.net/npm/$1@$2/build/$3" out="$dir/$3"
    if [[ ! -f "$out" ]] || ! echo "$4  $out" | sha384sum -c --status; then
        curl -fsSL "$url" -o "$out.tmp"
        if ! echo "$4  $out.tmp" | sha384sum -c --status; then
            rm -f "$out.tmp"
            echo "fetch_vega.sh: $url does not match its pinned hash" >&2
            exit 1
        fi
        mv "$out.tmp" "$out"
    fi
    echo "$out ($1 $2)"
}
fetch vega 6.4.0 vega.min.js 54a75c26bdd968120931b55ca294c023f244b9491263aaa7c15bbd88bbb0d039d0f60435263b1f649857150020362e37
fetch vega-lite 6.4.3 vega-lite.min.js f7fef480d09f3aee86ef15ef91dade31fbaa004b286862555d5d8137f24b45791299c1af9ccaadb11c7c1d9b54580bc8
fetch vega-embed 7.3.0 vega-embed.min.js 32ecb5411c5815e36b035cd285cd4ab4de0e7e2a6447feb582df875aa379edcd59c6edce7b5e93b2296caee5558df28b
fetch 3dmol 2.5.5 3Dmol-min.js 3ac73361b95dbeb1e0b25afd7c5a7f8b81a22d27aec3d97e40896ff7d213c3cb283b05dca0679f94530b83e094fd7d5d
