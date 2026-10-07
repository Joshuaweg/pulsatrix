// Checks the Vega-Lite charts in pulsatrix HTML pages (VIZ-3): each page's spec is compiled with
// Vega-Lite and rendered headlessly with Vega, and any error or warning fails the check. The
// rendered SVG is written next to the page for inspection.
//
//   docker run --rm -v "$PWD":/w -w /w node:22-slim sh -c \
//     "cd /tmp && npm install --silent vega@6.4.0 vega-lite@6.4.3 >/dev/null && cd /w && \
//      NODE_PATH=/tmp/node_modules node tools/render/validate_vega.mjs page1.html page2.html"
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

// Vega 6 is ES modules only: resolve the installed packages, then import them.
const require = createRequire(process.env.NODE_PATH + "/");
const load = async (name) => import(pathToFileURL(require.resolve(name)).href);
const vega = await load("vega");
const vegaLite = await load("vega-lite");

let failed = 0;
for (const page of process.argv.slice(2)) {
  const html = readFileSync(page, "utf8");
  const match = html.match(/<script type="application\/json" id="spec">([\s\S]*?)<\/script>/);
  if (!match) {
    console.log(`${page}: no chart (plain HTML)`);
    continue;
  }
  const warnings = [];
  const logger = {
    level() { return this; },
    // Zoom and pan listen on the window, which a headless render doesn't have: expected, not a defect.
    warn: (...m) => { const t = m.join(" "); if (!t.includes("Can not resolve event source: window")) warnings.push(t); return logger; },
    info: () => logger, debug: () => logger, error: (...m) => { warnings.push("error: " + m.join(" ")); return logger; },
  };
  try {
    const spec = JSON.parse(match[1]);
    const compiled = vegaLite.compile(spec, { logger }).spec;
    const view = new vega.View(vega.parse(compiled), { renderer: "none", logger });
    const svg = await view.toSVG();
    writeFileSync(page.replace(/\.html$/, ".rendered.svg"), svg);
    if (warnings.length) {
      failed++;
      console.log(`${page}: WARN ${warnings.join(" | ")}`);
    } else {
      console.log(`${page}: ok (${svg.length} bytes of SVG)`);
    }
  } catch (e) {
    failed++;
    console.log(`${page}: FAIL ${e.message}`);
  }
}
process.exit(failed ? 1 : 0);
