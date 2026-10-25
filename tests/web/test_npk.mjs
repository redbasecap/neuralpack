// Node test for the browser decoder's CPU path (no WebGPU, no ONNX): the parsed container,
// the rANS decoder and the integer hyper-synthesis must reproduce the Python encoder bit for bit.
// Usage: node tests/web/test_npk.mjs GOLDEN_DIR MODELS_DIR
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
globalThis.performance ??= performance;
const { parseNpk, tileGrid, tablesFromJson, decodeTileLatents, IntHyperCPU } = await import(new URL("../../src/neuralpack/web/npk.js", import.meta.url));

const [goldenDir, modelsDir] = process.argv.slice(2);
const golden = JSON.parse(readFileSync(join(goldenDir, "golden.json"), "utf8"));
const mdir = join(modelsDir, "mean-scale-q3");
const manifest = JSON.parse(readFileSync(join(mdir, "manifest.json"), "utf8"));
const tables = tablesFromJson(JSON.parse(readFileSync(join(mdir, "tables.json"), "utf8")));
const hyper = JSON.parse(readFileSync(join(mdir, "hyper.json"), "utf8"));
const wbuf = readFileSync(join(mdir, "hyper_weights.bin"));
const weights = new Int16Array(wbuf.buffer, wbuf.byteOffset, wbuf.byteLength / 2);
const cpu = new IntHyperCPU(hyper, weights);

let failures = 0;
for (const [name, g] of Object.entries(golden)) {
  const buf = readFileSync(join(goldenDir, g.file));
  const { header, tiles } = parseNpk(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  if (header.fingerprint !== manifest.fingerprint || header.fingerprint !== g.fingerprint) { console.log(`${name}: fingerprint mismatch`); failures++; }
  const grid = tileGrid(header.width, header.height, header.tile, header.overlap);
  if (grid.length !== g.tiles.length) { console.log(`${name}: tile count ${grid.length} != ${g.tiles.length}`); failures++; continue; }
  const stats = { z: 0, hyper: 0, y: 0 };
  for (let i = 0; i < grid.length; i++) {
    const t = grid[i];
    const r = await decodeTileLatents(tiles[i], t.ey1 - t.ey0, t.ex1 - t.ex0, tables, hyper, async (a, h, w) => cpu.run(a, h, w), stats);
    const e = g.tiles[i];
    const ok = r.qCrc === e.q_crc && r.indexesCrc === e.indexes_crc && r.ySymCrc === e.ysym_crc && r.hy === e.hy && r.wy === e.wy;
    if (!ok) { failures++; console.log(`${name} tile ${i}: MISMATCH q ${r.qCrc === e.q_crc} idx ${r.indexesCrc === e.indexes_crc} ysym ${r.ySymCrc === e.ysym_crc} shape ${r.hy}x${r.wy} vs ${e.hy}x${e.wy}`); }
  }
  console.log(`${name}: ${grid.length} tile(s) ${failures ? "checked" : "bit-exact"}  (z ${stats.z.toFixed(0)} ms, hyper ${stats.hyper.toFixed(0)} ms, y ${stats.y.toFixed(0)} ms)`);
}
if (failures) { console.log(`FAILED: ${failures}`); process.exit(1); }
console.log("OK: JavaScript decoder reproduces the Python encoder bit for bit");
