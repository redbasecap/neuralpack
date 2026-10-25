// neuralpack browser decoder - core (no DOM, no WebGPU; runs in browsers and Node).
//
// Pipeline per tile:  z bytes --rANS--> z symbols --+medians--> z_hat
//                     z_hat --integer hyper-synthesis--> scales, means (bit-exact with Python)
//                     y bytes --rANS(indexes(scales))--> y symbols --+means--> y_hat
//                     y_hat --g_s (ONNX Runtime)--> pixels
//
// Everything that decides how the bitstream is parsed is integer arithmetic here, so the
// result is identical to the encoder's regardless of CPU, GPU or browser.

export const PRECISION = 16;
const BYPASS_PRECISION = 4;
const MAX_BYPASS_VAL = (1 << BYPASS_PRECISION) - 1;
const RANS_L_HI = 0; // RANS64_L = 2^31  => state < L  <=>  hi == 0 && lo < 2^31

// ------------------------------------------------------------------ CRC32 (zlib)
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes, seed = 0) {
  let c = (seed ^ 0xffffffff) >>> 0;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// ------------------------------------------------------------------ container
export const FLAG_GRAYSCALE = 1, FLAG_METADATA = 2, FLAG_INT_HYPER = 4;
export const HEADER_SIZE = 32;

export class NpkFormatError extends Error {}

export function parseNpk(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < HEADER_SIZE + 4) throw new NpkFormatError("file too short to be an .npk stream");
  if (bytes[0] !== 0x4e || bytes[1] !== 0x50 || bytes[2] !== 0x4b) throw new NpkFormatError("not an .npk file (bad magic)");
  if (bytes[3] !== 1) throw new NpkFormatError(`unsupported .npk version ${bytes[3]}`);
  const dv = new DataView(buffer, bytes.byteOffset);
  const body = bytes.subarray(0, bytes.length - 4);
  const crc = dv.getUint32(bytes.length - 4, true);
  if (crc32(body) !== crc) throw new NpkFormatError("CRC mismatch: the file is corrupted or truncated");
  const h = {
    modelId: bytes[4], quality: bytes[5], flags: bytes[6],
    width: dv.getUint32(8, true), height: dv.getUint32(12, true),
    tile: dv.getUint16(16, true), overlap: dv.getUint16(18, true),
    ntiles: dv.getUint32(20, true), fingerprint: dv.getUint32(24, true),
    metaLen: dv.getUint32(28, true),
  };
  h.grayscale = !!(h.flags & FLAG_GRAYSCALE);
  h.intHyper = !!(h.flags & FLAG_INT_HYPER);
  let pos = HEADER_SIZE + h.metaLen; // metadata (zlib JSON) is skipped in the browser
  const tiles = [];
  for (let i = 0; i < h.ntiles; i++) {
    if (pos + 8 > body.length) throw new NpkFormatError("truncated tile table");
    const ly = dv.getUint32(pos, true), lz = dv.getUint32(pos + 4, true);
    pos += 8;
    if (pos + ly + lz > body.length) throw new NpkFormatError("truncated tile payload");
    tiles.push({ y: bytes.subarray(pos, pos + ly), z: bytes.subarray(pos + ly, pos + ly + lz) });
    pos += ly + lz;
  }
  if (pos !== body.length) throw new NpkFormatError("unexpected trailing bytes");
  return { header: h, tiles };
}

export function tileGrid(width, height, tile, overlap) {
  if (tile <= 0 || (width <= tile && height <= tile)) return [{ x0: 0, y0: 0, x1: width, y1: height, ex0: 0, ey0: 0, ex1: width, ey1: height }];
  const stride = tile - 2 * overlap;
  const out = [];
  for (let y0 = 0; y0 < height; y0 += stride)
    for (let x0 = 0; x0 < width; x0 += stride) {
      const x1 = Math.min(x0 + stride, width), y1 = Math.min(y0 + stride, height);
      out.push({ x0, y0, x1, y1, ex0: Math.max(0, x0 - overlap), ey0: Math.max(0, y0 - overlap),
        ex1: Math.min(width, x1 + overlap), ey1: Math.min(height, y1 + overlap) });
    }
  return out;
}

// ------------------------------------------------------------------ rANS decoder (port of CompressAI's rans_interface.cpp + ryg_rans rans64.h)
export class RansDecoder {
  constructor(bytes) {
    if (bytes.length % 4 !== 0 || bytes.length < 8) throw new NpkFormatError("bad rANS stream length");
    this.words = new Uint32Array(bytes.length / 4);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let i = 0; i < this.words.length; i++) this.words[i] = dv.getUint32(i * 4, true);
    this.ptr = 2;
    this.lo = this.words[0];
    this.hi = this.words[1];
  }
  _renorm() {
    if (this.hi === RANS_L_HI && this.lo < 0x80000000) {
      this.hi = this.lo;
      this.lo = this.ptr < this.words.length ? this.words[this.ptr] : 0;
      this.ptr++;
    }
  }
  _advance(start, freq) {
    // x = freq * (x >> 16) + (x & 0xffff) - start, exactly, with x = hi*2^32 + lo < 2^63
    const qHi = this.hi >>> 16;
    const qLo = (((this.hi & 0xffff) << 16) | (this.lo >>> 16)) >>> 0;
    const c = freq * qLo + (this.lo & 0xffff) - start; // < 2^49, exact in a double
    const carry = Math.floor(c / 4294967296);
    this.lo = (c - carry * 4294967296) >>> 0;
    this.hi = freq * qHi + carry;
    this._renorm();
  }
  _getBits(n) {
    const val = this.lo & ((1 << n) - 1);
    this.lo = (((this.hi & ((1 << n) - 1)) << (32 - n)) | (this.lo >>> n)) >>> 0;
    this.hi = this.hi >>> n;
    this._renorm();
    return val;
  }
  /** indexes: per-symbol row into cdfs; cdfs: array of Int32Array rows; returns Int32Array of values. */
  decodeWithIndexes(indexes, cdfs, cdfLengths, offsets) {
    const n = indexes.length;
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const idx = indexes[i];
      const cdf = cdfs[idx];
      const len = cdfLengths[idx];
      const maxValue = len - 2;
      const cum = this.lo & 0xffff;
      // first j with cdf[j] > cum, then s = j - 1  (cdf[0] == 0, cdf[len-1] == 2^16)
      let s = 0;
      while (s + 1 < len && cdf[s + 1] <= cum) s++;
      this._advance(cdf[s], cdf[s + 1] - cdf[s]);
      let value = s;
      if (value === maxValue) {
        let val = this._getBits(BYPASS_PRECISION);
        let nBypass = val;
        while (val === MAX_BYPASS_VAL) {
          val = this._getBits(BYPASS_PRECISION);
          nBypass += val;
        }
        let raw = 0;
        for (let j = 0; j < nBypass; j++) raw += this._getBits(BYPASS_PRECISION) * 2 ** (j * BYPASS_PRECISION);
        value = Math.floor(raw / 2);
        if (raw % 2 === 1) value = -value - 1;
        else value += maxValue;
      }
      out[i] = value + offsets[idx];
    }
    return out;
  }
}

export function tablesFromJson(t) {
  const conv = (o) => ({ cdf: o.cdf.map((r) => Int32Array.from(r)), cdfLength: Int32Array.from(o.cdf_length), offset: Int32Array.from(o.offset) });
  const eb = conv(t.eb);
  eb.medians = Float32Array.from(t.eb.medians);
  const gc = conv(t.gc);
  gc.scaleTable = Float32Array.from(t.gc.scale_table);
  return { eb, gc };
}

// ------------------------------------------------------------------ Gaussian conditional indexes
export function buildIndexes(scales, scaleTable) {
  // CompressAI: scales = max(scales, table[0]); idx = L-1 - #{t in table[:-1] : scales <= t}
  //            = #{t in table[:-1] : scales > t}   (table is strictly increasing)
  const L = scaleTable.length;
  const out = new Uint8Array(scales.length);
  const t0 = scaleTable[0];
  for (let i = 0; i < scales.length; i++) {
    const v = scales[i] > t0 ? scales[i] : t0;
    let idx = 0;
    while (idx < L - 1 && v > scaleTable[idx]) idx++;
    out[i] = idx;
  }
  return out;
}

// ------------------------------------------------------------------ integer hyper-synthesis (CPU reference)
export const ACT_MAX = 32767;
const INT32_MIN = -2147483648, INT32_MAX = 2147483647;

export function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

export function roundShift(s, shift) {
  let r = shift > 0 ? Math.floor((s + 2 ** (shift - 1)) / 2 ** shift) : s;
  return r < INT32_MIN ? INT32_MIN : r > INT32_MAX ? INT32_MAX : r;
}

/** z_hat (Float32Array, C*H*W) -> Int32Array input activations. */
export function quantizeInput(zHat, inShift) {
  const scale = 2 ** inShift;
  const out = new Int32Array(zHat.length);
  for (let i = 0; i < zHat.length; i++) {
    let v = roundHalfEven(zHat[i] * scale);
    out[i] = v < -ACT_MAX ? -ACT_MAX : v > ACT_MAX ? ACT_MAX : v;
  }
  return out;
}

export class IntHyperCPU {
  /** hyper: parsed hyper.json; weights: Int16Array of all layers (little-endian file). */
  constructor(hyper, weights) {
    this.hyper = hyper;
    this.layers = hyper.layers.map((l) => ({ ...l, w: weights.subarray(l.weight_offset / 2, l.weight_offset / 2 + l.weight_count) }));
  }
  static outSize(l, h, w) {
    if (l.kind === "deconv") return [(h - 1) * l.stride - 2 * l.padding + l.k + l.output_padding, (w - 1) * l.stride - 2 * l.padding + l.k + l.output_padding];
    return [Math.floor((h + 2 * l.padding - l.k) / l.stride) + 1, Math.floor((w + 2 * l.padding - l.k) / l.stride) + 1];
  }
  /** a: Int32Array (C,H,W). Returns {q: Int32Array (Cout,Hout,Wout), h, w} of the last layer. */
  run(a, h, w) {
    const n = this.layers.length;
    for (let li = 0; li < n; li++) {
      const l = this.layers[li];
      const [ho, wo] = IntHyperCPU.outSize(l, h, w);
      const out = new Int32Array(l.out * ho * wo);
      const last = li === n - 1;
      const k = l.k, cin = l.in, W = l.w;
      for (let o = 0; o < l.out; o++) {
        const bias = l.bias[o];
        const wBase = o * cin * k * k;
        for (let oy = 0; oy < ho; oy++) {
          for (let ox = 0; ox < wo; ox++) {
            let s = 0;
            if (l.kind === "deconv") {
              // oy = iy*stride - pad + ky  =>  ky = oy + pad - iy*stride in [0,k)
              const iy0 = Math.max(0, Math.ceil((oy + l.padding - k + 1) / l.stride)), iy1 = Math.min(h - 1, Math.floor((oy + l.padding) / l.stride));
              const ix0 = Math.max(0, Math.ceil((ox + l.padding - k + 1) / l.stride)), ix1 = Math.min(w - 1, Math.floor((ox + l.padding) / l.stride));
              for (let c = 0; c < cin; c++) {
                const aBase = c * h * w, wc = wBase + c * k * k;
                for (let iy = iy0; iy <= iy1; iy++) {
                  const ky = oy + l.padding - iy * l.stride;
                  for (let ix = ix0; ix <= ix1; ix++) {
                    const kx = ox + l.padding - ix * l.stride;
                    s += W[wc + ky * k + kx] * a[aBase + iy * w + ix];
                  }
                }
              }
            } else {
              for (let c = 0; c < cin; c++) {
                const aBase = c * h * w, wc = wBase + c * k * k;
                for (let ky = 0; ky < k; ky++) {
                  const iy = oy * l.stride - l.padding + ky;
                  if (iy < 0 || iy >= h) continue;
                  for (let kx = 0; kx < k; kx++) {
                    const ix = ox * l.stride - l.padding + kx;
                    if (ix < 0 || ix >= w) continue;
                    s += W[wc + ky * k + kx] * a[aBase + iy * w + ix];
                  }
                }
              }
            }
            let v = roundShift(s + bias, l.shift);
            if (!last) {
              v = v < -ACT_MAX ? -ACT_MAX : v > ACT_MAX ? ACT_MAX : v;
              if (l.act === "leaky" && v < 0) v = Math.floor((v * this.hyper.leaky[0]) / 2 ** this.hyper.leaky[1]);
              else if (l.act === "relu" && v < 0) v = 0;
            }
            out[(o * ho + oy) * wo + ox] = v;
          }
        }
      }
      a = out; h = ho; w = wo;
    }
    return { q: a, h, w };
  }
}

/** Integer outputs -> float32 scales/means (one correctly rounded IEEE multiply, as in Python). */
export function hyperOutputs(q, h, w, hyper) {
  const last = hyper.layers[hyper.layers.length - 1];
  const step = Float32Array.from(last.out_scale);
  const nS = hyper.n_scales, plane = h * w;
  const scales = new Float32Array(nS * plane);
  const means = hyper.has_means ? new Float32Array(nS * plane) : null;
  for (let c = 0; c < nS; c++) {
    const st = step[c];
    for (let i = 0; i < plane; i++) scales[c * plane + i] = Math.fround(Math.fround(q[c * plane + i]) * st);
  }
  if (means) {
    for (let c = 0; c < nS; c++) {
      const st = step[nS + c];
      for (let i = 0; i < plane; i++) means[c * plane + i] = Math.fround(Math.fround(q[(nS + c) * plane + i]) * st);
    }
  }
  return { scales, means };
}

// ------------------------------------------------------------------ tile decoding
export function padTo(n, s) { return n + ((s - (n % s)) % s); }

/**
 * Decode one tile's latents. `hyperRun(a, h, w)` -> Promise<{q,h,w}> (CPU or GPU implementation).
 * Returns { yHat: Float32Array (M, hy, wy), hy, wy } ready for g_s.
 */
export async function decodeTileLatents(tile, eh, ew, tables, hyper, hyperRun, stats) {
  const t0 = performance.now();
  const ph = padTo(eh, 64), pw = padTo(ew, 64);
  const hz = ph / 64, wz = pw / 64, hy = hz * 4, wy = wz * 4;
  const N = tables.eb.medians.length;
  // --- z
  const zIdx = new Uint8Array(N * hz * wz);
  for (let c = 0; c < N; c++) zIdx.fill(c, c * hz * wz, (c + 1) * hz * wz);
  const zSym = new RansDecoder(tile.z).decodeWithIndexes(zIdx, tables.eb.cdf, tables.eb.cdfLength, tables.eb.offset);
  const zHat = new Float32Array(zSym.length);
  for (let c = 0; c < N; c++) { const m = tables.eb.medians[c]; for (let i = c * hz * wz; i < (c + 1) * hz * wz; i++) zHat[i] = Math.fround(zSym[i] + m); }
  const t1 = performance.now();
  // --- integer hyper-synthesis
  const { q, h, w } = await hyperRun(quantizeInput(zHat, hyper.in_shift), hz, wz);
  if (h !== hy || w !== wy) throw new Error(`hyper-synthesis produced ${h}x${w}, expected ${hy}x${wy}`);
  const { scales, means } = hyperOutputs(q, hy, wy, hyper);
  const t2 = performance.now();
  // --- y
  const indexes = buildIndexes(scales, tables.gc.scaleTable);
  const ySym = new RansDecoder(tile.y).decodeWithIndexes(indexes, tables.gc.cdf, tables.gc.cdfLength, tables.gc.offset);
  const yHat = new Float32Array(ySym.length);
  if (means) for (let i = 0; i < ySym.length; i++) yHat[i] = Math.fround(ySym[i] + means[i]);
  else for (let i = 0; i < ySym.length; i++) yHat[i] = ySym[i];
  const t3 = performance.now();
  if (stats) { stats.z += t1 - t0; stats.hyper += t2 - t1; stats.y += t3 - t2; }
  return { yHat, hy, wy, indexesCrc: crc32(indexes), ySymCrc: crc32(new Uint8Array(ySym.buffer)), qCrc: crc32(new Uint8Array(q.buffer)) };
}
