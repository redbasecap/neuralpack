// WebGPU implementation of the integer hyper-synthesis. Bit-exact with IntHyperCPU (npk.js)
// and with the Python encoder: int16 weights x int16 activations, products in i32, sums in a
// carry-tracked (hi, lo) 64-bit pair, 64-bit bias, round-shift with int32 saturation.

const WG = 8;

function shaderSource(kind) {
  const gather = kind === "deconv" ? `
      // transposed conv: oy = iy*stride - pad + ky  =>  iy in [ceil((oy+pad-k+1)/stride), floor((oy+pad)/stride)]
      let oyp = i32(oy) + i32(P.pad); let oxp = i32(ox) + i32(P.pad);
      let st = i32(P.stride); let kk = i32(P.k);
      var iy0: i32 = 0; if (oyp - kk + 1 > 0) { iy0 = (oyp - kk + 1 + st - 1) / st; }
      var ix0: i32 = 0; if (oxp - kk + 1 > 0) { ix0 = (oxp - kk + 1 + st - 1) / st; }
      let iy1 = min(oyp / st, i32(P.h_in) - 1); let ix1 = min(oxp / st, i32(P.w_in) - 1);
      for (var c = 0u; c < P.c_in; c++) {
        let aBase = c * P.h_in * P.w_in; let wBase = (o * P.c_in + c) * P.k * P.k;
        for (var iy = iy0; iy <= iy1; iy++) {
          let ky = u32(oyp - iy * st);
          for (var ix = ix0; ix <= ix1; ix++) {
            let kx = u32(oxp - ix * st);
            let p = W[wBase + ky * P.k + kx] * A[aBase + u32(iy) * P.w_in + u32(ix)];
            let pu = bitcast<u32>(p); let nlo = lo + pu;
            hi = hi + select(0, -1, p < 0) + i32(select(0u, 1u, nlo < lo)); lo = nlo;
          }
        }
      }` : `
      for (var c = 0u; c < P.c_in; c++) {
        let aBase = c * P.h_in * P.w_in; let wBase = (o * P.c_in + c) * P.k * P.k;
        for (var ky = 0u; ky < P.k; ky++) {
          let iy = i32(oy * P.stride) - i32(P.pad) + i32(ky);
          if (iy < 0 || iy >= i32(P.h_in)) { continue; }
          for (var kx = 0u; kx < P.k; kx++) {
            let ix = i32(ox * P.stride) - i32(P.pad) + i32(kx);
            if (ix < 0 || ix >= i32(P.w_in)) { continue; }
            let p = W[wBase + ky * P.k + kx] * A[aBase + u32(iy) * P.w_in + u32(ix)];
            let pu = bitcast<u32>(p); let nlo = lo + pu;
            hi = hi + select(0, -1, p < 0) + i32(select(0u, 1u, nlo < lo)); lo = nlo;
          }
        }
      }`;
  return `
    struct Params { h_in: u32, w_in: u32, c_in: u32, c_out: u32, h_out: u32, w_out: u32, k: u32, stride: u32,
                    pad: u32, shift: u32, act: u32, last: u32, leaky_num: i32, leaky_shift: u32, act_max: i32, _p: u32 };
    @group(0) @binding(0) var<uniform> P: Params;
    @group(0) @binding(1) var<storage, read> W: array<i32>;
    @group(0) @binding(2) var<storage, read> BHI: array<i32>;
    @group(0) @binding(3) var<storage, read> BLO: array<u32>;
    @group(0) @binding(4) var<storage, read> A: array<i32>;
    @group(0) @binding(5) var<storage, read_write> O: array<i32>;

    @compute @workgroup_size(${WG}, ${WG}, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let ox = gid.x; let oy = gid.y; let o = gid.z;
      if (ox >= P.w_out || oy >= P.h_out || o >= P.c_out) { return; }
      var hi: i32 = 0; var lo: u32 = 0u;
      ${gather}
      // + bias (64-bit)
      let blo = BLO[o]; let nlo = lo + blo; hi = hi + BHI[o] + i32(select(0u, 1u, nlo < lo)); lo = nlo;
      // round shift with int32 saturation
      var r: i32;
      if (P.shift == 0u) {
        if (hi > 0 || (hi == 0 && lo >= 0x80000000u)) { r = 2147483647; }
        else if (hi < -1 || (hi == -1 && lo < 0x80000000u)) { r = -2147483647 - 1; }
        else { r = bitcast<i32>(lo); }
      } else {
        let half = 1u << (P.shift - 1u); let nlo2 = lo + half; hi = hi + i32(select(0u, 1u, nlo2 < lo)); lo = nlo2;
        let lim = i32(1u << (P.shift - 1u));
        if (hi >= lim) { r = 2147483647; } else if (hi < -lim) { r = -2147483647 - 1; }
        else { r = bitcast<i32>((bitcast<u32>(hi) << (32u - P.shift)) | (lo >> P.shift)); }
      }
      if (P.last == 0u) {
        r = clamp(r, -P.act_max, P.act_max);
        if (P.act == 1u && r < 0) { r = (r * P.leaky_num) >> P.leaky_shift; }
        else if (P.act == 2u && r < 0) { r = 0; }
      }
      O[(o * P.h_out + oy) * P.w_out + ox] = r;
    }`;
}

export class IntHyperGPU {
  static async create(hyper, weights) {
    if (!navigator.gpu) throw new Error("WebGPU not available");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const need = Math.max(...hyper.layers.map((l) => l.weight_count * 4), 256 << 20);
    const device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, Math.max(need, 128 << 20)),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, Math.max(need, 256 << 20)) } });
    return new IntHyperGPU(device, hyper, weights);
  }
  constructor(device, hyper, weights) {
    this.device = device; this.hyper = hyper;
    this.pipelines = {};
    this.layers = hyper.layers.map((l) => {
      const w32 = Int32Array.from(weights.subarray(l.weight_offset / 2, l.weight_offset / 2 + l.weight_count));
      const bhi = new Int32Array(l.out), blo = new Uint32Array(l.out);
      for (let i = 0; i < l.out; i++) { const b = l.bias[i]; const hi = Math.floor(b / 4294967296); bhi[i] = hi; blo[i] = (b - hi * 4294967296) >>> 0; }
      return { ...l, wBuf: this._buf(w32, GPUBufferUsage.STORAGE), bhiBuf: this._buf(bhi, GPUBufferUsage.STORAGE), bloBuf: this._buf(blo, GPUBufferUsage.STORAGE),
        pipeline: this._pipeline(l.kind) };
    });
  }
  _buf(arr, usage) {
    const b = this.device.createBuffer({ size: Math.max(16, arr.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return b;
  }
  _pipeline(kind) {
    if (!this.pipelines[kind]) {
      const module = this.device.createShaderModule({ code: shaderSource(kind) });
      this.pipelines[kind] = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    }
    return this.pipelines[kind];
  }
  /** a: Int32Array (C,H,W) -> Promise<{q: Int32Array, h, w}> */
  async run(a, h, w) {
    const dev = this.device;
    let inBuf = this._buf(a, GPUBufferUsage.STORAGE);
    const temps = [inBuf];
    let outBuf = null, ho = h, wo = w;
    const n = this.layers.length;
    const enc = dev.createCommandEncoder();
    for (let li = 0; li < n; li++) {
      const l = this.layers[li];
      if (l.kind === "deconv") { ho = (h - 1) * l.stride - 2 * l.padding + l.k + l.output_padding; wo = (w - 1) * l.stride - 2 * l.padding + l.k + l.output_padding; }
      else { ho = Math.floor((h + 2 * l.padding - l.k) / l.stride) + 1; wo = Math.floor((w + 2 * l.padding - l.k) / l.stride) + 1; }
      outBuf = dev.createBuffer({ size: l.out * ho * wo * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      temps.push(outBuf);
      const params = new ArrayBuffer(64); const u = new Uint32Array(params); const i = new Int32Array(params);
      u[0] = h; u[1] = w; u[2] = l.in; u[3] = l.out; u[4] = ho; u[5] = wo; u[6] = l.k; u[7] = l.stride; u[8] = l.padding; u[9] = l.shift;
      u[10] = l.act === "leaky" ? 1 : l.act === "relu" ? 2 : 0; u[11] = li === n - 1 ? 1 : 0; i[12] = this.hyper.leaky[0]; u[13] = this.hyper.leaky[1]; i[14] = this.hyper.act_max;
      const pBuf = dev.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(pBuf, 0, params); temps.push(pBuf);
      const bind = dev.createBindGroup({ layout: l.pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: pBuf } }, { binding: 1, resource: { buffer: l.wBuf } }, { binding: 2, resource: { buffer: l.bhiBuf } },
        { binding: 3, resource: { buffer: l.bloBuf } }, { binding: 4, resource: { buffer: inBuf } }, { binding: 5, resource: { buffer: outBuf } }] });
      const pass = enc.beginComputePass();
      pass.setPipeline(l.pipeline); pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(wo / WG), Math.ceil(ho / WG), l.out);
      pass.end();
      inBuf = outBuf; h = ho; w = wo;
    }
    const read = dev.createBuffer({ size: outBuf.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(outBuf, 0, read, 0, outBuf.size);
    dev.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const q = new Int32Array(read.getMappedRange().slice(0));
    read.unmap(); read.destroy();
    for (const t of temps) t.destroy();
    return { q, h, w };
  }
}
