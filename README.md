# neuralpack · `npk`

**Send the idea of a picture, not its pixels.**

`npk` is a neural image (and experimental video) codec built on a *shared learned prior*. Sender and
receiver both hold the same pretrained model, a learned dictionary of how photographs are
structured. A `.npk` file carries nothing but entropy‑coded indexes into that dictionary, so it is
tiny. When the receiver opens it, their copy of the model turns the indexes back into a
full‑resolution picture.

```
16.5 MB PNG (9.4 megapixels)  ──npk q5──▶  0.84 MB  ──open──▶  full-resolution picture, 34.7 dB PSNR
                                              │
                                              └─ JPEG needs 1.77 MB for the same quality
```

| | |
|---|---|
| **Smaller** | At equal file size: **+4.1 to +5.4 dB PSNR vs JPEG**, **+2.0 to +2.6 dB vs WebP** (Kodak set, 24 photos). JPEG needs **~2× the bytes** to match. |
| **Any size** | Tiled coding with halos: 100 megapixels on a laptop, no visible seams. |
| **Any device** | Bitstreams are device‑independent by construction. Encode on an NVIDIA GPU, open on a MacBook or a Raspberry Pi. |
| **Honest** | A container with a CRC, a model fingerprint, EXIF/ICC pass‑through, and loud failures instead of silent garbage. |
| **Simple** | `pip install .` then `npk encode photo.jpg` and `npk decode photo.npk`. One Python call each. |

---

## Why this works (and what it is not)

Classic codecs rotate pixels into a fixed basis (the DCT in JPEG, a wavelet in JPEG 2000, a
*fixed* linear algebra "rotation") and then throw away small coefficients. The basis knows nothing
about photographs; everything it knows must be sent again in every file.

A learned codec replaces that fixed rotation with a **non‑linear transform trained on millions of
photographs** (`g_a`, the analysis transform). It is the generalisation of the Karhunen‑Loève
transform, the optimal linear rotation, to a deep network. Its output latents are quantised and
coded with a **learned probability model**: a second, much smaller latent (`z`, the *hyperprior*)
tells the arithmetic coder, for every coefficient, how likely each value is. Then the synthesis
transform `g_s` paints the picture back.

Everything the model knows about edges, skin, foliage, sky, fabric, noise statistics, lives in
its weights, **once, on both sides**. The file only has to say *which* of the structures the model
already knows are present, where, and how strong. That is why a 36‑byte header plus a few hundred
kilobytes of indexes can reconstruct a 16 MB photograph.

What it is **not**: lossless. Information theory does not allow a 20:1 lossless ratio on real
photographs (their sensor noise alone carries more entropy than that). `npk` is *perceptually*
faithful: at quality 5 the mean error is below one grey level and MS‑SSIM is ~18 dB; at quality 8
it is visually transparent. Pixel values are not bit‑identical to the source, and decodes on two
different GPUs may differ by ±1 grey level in a few pixels (the *bitstream* itself is parsed
identically everywhere, see [Determinism](#determinism-across-devices)).

## Install

```bash
git clone https://github.com/nickspieser/neuralpack && cd neuralpack
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # torch, compressai, pillow, numpy (+ pytest, msssim, ruff)
npk models --download            # optional: pre-fetch all weights (~450 MB); otherwise lazy, per quality level
```

Python ≥ 3.10. Runs on CUDA, Apple Silicon (MPS) and plain CPU. `ffmpeg` on `PATH` enables video.

## Use

```bash
npk encode holiday.jpg                     # -> holiday.npk     (default: model mean-scale, quality 5)
npk encode holiday.jpg -q 3                # smaller (~0.3 bpp), still good
npk encode shoot/*.tif -o packed/ -q 6     # many files into a directory
npk decode holiday.npk                     # -> holiday.png (lossless container for the decoded pixels)
npk decode holiday.npk -f jpg              # or jpg / webp / tiff
npk open holiday.npk                       # decode to a temp file and show it in the system viewer
npk info holiday.npk                       # what is inside: model, quality, tiles, bytes per part
npk bench holiday.jpg -q 1,3,5,8           # rate/quality table vs JPEG & WebP at *equal file size*
npk encode clip.mp4 && npk decode clip.npkv # experimental intra-frame video (see below)
```

```python
from PIL import Image
import neuralpack as npk

res = npk.encode_image(Image.open("holiday.jpg"), quality=5)   # res.data is the .npk byte string
print(res.bytes, res.bpp, res.tiles, res.seconds)

img = npk.decode_bytes(res.data)                                # PIL image, full resolution
img.save("holiday.png")

npk.encode_file("holiday.jpg", "holiday.npk", quality=3, model="hyperprior")
npk.decode_file("holiday.npk", "holiday.jpg")
```

### Quality levels

| quality | typical bpp (photos) | character |
|---|---|---|
| 1–2 | 0.08–0.2 | thumbnails, previews, messaging under terrible connectivity |
| 3–4 | 0.3–0.45 | "looks like the original" on a screen; JPEG at this size is a mess |
| **5** (default) | 0.6–0.7 | fine detail intact, mean error < 1 grey level |
| 6–7 | 0.9–1.3 | critical viewing |
| 8 | 1.6 | visually transparent, ~40 dB PSNR |

Bits per pixel vary with content: a sky costs almost nothing, a forest costs more. Kodak average given.

### Models (priors)

| name | origin | weights | notes |
|---|---|---|---|
| `mean-scale` (default) | Minnen, Ballé & Toderici 2018, mean‑scale hyperprior | 29–70 MB | 5–10 % smaller files |
| `hyperprior` | Ballé et al. 2018, scale hyperprior | 20–48 MB | smallest download, fastest |

Weights are the published CompressAI checkpoints, downloaded once into `~/.cache/torch/hub/checkpoints`.
Only models with *parallel* entropy decoding are offered: autoregressive context models
(mbt2018, cheng2020) gain another 2–4 % but decode a 24 MP picture in minutes on a CPU, which
defeats "open it and it is there".

## Benchmarks

Measured on this repository (`bench/run_kodak.sh`, `bench/mosaic_bench.py`), Apple M‑series, MPS.
PSNR in dB on RGB; MS‑SSIM shown as −10·log10(1−MS‑SSIM). *Same size* means the classic codec's quality
was searched so that its file is at most as large as the `.npk` file.

**Kodak photo set (24 images, 768×512), averages**

| q | bpp | npk PSNR | JPEG PSNR (same size) | WebP PSNR (same size) | npk MS-SSIM dB | JPEG | WebP |
|---|-----|----------|------------------------|------------------------|----------------|------|------|
| 1 | 0.125 | **27.70** | 22.33 | 26.98 | **11.04** | 6.35 | 10.09 |
| 3 | 0.308 | **31.12** | 26.97 | 29.29 | **14.32** | 10.30 | 12.64 |
| 5 | 0.663 | **34.95** | 30.54 | 32.87 | **18.00** | 14.52 | 16.00 |
| 8 | 1.620 | **40.57** | 35.44 | 38.19 | **23.51** | 19.71 | 20.77 |

Encode/decode per Kodak image: 0.15–0.25 s on MPS (after the one‑time model load).

**One large photo** (4608×2048 mosaic of all 24 Kodak frames, 9.4 MP; source PNG 16.5 MB, raw 28.3 MB)

| q | .npk size | vs PNG | vs raw | PSNR | MS-SSIM dB | same-size JPEG | same-size WebP | JPEG bytes needed for equal PSNR |
|---|---|---|---|---|---|---|---|---|
| 3 | **0.39 MB** | 42× | 72× | 30.6 | 14.3 | 27.2 dB | 29.3 dB | 0.83 MB (2.1×) |
| 5 | **0.84 MB** | 20× | 34× | 34.7 | 18.0 | 30.7 dB | 32.9 dB | 1.77 MB (2.1×) |
| 8 | **2.06 MB** | 8× | 14× | 40.5 | 23.4 | 35.7 dB | 38.4 dB | 3.94 MB (1.9×) |

Throughput on this machine: roughly 5–10 s per 10 megapixels for each direction on MPS, including
PNG writing. A 24 MP camera JPEG of ~20 MB becomes ~1 MB at quality 3 and ~2 MB at quality 5.

**Tiling overhead** (same 9.4 MP photo, q5): one piece 785 kB · 2048‑px tiles +4 % · 1024‑px tiles
(default) +7 % · 512‑px tiles +16 %. Use `--tile 0` or `--tile 2048` when memory allows.

## Design

### File format `.npk`

```
 4  magic "NPK\x01"               4  width          4  tile count
 1  model id                      4  height         4  entropy-table fingerprint (CRC32)
 1  quality                       2  tile edge      4  metadata length
 1  flags (grayscale, metadata)   2  tile overlap   n  metadata: zlib(JSON{exif, icc, dpi, source_format})
 1  reserved
 per tile: u32 len(y) · u32 len(z) · y bytes (main latent) · z bytes (hyper-latent)
 4  CRC32 of everything above
```

36‑byte header, 8 bytes per tile. On a 24 MP picture the container costs < 0.02 %. Tile geometry
is a pure function of (width, height, tile, overlap), so nothing per tile is stored.

### Tiling

Large pictures are coded as a grid of tiles. `--tile` is the edge of the *coded* block
(a multiple of 64 so no interior tile needs padding); each tile carries a halo of `--overlap`
pixels from its neighbours that the decoder crops away again. The halo gives the synthesis
transform real context at borders. Measured on Kodak with 256‑px tiles: without halo the mean
error on seam columns is 25 % higher than elsewhere; with an 8‑px halo it is identical. The
default is a 16‑px halo on 1024‑px tiles.

### Determinism across devices

Learned codecs have a notorious failure mode: if the decoder's probability tables differ from the
encoder's by one floating‑point ulp, the arithmetic decoder desynchronises and the picture turns
to noise. `npk` prevents this structurally:

* Everything that decides *how the bitstream is parsed* (the hyper‑synthesis `h_s`, the index
  tables, the range coder) runs on the **CPU in float64** on both sides. The heavy transforms
  `g_a`/`g_s` run on whatever accelerator is available; they influence *which* symbols are coded
  and *what pixels come out*, never how the bits are parsed.
* Every file carries a **CRC32 fingerprint of the model's quantised entropy tables**. A decoder whose
  tables differ (different weights version, broken install) refuses with a clear message instead
  of emitting garbage. `strict=False` overrides.
* The whole file is CRC‑protected; truncation and bit flips are reported, never "decoded".

Consequence: encode on CUDA, decode on CPU or MPS, same bitstream, pixels within ±1 grey level
(tested in `tests/test_codec.py::test_cross_device_bitstream_compatibility`).

### Metadata, colour, alpha

EXIF, ICC profile and DPI are carried through (`--no-metadata` to strip). Grayscale sources come
back as grayscale. Alpha channels are not supported by the priors and are dropped with a warning.
The transforms work in sRGB; wide‑gamut sources are tagged via the ICC profile but coded as 8‑bit RGB.

### Video (experimental)

`npk encode clip.mp4` produces `clip.npkv`: every frame is an independent `.npk` picture
(intra‑only, like Motion‑JPEG with a neural codec). `npk decode clip.npkv` writes H.264 (`.mp4`),
lossless FFV1 (`.mkv`) or a folder of PNGs. It makes the whole toolchain work end‑to‑end on video
and is excellent for frame sequences and time‑lapses, but it does **not** exploit motion between
frames, so HEVC/AV1 will beat it on ordinary footage at equal bitrate. Audio is not carried.
A learned *inter*‑frame codec is the obvious next step (see roadmap).

## Roadmap

* **Perceptual priors**: HiFiC/GAN‑ or diffusion‑decoded models for 0.1 bpp pictures that look sharp.
* **Inter‑frame neural video** (DCVC family) with the same container philosophy.
* **Lossless layer**: optional residual so `npk` can also be a bit‑exact archive format (ratio then
  drops to ~2:1, as for all lossless photo codecs).
* **Progressive decode / thumbnails** from the hyper‑latent alone.
* **Viewer integrations**: macOS Quick Look plugin, browser WASM decoder.
* **Own prior**: train a larger model on a broader corpus and publish it with a stable fingerprint.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests bench
pytest -q                       # 37 tests; the end-to-end ones download the q3 weights on first run
bench/run_kodak.sh              # reproduce the Kodak table
python bench/mosaic_bench.py big_photo.jpg
```

Layout: `src/neuralpack/` — `models.py` (prior registry, device handling, fingerprint),
`codec.py` (tile coding, public API), `container.py` (file format), `tiling.py`, `video.py`,
`metrics.py`, `bench.py`, `cli.py`. Tests mirror the modules.

Start‑up note: CompressAI imports `torch_geometric` and `torchvision` unguarded, which costs
10–20 s on many machines. `neuralpack.models` satisfies those imports with a throw‑away stub so
`import neuralpack` takes a few seconds (set `NEURALPACK_NO_IMPORT_SHIM=1` to disable).

## Credits

Built on [CompressAI](https://github.com/InterDigitalInc/CompressAI) (InterDigital, BSD‑3) and
the published pretrained weights of

* J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston, *Variational image compression with a scale hyperprior*, ICLR 2018.
* D. Minnen, J. Ballé, G. Toderici, *Joint autoregressive and hierarchical priors for learned image compression*, NeurIPS 2018.

Kodak images from the Kodak Lossless True Color Image Suite.

## Kurzfassung (Deutsch)

`npk` komprimiert Fotos mit einem **geteilten, vortrainierten Modell**: Sender und Empfänger haben
dasselbe "Wörterbuch" aus gelernten Bildstrukturen (eine nichtlineare Verallgemeinerung der
DCT/PCA‑Rotation). Die Datei enthält nur noch entropie‑codierte Indizes in dieses Wörterbuch.
Gemessen: 16,5 MB PNG → 0,84 MB bei 34,7 dB; JPEG braucht für dieselbe Qualität 1,77 MB. Verlustfrei
ist das nicht (das verbietet die Informationstheorie bei 20:1), aber wahrnehmungstreu, geräte‑
unabhängig, mit CRC, Modell‑Fingerabdruck und EXIF‑Durchreichung. `npk encode foto.jpg`,
`npk decode foto.npk`, fertig.

## License

MIT. Model weights keep the licence of their authors (CompressAI, BSD‑3‑Clause).
