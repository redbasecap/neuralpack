"""Average an ``npk bench --json`` file per quality level (used for the README tables)."""
import json
import statistics as st
import sys


def main(path: str) -> None:
    with open(path) as fh:
        rows = json.load(fh)
    groups: dict[int, list[tuple[dict, dict, dict]]] = {}
    for i in range(0, len(rows), 3):  # bench emits npk, jpeg@same-size, webp@same-size per image
        n, j, w = rows[i:i + 3]
        groups.setdefault(n["quality"], []).append((n, j, w))
    print("| q | bpp | npk PSNR | JPEG PSNR (same size) | WebP PSNR (same size) | npk MS-SSIM dB | JPEG | WebP |")
    print("|---|-----|----------|------------------------|------------------------|----------------|------|------|")
    for q, t in sorted(groups.items()):
        print(_row(q, t))


def _row(q: int, t: list) -> str:
    def m(idx: int, key: str) -> float:
        return st.mean(x[idx][key] for x in t)

    return (f"| {q} | {m(0, 'bpp'):.3f} | **{m(0, 'psnr'):.2f}** | {m(1, 'psnr'):.2f} | {m(2, 'psnr'):.2f} | "
            f"**{m(0, 'msssim_db'):.2f}** | {m(1, 'msssim_db'):.2f} | {m(2, 'msssim_db'):.2f} |")


if __name__ == "__main__":
    main(sys.argv[1])
