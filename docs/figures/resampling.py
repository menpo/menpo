"""Make docs/source/api/shape/resampling_jaw.png: what resampling does to real ibug68 jaws.

    python docs/figures/resampling.py

For each of two faces that come with menpo, the jaw (points 0-16 of ibug68) as annotated, resampled
to equal steps of arc length between the chin and the two ends, and resampled to the mean fractions
of the 300-W annotators. The curve is the Catmull-Rom spline through the points that are drawn.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import menpo.io as mio  # noqa: E402
from menpo.shape import CatmullRom  # noqa: E402

JAW = list(range(17))
ANCHORS = [0, 8, 16]
# the mean position of each jaw point between its anchors over the 300-W annotations
FRACTIONS_300W = {
    1: 0.125, 2: 0.251, 3: 0.377, 4: 0.501, 5: 0.625, 6: 0.75, 7: 0.876,
    9: 0.124, 10: 0.249, 11: 0.373, 12: 0.496, 13: 0.621, 14: 0.748, 15: 0.874,
}  # fmt: skip


def draw(ax, image, points, title, spline):
    if image.n_channels == 1:
        ax.imshow(image.pixels[0], cmap="gray")
    else:
        ax.imshow(image.pixels.transpose(1, 2, 0))
    curve = spline.points_at(np.linspace(0, spline.length, 300))
    ax.plot(curve[:, 1], curve[:, 0], color="#fdd835", lw=1.2)
    ax.plot(points[:, 1], points[:, 0], "o", color="#d81b60", ms=4)
    ax.plot(points[ANCHORS, 1], points[ANCHORS, 0], "s", color="#1e88e5", ms=5)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def main():
    faces = ["breakingbad.jpg", "einstein.jpg"]
    fig, axes = plt.subplots(len(faces), 3, figsize=(10, 3.6 * len(faces)))
    for row, name in zip(axes, faces):
        image = mio.import_builtin_asset(name)
        original = image.landmarks["PTS"].points[JAW]
        spline = CatmullRom(original)
        even = spline.resample(ANCHORS)
        by_300w = spline.resample(ANCHORS, FRACTIONS_300W)
        lo, hi = original.min(axis=0) - 25, original.max(axis=0) + 25
        for ax, points, title in zip(
            row,
            (original, even, by_300w),
            ("as annotated", "equal steps", "mean 300-W spacing"),
        ):
            draw(ax, image, points, title, spline)
            ax.set_xlim(lo[1], hi[1])
            ax.set_ylim(hi[0], lo[0])
    fig.suptitle(
        "Resampling the jaw: anchors (squares) stay, the other points move along the spline",
        fontsize=10,
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "source/api/shape/resampling_jaw.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
