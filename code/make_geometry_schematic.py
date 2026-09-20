"""Generates a 3D schematic of the generic rotor-stator cavity geometry
underlying the Pi-group definitions of Section 3.2 (R, R_chamber, h_rotor,
gap delta = R_chamber - R). Purely illustrative (no data plotted); the
dimensions drawn are representative proportions, not any one facility's
real numbers, and the figure caption says so explicitly.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({"font.size": 11, "figure.dpi": 200})

# Representative (illustrative only) proportions.
R = 1.0            # rotor radius
R_chamber = 1.35    # chamber radius
h_rotor = 0.55       # rotor axial height
H_chamber = 1.1      # chamber axial height (just for drawing the shell)
gap = R_chamber - R  # delta

fig = plt.figure(figsize=(7.2, 5.6))
ax = fig.add_subplot(111, projection="3d")

theta = np.linspace(0, 2 * np.pi, 80)

def cyl_surface(radius, z0, z1, alpha, color, n_theta=80):
    th = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(z0, z1, 2)
    Th, Z = np.meshgrid(th, z)
    X = radius * np.cos(Th)
    Y = radius * np.sin(Th)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0,
                     antialiased=True, shade=True)

def disk(radius, z, alpha, color, n_theta=80):
    th = np.linspace(0, 2 * np.pi, n_theta)
    x = radius * np.cos(th)
    y = radius * np.sin(th)
    verts = [list(zip(x, y, [z] * n_theta))]
    pc = Poly3DCollection(verts, color=color, alpha=alpha, linewidth=0)
    ax.add_collection3d(pc)

# Stator chamber: outer translucent shell + top ring (cutaway, no bottom/top
# cap on the near side so the rotor is visible inside).
z0c, z1c = -H_chamber / 2, H_chamber / 2
cyl_surface(R_chamber, z0c, z1c, alpha=0.10, color="tab:gray")
disk(R_chamber, z1c, alpha=0.12, color="tab:gray")
disk(R_chamber, z0c, alpha=0.12, color="tab:gray")

# Rotor: solid-looking inner cylinder, centered axially.
z0r, z1r = -h_rotor / 2, h_rotor / 2
cyl_surface(R, z0r, z1r, alpha=0.55, color="tab:blue")
disk(R, z1r, alpha=0.75, color="tab:blue")
disk(R, z0r, alpha=0.75, color="tab:blue")

# Rotation arrow (omega) on top of the rotor.
arrow_r = R * 0.72
n_arc = 40
arc_th = np.linspace(0.15 * np.pi, 0.85 * np.pi, n_arc)
ax.plot(arrow_r * np.cos(arc_th), arrow_r * np.sin(arc_th),
        [z1r + 0.02] * n_arc, color="black", linewidth=1.6)
# arrowhead
tip = np.array([arrow_r * np.cos(arc_th[-1]), arrow_r * np.sin(arc_th[-1]), z1r + 0.02])
tang = np.array([-np.sin(arc_th[-1]), np.cos(arc_th[-1]), 0]) * 0.14
perp = np.array([np.cos(arc_th[-1]), np.sin(arc_th[-1]), 0]) * 0.07
for sgn in (1, -1):
    p = tip - tang + sgn * perp
    ax.plot([tip[0], p[0]], [tip[1], p[1]], [tip[2], p[2]], color="black", linewidth=1.6)
ax.text(0, arrow_r * 1.25, z1r + 0.05, r"$\omega$", fontsize=13)

# Radial gap indicator: a thin double-headed line at theta=0 from rotor edge
# to chamber wall, at mid-height.
zg = 0
ax.plot([R, R_chamber], [0, 0], [zg, zg], color="tab:red", linewidth=2.0)
ax.scatter([R, R_chamber], [0, 0], [zg, zg], color="tab:red", s=18)
ax.text((R + R_chamber) / 2, 0.05, zg + 0.06, r"$\delta = R_{\mathrm{chamber}}-R$",
        color="tab:red", fontsize=10.5, ha="center")

# Rotor radius R: from axis to rotor edge, at theta=-40deg, top face.
thR = -np.pi / 4.5
ax.plot([0, R * np.cos(thR)], [0, R * np.sin(thR)], [z1r, z1r], color="tab:blue", linewidth=1.6)
ax.text(R * np.cos(thR) * 0.55, R * np.sin(thR) * 0.55, z1r + 0.05, r"$R$",
        color="tab:blue", fontsize=11)

# Chamber radius R_chamber: from axis to chamber wall, at theta=140deg, top ring.
thC = 2.5
ax.plot([0, R_chamber * np.cos(thC)], [0, R_chamber * np.sin(thC)], [z1c, z1c],
        color="dimgray", linewidth=1.4, linestyle="--")
ax.text(R_chamber * np.cos(thC) * 0.55, R_chamber * np.sin(thC) * 0.55, z1c + 0.05,
        r"$R_{\mathrm{chamber}}$", color="dimgray", fontsize=10.5)

# h_rotor: vertical double-headed line at the rotor's outer edge, theta=180deg.
xh, yh = -R, 0
ax.plot([xh, xh], [yh, yh], [z0r, z1r], color="tab:blue", linewidth=2.0)
ax.scatter([xh, xh], [yh, yh], [z0r, z1r], color="tab:blue", s=18)
ax.text(xh - 0.05, yh, 0, r"$h_{\mathrm{rotor}}$", color="tab:blue", fontsize=11,
        ha="right", va="center")

ax.set_box_aspect((1, 1, 0.75))
ax.set_xlim(-R_chamber * 1.15, R_chamber * 1.15)
ax.set_ylim(-R_chamber * 1.15, R_chamber * 1.15)
ax.set_zlim(-H_chamber * 0.75, H_chamber * 0.75)
ax.set_axis_off()
ax.view_init(elev=18, azim=-55)

fig.tight_layout()
OUT_PATH = _ROOT + "/figures/fig0_geometry_schematic.png"
fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# matplotlib's 3D Axes bbox_inches="tight" does not tightly bound actual
# drawn content (a known Axes3D limitation), leaving large blank margins
# in the saved PNG -- confirmed by external review (2026-08-24) flagging
# this exact figure as having an oversized blank gap. Auto-crop to the
# real content bounding box as a second pass.
from PIL import Image
import numpy as np

im = Image.open(OUT_PATH).convert("RGBA")
arr = np.array(im)
alpha = arr[:, :, 3]
rgb = arr[:, :, :3]
is_bg = (alpha == 0) | np.all(rgb >= 250, axis=-1)
rows = np.where(~np.all(is_bg, axis=1))[0]
cols = np.where(~np.all(is_bg, axis=0))[0]
pad = 25
box = (max(0, cols.min() - pad), max(0, rows.min() - pad),
       min(arr.shape[1], cols.max() + pad), min(arr.shape[0], rows.max() + pad))
im.crop(box).save(OUT_PATH)

print(f"Saved {OUT_PATH}")
