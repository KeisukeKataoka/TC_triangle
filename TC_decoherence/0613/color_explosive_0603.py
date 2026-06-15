from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from collections.abc import Iterable
import numpy as np
from mpi4py import MPI
from scipy.sparse import csr_matrix




def selected_link_colors() -> list[str]:
    if SELECT_LINK_COLORS == ["all"]:
        return list(COLOR_NAMES.values())

    allowed = set(COLOR_NAMES.values())
    bad = [c for c in SELECT_LINK_COLORS if c not in allowed]
    if bad:
        raise ValueError(f"unknown link color(s): {bad}; allowed={sorted(allowed)} or [\"all\"]")
    return list(SELECT_LINK_COLORS)


def plaquette_color_for_link_direction(link_color: str, direction_name: str) -> str:
    """Return the plaquette color whose operator uses the selected link color in this direction."""
    matches = [
        plaquette_color_name
        for plaquette_color_name in COLOR_NAMES.values()
        if DIR_TO_LINK_COLOR[plaquette_color_name][direction_name] == link_color
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"link_color={link_color}, direction={direction_name} should correspond to exactly one plaquette color, "
            f"but got {matches}"
        )
    return matches[0]


def selected_directions() -> list[str]:
    if SELECT_DIRECTIONS == ["all"]:
        return list(DIR_NAMES)

    allowed = set(DIR_NAMES)
    bad = [d for d in SELECT_DIRECTIONS if d not in allowed]
    if bad:
        raise ValueError(f"unknown direction(s): {bad}; allowed={sorted(allowed)} or [\"all\"]")
    return list(SELECT_DIRECTIONS)


# ============================================================
# MPI setup
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

os.environ["MPLCONFIGDIR"] = f"/tmp/{os.environ.get('USER', 'user')}/mpl_{rank}"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


# ============================================================
# Geometry
# ============================================================
def idx(x: int, y: int, Lx: int, Ly: int) -> int:
    return (x % Lx) + Lx * (y % Ly)

def inv_idx(i: int, Lx: int, Ly: int) -> tuple[int, int]:
    return i % Lx, i // Lx

def qidx(ix: int, iy: int, s: int, Lx: int, Ly: int) -> int:
    return 2 * idx(ix, iy, Lx, Ly) + s


def plaquette_color(ix: int, iy: int) -> int:
    # 0:red, 1:blue, 2:green
    return (ix + 1 - (iy % 2)) % 3


def plaquette_vertices(ix: int, iy: int, Lx: int, Ly: int) -> np.ndarray:
    if iy % 2 == 0:
        vertices = [
            qidx(ix,     iy,     0, Lx, Ly),
            qidx(ix,     iy,     1, Lx, Ly),
            qidx(ix + 1, iy,     0, Lx, Ly),
            qidx(ix,     iy + 1, 1, Lx, Ly),
            qidx(ix,     iy + 1, 0, Lx, Ly),
            qidx(ix - 1, iy + 1, 1, Lx, Ly),
        ]
    else:
        vertices = [
            qidx(ix,     iy,     0, Lx, Ly),
            qidx(ix,     iy,     1, Lx, Ly),
            qidx(ix + 1, iy,     0, Lx, Ly),
            qidx(ix + 1, iy + 1, 1, Lx, Ly),
            qidx(ix + 1, iy + 1, 0, Lx, Ly),
            qidx(ix,     iy + 1, 1, Lx, Ly),
        ]
    return np.array(vertices, dtype=np.int64)


def create_hexagonal_plaquettes(Lx: int, Ly: int) -> np.ndarray:
    p_ind = np.zeros((Lx * Ly, 6), dtype=np.int64)
    for iy in range(Ly):
        for ix in range(Lx):
            p_ind[idx(ix, iy, Lx, Ly)] = plaquette_vertices(ix, iy, Lx, Ly)
    return p_ind


# ============================================================
# Stabilizer state and dephasing links
# ============================================================
def initial_stabilizer_state(
    Lv: int,
    Lx: int,
    Ly: int,
    p_ind: np.ndarray,
    zero_x: list[int] | None = None,
    zero_z: list[int] | None = None,
) -> tuple[np.ndarray, int]:
    zero_x = set(zero_x or [])
    zero_z = set(zero_z or [])

    MR = np.zeros((Lv, 2 * Lv), dtype=np.uint8)
    for ip in range(Lx * Ly):
        if ip not in zero_x:
            MR[ip, p_ind[ip]] = 1
        if ip not in zero_z:
            MR[Lx * Ly + ip, p_ind[ip] + Lv] = 1
    return MR, Lv


def red_zz(p_ind: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    zz = np.zeros((Lx * Ly, 2), dtype=np.int64)
    for iy in range(Ly):
        for ix in range(Lx):
            color_id = plaquette_color(ix, iy)
            ip0 = idx(ix, iy, Lx, Ly)

            if color_id == 0:  # red plaquette
                ip1 = idx(ix - 1, iy, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
            elif color_id == 1:  # blue plaquette
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip0, 1])
            else:  # green plaquette
                ip1 = idx(ix - 1, iy + 1, Lx, Ly) if iy % 2 == 0 else idx(ix, iy + 1, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
    return zz


def blue_zz(p_ind: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    zz = np.zeros((Lx * Ly, 2), dtype=np.int64)
    for iy in range(Ly):
        for ix in range(Lx):
            color_id = plaquette_color(ix, iy)
            ip0 = idx(ix, iy, Lx, Ly)

            if color_id == 0:  # red plaquette
                ip1 = idx(ix - 1, iy + 1, Lx, Ly) if iy % 2 == 0 else idx(ix, iy + 1, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
            elif color_id == 1:  # blue plaquette
                ip1 = idx(ix - 1, iy, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
            else:  # green plaquette
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip0, 1])
    return zz


def green_zz(p_ind: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    zz = np.zeros((Lx * Ly, 2), dtype=np.int64)
    for iy in range(Ly):
        for ix in range(Lx):
            color_id = plaquette_color(ix, iy)
            ip0 = idx(ix, iy, Lx, Ly)

            if color_id == 0:  # red plaquette
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip0, 1])
            elif color_id == 1:  # blue plaquette
                ip1 = idx(ix - 1, iy + 1, Lx, Ly) if iy % 2 == 0 else idx(ix, iy + 1, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
            else:  # green plaquette
                ip1 = idx(ix - 1, iy, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
    return zz


def create_zz_links_by_color(p_ind: np.ndarray, Lx: int, Ly: int) -> dict[str, np.ndarray]:
    return {
        "red": red_zz(p_ind, Lx, Ly),
        "blue": blue_zz(p_ind, Lx, Ly),
        "green": green_zz(p_ind, Lx, Ly),
    }

def dephasing_dlinkZZ_inplace(dMR: np.ndarray, ipd: int, zz: np.ndarray, nsdd: int) -> int:
    q0, q1 = map(int, zz[ipd])
    active = dMR[:nsdd]
    bad = np.flatnonzero((active[:, q0] ^ active[:, q1]) == 1)

    if bad.size == 0:
        return nsdd

    k0 = int(bad[0])
    pivot = active[k0].copy()
    if bad.size > 1:
        active[bad[1:]] ^= pivot

    last = nsdd - 1
    if k0 != last:
        dMR[k0] = dMR[last]

    return nsdd - 1

# ============================================================
# Union-Find
# ============================================================

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union_score(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return self.size[ra] ** 2
        return self.size[ra] * self.size[rb]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)

        if ra == rb:
            return self.size[ra]

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

        return self.size[ra]

# ============================================================
# Geometry
# ============================================================
def idx(x: int, y: int, Lx: int, Ly: int) -> int:
    return (x % Lx) + Lx * (y % Ly)

def inv_idx(i: int, Lx: int, Ly: int) -> tuple[int, int]:
    return i % Lx, i // Lx

def qidx(ix: int, iy: int, s: int, Lx: int, Ly: int) -> int:
    return 2 * idx(ix, iy, Lx, Ly) + s

# ============================================================
# setting of plaquettes and its color
# ============================================================

def plaquette_color(ix: int, iy: int) -> int:
    # 0:red, 1:blue, 2:green
    return (ix + 1 - (iy % 2)) % 3

def plaquette_vertices(ix: int, iy: int, Lx: int, Ly: int) -> np.ndarray:
    if iy % 2 == 0:
        vertices = [
            qidx(ix,     iy,     0, Lx, Ly),
            qidx(ix,     iy,     1, Lx, Ly),
            qidx(ix + 1, iy,     0, Lx, Ly),
            qidx(ix,     iy + 1, 1, Lx, Ly),
            qidx(ix,     iy + 1, 0, Lx, Ly),
            qidx(ix - 1, iy + 1, 1, Lx, Ly),
        ]
    else:
        vertices = [
            qidx(ix,     iy,     0, Lx, Ly),
            qidx(ix,     iy,     1, Lx, Ly),
            qidx(ix + 1, iy,     0, Lx, Ly),
            qidx(ix + 1, iy + 1, 1, Lx, Ly),
            qidx(ix + 1, iy + 1, 0, Lx, Ly),
            qidx(ix,     iy + 1, 1, Lx, Ly),
        ]
    return np.array(vertices, dtype=np.int64)

def create_hexagonal_plaquettes(Lx: int, Ly: int) -> np.ndarray:
    # plaquette番号 -> plaquette番号+頂点のqubit番号の配列
    p_ind = np.zeros((Lx * Ly, 6), dtype=np.int64)
    for iy in range(Ly):
        for ix in range(Lx):
            p_ind[idx(ix, iy, Lx, Ly)] = plaquette_vertices(ix, iy, Lx, Ly)
    return p_ind

# ============================================================
# setting of zz links
# ============================================================

def red_zz(p_ind: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    # red_zz linkでつながっている2点のqubit番号を返す
    zz = np.zeros((Lx * Ly, 2), dtype=np.int64)
    for iy in range(Ly):
        for ix in range(Lx):
            color_id = plaquette_color(ix, iy)
            ip0 = idx(ix, iy, Lx, Ly)

            if color_id == 0:  # red plaquette
                ip1 = idx(ix - 1, iy, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
            elif color_id == 1:  # blue plaquette
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip0, 1])
            else:  # green plaquette
                ip1 = idx(ix - 1, iy + 1, Lx, Ly) if iy % 2 == 0 else idx(ix, iy + 1, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
    return zz


def blue_zz(p_ind: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    zz = np.zeros((Lx * Ly, 2), dtype=np.int64)
    for iy in range(Ly):
        for ix in range(Lx):
            color_id = plaquette_color(ix, iy)
            ip0 = idx(ix, iy, Lx, Ly)

            if color_id == 0:  # red plaquette
                ip1 = idx(ix - 1, iy + 1, Lx, Ly) if iy % 2 == 0 else idx(ix, iy + 1, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
            elif color_id == 1:  # blue plaquette
                ip1 = idx(ix - 1, iy, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
            else:  # green plaquette
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip0, 1])
    return zz


def green_zz(p_ind: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    zz = np.zeros((Lx * Ly, 2), dtype=np.int64)
    for iy in range(Ly):
        for ix in range(Lx):
            color_id = plaquette_color(ix, iy)
            ip0 = idx(ix, iy, Lx, Ly)

            if color_id == 0:  # red plaquette
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip0, 1])
            elif color_id == 1:  # blue plaquette
                ip1 = idx(ix - 1, iy + 1, Lx, Ly) if iy % 2 == 0 else idx(ix, iy + 1, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
            else:  # green plaquette
                ip1 = idx(ix - 1, iy, Lx, Ly)
                zz[ip0] = (p_ind[ip0, 0], p_ind[ip1, 1])
    return zz


def create_zz_links_by_color(p_ind: np.ndarray, Lx: int, Ly: int) -> dict[str, np.ndarray]:
    return {
        "red": red_zz(p_ind, Lx, Ly),
        "blue": blue_zz(p_ind, Lx, Ly),
        "green": green_zz(p_ind, Lx, Ly),
    }

# def qubit_to_red_plaquette(Lx, Ly):
#     reds = red_plaquettes(Lx, Ly)
#     q_to_p = {}
#     for ip, qs in reds:
#         for q in qs:
#             q_to_p[int(q)] = int(ip)

#     return q_to_p

def red_qubit_to_plaquette(Lx, Ly):
    # qubit番号 -> 赤色プラケット番号の辞書を作る
    q_to_p = {}

    for iy in range(Ly):
        for ix in range(Lx):

            if plaquette_color(ix, iy) != 0:
                continue

            ip = idx(ix, iy, Lx, Ly)
            qs = plaquette_vertices(ix, iy, Lx, Ly)

            for q in qs:
                q_to_p[int(q)] = ip
    return q_to_p

def red_zz_to_plaquette_edges(zz, Lx, Ly):
    # red_zz linkでつながっている2点がどの2つのred plaquetteに属しているかを調べる
    q_to_p = red_qubit_to_plaquette(Lx, Ly)

    pedges = []

    for q1, q2 in zz:

        p1 = q_to_p[int(q1)]
        p2 = q_to_p[int(q2)]

        pedges.append((p1, p2))
        
        # print(
        #     f"zz ({q1:2d}, {q2:2d}) "
        #     f"-> red plaquettes ({p1:2d}, {p2:2d})"
        # )

    return np.array(pedges, dtype=np.int64)

# ============================================================
# brick wall geometry and plot
# ============================================================

def make_brick_wall(Lx, Ly):
    """
    Lx, Ly: unit cell numbers
    各セルに A, B の2点を置く
    """
    pos = []
    #edges = []

    for ip in range(Lx * Ly):
        ix, iy = inv_idx(ip, Lx, Ly)
        if iy % 2 == 0:
                # even site
                iq = qidx(ix, iy, 0, Lx, Ly)
                pos.append((iq, ix, iy))
                # odd site
                iq = qidx(ix, iy, 1, Lx, Ly)
                pos.append((iq, ix + 0.5, iy))
        else:
                # even site
                iq = qidx(ix, iy, 0, Lx, Ly)
                pos.append((iq, ix + 0.5, iy))
                # odd site
                iq = qidx(ix, iy, 1, Lx, Ly)
                pos.append((iq, ix + 1.0, iy))
    return np.array(pos)

# =========================
# plot
# =========================
def plot_brick_wall(pos):

    fig, ax = plt.subplots(figsize=(8, 5))

    for iq, x, y in pos:

        # 点
        ax.scatter(x, y, s=80)

        # qubit番号
        ax.text(
            x,
            y + 0.12,
            f"{int(iq)}",
            ha="center",
            fontsize=10
        )

    ax.set_aspect("equal")

    ax.set_xlim(np.min(pos[:,1]) - 0.5,
                np.max(pos[:,1]) + 0.5)

    ax.set_ylim(np.min(pos[:,2]) - 0.5,
                np.max(pos[:,2]) + 0.5)

    ax.axis("off")

    plt.show()

def plot_red_zz_edges_with_red_plaquette(pos, zz, Lx, Ly):

    fig, ax = plt.subplots(figsize=(8, 6))

    qpos = {int(iq): (x, y) for iq, x, y in pos}

    # red plaquette を正方形で表示
    for iy in range(Ly):
        for ix in range(Lx):

            if plaquette_color(ix, iy) != 0:
                continue

            if iy % 2 == 0:
                xc = ix
            else:
                xc = ix + 0.5

            yc = iy

            square = Rectangle(
                (xc, yc),
                1.0,
                1.0,
                facecolor="red",
                alpha=0.25,
                edgecolor="red",
                lw=1.5,
                zorder=0
            )
            ax.add_patch(square)

    # red_zz edge
    for q1, q2 in zz:
        x1, y1 = qpos[int(q1)]
        x2, y2 = qpos[int(q2)]

        dx = x2 - x1
        dy = y2 - y1

        # 周期境界をまたぐ線は描かない
        if abs(dx) > Lx / 2 or abs(dy) > Ly / 2:
            continue

        ax.plot([x1, x2], [y1, y2], color="red", lw=2, zorder=1)

    # qubit
    ax.scatter(pos[:, 1], pos[:, 2],
               s=80, color="white", edgecolors="black", zorder=2)

    for iq, x, y in pos:
        ax.text(x, y + 0.1, str(int(iq)),
                ha="center", fontsize=9, zorder=3)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(-0.5, np.max(pos[:, 1]) + 0.5)
    ax.set_ylim(-0.5, np.max(pos[:, 2]) + 0.5)

    plt.show()

def plot_red_zz_edges_with_highlight(
    pos,
    zz,
    Lx,
    Ly,
    highlight_indices=None
    ):

    fig, ax = plt.subplots(figsize=(8, 6))

    qpos = {int(iq): (x, y) for iq, x, y in pos}

    q_to_p = red_qubit_to_plaquette(Lx, Ly)

    # -------------------------
    # highlight 用
    # -------------------------
    if highlight_indices is None:
        highlight_indices = []

    highlight_indices = set(highlight_indices)

    highlight_ps = set()
    highlight_qs = set()

    for i in highlight_indices:

        q1, q2 = zz[i]

        q1 = int(q1)
        q2 = int(q2)

        highlight_qs.add(q1)
        highlight_qs.add(q2)

        highlight_ps.add(q_to_p[q1])
        highlight_ps.add(q_to_p[q2])

        # print(
        #     f"zz[{i}] = ({q1}, {q2}) "
        #     f"-> plaquettes "
        #     f"({q_to_p[q1]}, {q_to_p[q2]})"
        # )

    # -------------------------
    # red plaquette
    # -------------------------
    for iy in range(Ly):
        for ix in range(Lx):

            if plaquette_color(ix, iy) != 0:
                continue

            ip = idx(ix, iy, Lx, Ly)

            if iy % 2 == 0:
                xc = ix
            else:
                xc = ix + 0.5

            yc = iy

            if ip in highlight_ps:
                facecolor = "red"
                edgecolor = "red"
                alpha = 0.65
                lw = 3
                zorder = 1
            else:
                facecolor = "red"
                edgecolor = "red"
                alpha = 0.18
                lw = 1
                zorder = 0

            square = Rectangle(
                (xc, yc),
                1.0,
                1.0,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                lw=lw,
                zorder=zorder
            )

            ax.add_patch(square)

    # -------------------------
    # zz edges
    # -------------------------
    for i, (q1, q2) in enumerate(zz):

        q1 = int(q1)
        q2 = int(q2)

        x1, y1 = qpos[q1]
        x2, y2 = qpos[q2]

        dx = x2 - x1
        dy = y2 - y1

        # PBC をまたぐ線は描かない
        if abs(dx) > Lx / 2 or abs(dy) > Ly / 2:
            continue

        if i in highlight_indices:
            color = "red"
            lw = 2
            alpha = 1.0
            zorder = 4
        else:
            color = "red"
            lw = 1
            alpha = 0.3
            zorder = 2

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=zorder
        )

    # -------------------------
    # qubits
    # -------------------------
    for iq, x, y in pos:

        iq = int(iq)

        if iq in highlight_qs:
            ax.scatter(
                x, y,
                s=20,
                color="red",
                edgecolors="red",
                linewidths=2.5,
                zorder=5
            )
        else:
            ax.scatter(
                x, y,
                s=10,
                color="white",
                edgecolors="black",
                zorder=3
            )

        # ax.text(
        #     x,
        #     y + 0.1,
        #     str(iq),
        #     ha="center",
        #     fontsize=9,
        #     zorder=6
        # )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(-0.5, np.max(pos[:, 1]) + 0.5)
    ax.set_ylim(-0.5, np.max(pos[:, 2]) + 0.5)

    plt.show()

def red_plaquettes(Lx, Ly):
    p_ind = create_hexagonal_plaquettes(Lx, Ly)
    reds = []
    for iy in range(Ly):
        for ix in range(Lx):

            if plaquette_color(ix, iy) != 0:
                continue

            ip = idx(ix, iy, Lx, Ly)

            # ip も一緒に保存
            reds.append((ip, p_ind[ip]))

    return reds

def explosive_percolation_zz(zz, Lx, Ly, ntrial=2, seed=0, plot_every=20):
    rng = np.random.default_rng(seed)
    pos = make_brick_wall(Lx, Ly)
    pedges = red_zz_to_plaquette_edges(zz, Lx, Ly)

    red_ids = sorted(set(pedges[:, 0]) | set(pedges[:, 1]))
    p_to_c = {p: i for i, p in enumerate(red_ids)}
    c_to_p = {i: p for p, i in p_to_c.items()}

    edges = np.array([[p_to_c[int(a)], p_to_c[int(b)]] for a, b in pedges], dtype=int)
    #print("M_raw =", len(edges))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    #print("M_unique =", len(edges))


    N = len(red_ids)
    M = len(edges)

    uf = UnionFind(N)
    remaining = list(range(M))
    chosen = []

    t_list = []
    largest_list = []

    for step in range(M):
        cand = rng.choice(remaining, size=min(ntrial, len(remaining)), replace=False)

        scores = []
        for ei in cand:
            a, b = edges[ei]
            scores.append(uf.union_score(a, b))

        best = cand[np.argmin(scores)]

        # q0, q1 = zz[best]

        # print(
        #     f"[EXP] step={step:4d} "
        #     f"best={best:4d} "
        #     f"qubits=({q0:4d},{q1:4d})"
        # )

        a, b = edges[best]
        uf.union(a, b)

        remaining.remove(best)
        chosen.append(best)

        largest = max(uf.size[uf.find(i)] for i in range(N))

        # M = len(edges) = total number of red_zz edges
        t_list.append((step + 1) / M)
        # N = number of red plaquettes
        #t_list.append((step + 1) / N)

        largest_list.append(largest / N)

        # print(
        #     f"step={step+1:4d}  "
        #     f"p={(step+1)/M:.3f}  "
        #     f"largest/N={largest/N:.3f}"
        #     f"(largest cluster size={largest:4d})"
        # )

        # if (step + 1) % plot_every == 0:
        #     plot_red_zz_edges_with_highlight(
        #         pos, zz, Lx, Ly,
        #         highlight_indices=chosen
        #     )

    return chosen, uf, np.array(t_list), np.array(largest_list)


# ------------------------------
# Fast GF(2) linear algebra
# ------------------------------
@njit(cache=True)
def rank_mod2_numba(A: np.ndarray) -> int:
    A = A.copy()
    nrow, ncol = A.shape
    r = 0
    for c in range(ncol):
        pivot = -1
        for i in range(r, nrow):
            if A[i, c] != 0:
                pivot = i
                break
        if pivot == -1:
            continue

        if pivot != r:
            tmp = A[r].copy()
            A[r] = A[pivot]
            A[pivot] = tmp

        for i in range(nrow):
            if i != r and A[i, c] != 0:
                A[i, :] ^= A[r, :]

        r += 1
        if r == nrow:
            break
    return r

# ============================================================
# Negativity calculation using rank of commutation matrix
# ============================================================

def negativity_E_fast(MRdn: np.ndarray, nsdd2: int, Ld: int, sub_idx: np.ndarray) -> float:
    mAB = nsdd2
    active = MRdn[:mAB]

    X = active[:, sub_idx].astype(np.uint16, copy=False)
    Z = active[:, Ld + sub_idx].astype(np.uint16, copy=False)

    ComM = ((X @ Z.T) + (Z @ X.T)) & 1
    rankJ = rank_mod2_numba(ComM.astype(np.uint8, copy=False))
    return 0.5 * rankJ

# ============================================================
# Geometry for TEN subsystems and operators
# ============================================================

def plaq_rel(r: int) -> list[tuple[int, int]]:
    rel = []
    for dy in range(-r, r + 1):
        dx_min = max(-r, dy - r)
        dx_max = min(r, dy + r)
        for dx in range(dx_min, dx_max + 1):
            rel.append((dx, dy))
    return rel

def plaqN(A0: int, r: int, Lx: int, Ly: int) -> list[int]:
    x0, y0 = inv_idx(A0, Lx, Ly)
    return [idx(x0 + dx, y0 + dy, Lx, Ly) for dx, dy in plaq_rel(r)]


def plaq7(A0: int, Lx: int, Ly: int) -> list[int]:
    x0, y0 = inv_idx(A0, Lx, Ly)
    if y0 % 2 == 0:
        rel = [
            (-1, -1), ( 0, -1),
            (-1,  0), ( 0,  0), ( 1,  0),
            (-1,  1), ( 0,  1),
        ]
    else:
        rel = [
            ( 0, -1), ( 1, -1),
            (-1,  0), ( 0,  0), ( 1,  0),
            ( 0,  1), ( 1,  1),
        ]
    return [idx(x0+dx, y0+dy, Lx, Ly) for dx,dy in rel]

def plaq19(A0: int, Lx: int, Ly: int) -> list[int]:
    x0, y0 = inv_idx(A0, Lx, Ly)
    rel = [(-2,-2), (-1,-2), ( 0,-2),
           (-2,-1), (-1,-1), ( 0,-1), ( 1,-1),
           (-2, 0), (-1, 0), ( 0, 0), ( 1, 0), ( 2, 0),
           (-1, 1), ( 0, 1), ( 1, 1), ( 2, 1),
           ( 0, 2), ( 1, 2), ( 2, 2)]
    return [idx(x0 + dx, y0 + dy, Lx, Ly) for dx, dy in rel]

def plaq37(A0: int, Lx: int, Ly: int) -> list[int]:
    x0, y0 = inv_idx(A0, Lx, Ly)
    rel = [(-3,-3), (-2,-3), (-1,-3), ( 0,-3), 
           (-3,-2), (-2,-2), (-1,-2), ( 0,-2), ( 1,-2),
           (-3,-1), (-2,-1), (-1,-1), ( 0,-1), ( 1,-1), ( 2,-1),
           (-3, 0), (-2, 0), (-1, 0), ( 0, 0), ( 1, 0), ( 2, 0), ( 3, 0),
           (-2, 1), (-1, 1), ( 0, 1), ( 1, 1), ( 2, 1), ( 3, 1),
           (-1, 2), ( 0, 2), ( 1, 2), ( 2, 2), ( 3, 2),           
           ( 0, 3), ( 1, 3), ( 2, 3), ( 3, 3)]
    return [idx(x0 + dx, y0 + dy, Lx, Ly) for dx, dy in rel]

def plaq37(A0: int, Lx: int, Ly: int) -> list[int]:
    return plaqN(A0, 3, Lx, Ly)




# ============================================================
# Operators and correlators
# ============================================================
def symplectic_metric(Lv: int) -> csr_matrix:
    G = np.zeros((2 * Lv, 2 * Lv), dtype=np.uint8)
    for q in range(Lv):
        G[q, Lv + q] = 1
        G[Lv + q, q] = 1
    return csr_matrix(G)


def renyi2_fraction(dMR: np.ndarray, G_csr: csr_matrix, ST_csr: csr_matrix) -> float:
    M = csr_matrix(dMR).dot(G_csr.dot(ST_csr)).toarray() % 2
    return float(np.mean(np.all(M == 0, axis=0)))


def step_direction(x: int, y: int, direction_name: str) -> tuple[int, int]:
    if direction_name == "up":
        return 0, 2
    if direction_name == "up_right":
        return (1, 1) if y % 2 == 0 else (2, 1)
    if direction_name == "up_left":
        return (-2, 1) if y % 2 == 0 else (-1, 1)
    raise ValueError(f"unknown direction_name: {direction_name}")


def add_link_to_operator(op: np.ndarray, q0: int, q1: int, Lv: int, pauli: str) -> None:
    if pauli == "X":
        op[q0] ^= 1
        op[q1] ^= 1
    elif pauli == "Z":
        op[Lv + q0] ^= 1
        op[Lv + q1] ^= 1
    else:
        raise ValueError("pauli must be 'X' or 'Z'")


def make_string_or_loop(
    ix0: int,
    iy0: int,
    direction_name: str,
    Lx: int,
    Ly: int,
    zz_links_by_color: dict[str, np.ndarray],
    pauli: str,
    length: int | None,
) -> np.ndarray:
    Lv = 2 * Lx * Ly
    op = np.zeros(2 * Lv, dtype=np.uint8)

    plaquette_name = COLOR_NAMES[plaquette_color(ix0, iy0)]
    link_color = DIR_TO_LINK_COLOR[plaquette_name][direction_name]
    zz_links = zz_links_by_color[link_color]

    x, y = ix0, iy0
    visited = set()
    step = 0

    while True:
        if length is None:
            key = (x % Lx, y % Ly)
            if key in visited:
                break
            visited.add(key)
        elif step >= length:
            break

        q0, q1 = zz_links[idx(x, y, Lx, Ly)]
        add_link_to_operator(op, int(q0), int(q1), Lv, pauli)

        dx, dy = step_direction(x, y, direction_name)
        x = (x + dx) % Lx
        y = (y + dy) % Ly
        step += 1

    return op


def unique_columns(mat: np.ndarray) -> np.ndarray:
    seen: set[bytes] = set()
    cols = []
    for j in range(mat.shape[1]):
        key = mat[:, j].tobytes()
        if key not in seen:
            seen.add(key)
            cols.append(mat[:, j])
    return np.stack(cols, axis=1).astype(np.uint8)


def create_string_loop_operators(
    Lx: int,
    Ly: int,
    zz_links_by_color: dict[str, np.ndarray],
    max_string_length: int,
    link_colors: list[str],
    directions: list[str],
) -> dict[str, dict[str, dict[str, object]]]:
    """
    ops[link_color][direction_name] contains R2x/R2z/R2x_loop/R2z_loop.

    The user selects link_color and direction.
    The required plaquette color is determined internally by DIR_TO_LINK_COLOR.
    """
    ops = {}
    color_to_id = {name: cid for cid, name in COLOR_NAMES.items()}

    for link_color in link_colors:
        ops[link_color] = {}

        for dname in directions:
            plaquette_name = plaquette_color_for_link_direction(link_color, dname)
            plaquette_id = color_to_id[plaquette_name]

            ops[link_color][dname] = {}
            ops[link_color][dname]["plaquette_color"] = plaquette_name

            # ----------------------------
            # string operators by length
            # ----------------------------
            for slen in range(1, max_string_length + 1):
                lists = {"R2x": [], "R2z": []}

                for iy in range(Ly):
                    for ix in range(Lx):
                        if plaquette_color(ix, iy) != plaquette_id:
                            continue

                        lists["R2x"].append(
                            make_string_or_loop(ix, iy, dname, Lx, Ly, zz_links_by_color, "X", slen)
                        )
                        lists["R2z"].append(
                            make_string_or_loop(ix, iy, dname, Lx, Ly, zz_links_by_color, "Z", slen)
                        )

                if not lists["R2x"]:
                    raise RuntimeError(
                        f"No plaquettes found for link_color={link_color}, direction={dname}, "
                        f"plaquette_color={plaquette_name}"
                    )

                ops[link_color][dname][slen] = {}

                for name in ["R2x", "R2z"]:
                    mat = unique_columns(np.stack(lists[name], axis=1))
                    ops[link_color][dname][slen][name] = csr_matrix(mat)

                    # if rank == 0:
                    #     print(
                    #         f"link={link_color:5s} dir={dname:8s} "
                    #         f"len={slen:2d} plaquette={plaquette_name:5s} "
                    #         f"{name:8s} {mat.shape}",
                    #         flush=True,
                    #     )

            # ----------------------------
            # loop operators
            # ----------------------------
            loop_lists = {"R2x_loop": [], "R2z_loop": []}

            for iy in range(Ly):
                for ix in range(Lx):
                    if plaquette_color(ix, iy) != plaquette_id:
                        continue

                    loop_lists["R2x_loop"].append(
                        make_string_or_loop(ix, iy, dname, Lx, Ly, zz_links_by_color, "X", None)
                    )
                    loop_lists["R2z_loop"].append(
                        make_string_or_loop(ix, iy, dname, Lx, Ly, zz_links_by_color, "Z", None)
                    )

            ops[link_color][dname]["loop"] = {}

            for name in ["R2x_loop", "R2z_loop"]:
                mat = unique_columns(np.stack(loop_lists[name], axis=1))
                ops[link_color][dname]["loop"][name] = csr_matrix(mat)

                # if rank == 0:
                #     print(
                #         f"link={link_color:5s} dir={dname:8s} "
                #         f"loop plaquette={plaquette_name:5s} "
                #         f"{name:8s} {mat.shape}",
                #         flush=True,
                #     )

    return ops

# ============================================================
# GF(2) rank and stabilizer-dependence check
# ============================================================
def rank_mod2(A: np.ndarray) -> int:
    A = np.asarray(A, dtype=np.uint8).copy() % 2
    m, n = A.shape
    r = 0
    for c in range(n):
        pivots = np.flatnonzero(A[r:, c])
        if pivots.size == 0:
            continue
        p = r + int(pivots[0])
        A[[r, p]] = A[[p, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
        if r == m:
            break
    return r



def iter_operator_columns(ST, nrow: int):
    M = ST.toarray() if hasattr(ST, "toarray") else np.asarray(ST)
    M = np.asarray(M, dtype=np.uint8) % 2
    if M.ndim == 1:
        if M.shape[0] != nrow:
            raise ValueError(f"bad op shape: {M.shape}, expected ({nrow},)")
        yield 0, M
    elif M.ndim == 2 and M.shape[0] == nrow:
        for j in range(M.shape[1]):
            yield j, M[:, j]
    else:
        raise ValueError(f"bad op shape: {M.shape}, expected ({nrow}, n_ops)")


def in_span_mod2(S: np.ndarray, v: np.ndarray) -> tuple[bool, int, int]:
    rS = rank_mod2(S)
    rSV = rank_mod2(np.vstack([S, v]))
    return (rSV == rS), rS, rSV


def dependence_fraction(S: np.ndarray, ST) -> tuple[float, int, int, int, int]:
    n_dep = 0
    n_ops = 0
    rank_S = 0
    max_rank = 0

    for _, op_vec in iter_operator_columns(ST, S.shape[1]):
        dep, rS, rSO = in_span_mod2(S, op_vec)
        rank_S = rS
        max_rank = max(max_rank, rSO)
        n_dep += int(dep)
        n_ops += 1

    if n_ops == 0:
        return np.nan, 0, 0, rank_S, max_rank
    return n_dep / n_ops, n_dep, n_ops, rank_S, max_rank


def print_dependence_details(S: np.ndarray, ST, ip: int, p: float, link_color: str, dname: str, opname: str) -> None:
    for j, op_vec in iter_operator_columns(ST, S.shape[1]):
        dep, rS, rSO = in_span_mod2(S, op_vec)
        print(
            f"ip={ip}, p={p:.3f}, link={link_color}, dir={dname}, {opname}[{j}]: "
            f"rank(S)={rS}, rank(S+op)={rSO}, dependent={dep}",
            flush=True,
        )


def should_print_details(ip: int) -> bool:
    if not DEBUG_DEPENDENCE_DETAILS:
        return False
    if DETAIL_ONLY_RANK0 and rank != 0:
        return False
    return DETAIL_IP_LIST is None or ip in DETAIL_IP_LIST



# ============================================================
# Parameters
# ============================================================
Nd = 1000
ps, pl = 0.0, 0.5
Np = 21
Lx, Ly = 12, 15

ntrial = Lx*Ly

MAX_STRING_LENGTH = Ly // 4
BASE_SEED = 54321

# The center plaquette for TEN regions A/B/C. Must be less than Lx*Ly.
# 41 => green, 42 => red, 40 => blue (Lx=24, Ly=16)
CENTER_COLOR = 101



cx, cy = inv_idx(CENTER_COLOR, Lx, Ly)

COLOR = plaquette_color(cx, cy)

# Decoherence link color: "red", "blue", or "green"
DEPHASING_LINK_COLOR = "red"

# Operator selection
# Select by ZZ link color and direction.
# Use ["all"] to calculate all 9 link-color/direction combinations.
# Examples:
#   SELECT_LINK_COLORS = ["red"]
#   SELECT_DIRECTIONS = ["up_right"]
SELECT_LINK_COLORS = ["red"]        # "red", "blue", "green", or "all"
SELECT_DIRECTIONS = ["up"]         # "up_right", "up", "up_left", or "all"

# Debug options
DEBUG_PRINT_OPS = False          # print string/loop vectors once
DEBUG_PRINT_MR = False           # print active stabilizer matrix at each p
DEBUG_DEPENDENCE_DETAILS = False # print each operator's rank check
DETAIL_IP_LIST = None            # e.g. [0, 5, 10]; None means all p values
DETAIL_ONLY_RANK0 = True

COLOR_NAMES = {0: "red", 1: "blue", 2: "green"}
DIR_NAMES = ["up_right", "up", "up_left"]

cname = COLOR_NAMES[COLOR]

#print(cname, "centered at", (cx, cy))


R2_string_KEYS = ["R2x", "R2z"]
R2_loop_KEYS = ["R2x_loop", "R2z_loop"]
DEP_string_KEYS = ["dep_R2x", "dep_R2z"]
DEP_loop_KEYS = ["dep_R2x_loop", "dep_R2z_loop"]

DIR_TO_LINK_COLOR = {
    "red":   {"up": "blue",  "up_left": "green", "up_right": "red"},
    "blue":  {"up": "green", "up_left": "red",   "up_right": "blue"},
    "green": {"up": "red",   "up_left": "blue",  "up_right": "green"},
}



# ============================================================
# Main
# ============================================================
def make_nested_accumulator(keys: list[str], Np: int, link_colors: list[str], directions: list[str], string_lengths: Iterable[int],) -> dict:
    return {
        link_color: {
            dname: {
                key: {
                    slen: np.zeros(Np, dtype=np.float64)
                    for slen in string_lengths
                }
                for key in keys
            }
            for dname in directions
        }
        for link_color in link_colors
    }

def make_loop_accumulator(keys, Np, link_colors, directions):
    return {
        link_color: {
            dname: {
                key: np.zeros(Np, dtype=np.float64)
                for key in keys
            }
            for dname in directions
        }
        for link_color in link_colors
    }

def main() -> None:
    Lv = 2 * Lx * Ly
    Lh = Lx * Ly
    p_list = np.linspace(ps, pl, Np)
    start = time.time()

    pos = make_brick_wall(Lx, Ly)
    p_ind = create_hexagonal_plaquettes(Lx, Ly)
    zz_links_by_color = create_zz_links_by_color(p_ind, Lx, Ly)
    zz_dephasing = zz_links_by_color[DEPHASING_LINK_COLOR]

    # ------------------------------
    # TEN subsystems
    # ------------------------------
    

    xc, yc = inv_idx(CENTER_COLOR, Lx, Ly)

    print(f"Center plaquette: ip={CENTER_COLOR}, color={cname}, at (x={xc}, y={yc})", flush=True)

    # subsystem A is two overlapping plaq7 regions, centered at Ac1 and Ac2
    if yc % 2 == 0:
        xA1c, yA1c = (xc - 2) % Lx, (yc - 1) % Ly
        xA2c, yA2c = (xc - 3) % Lx, yc % Ly
        xA3c, yA3c = (xc - 3) % Lx, (yc + 2) % Ly
        xA4c, yA4c = (xc - 3) % Lx, (yc + 4) % Ly
        xA5c, yA5c = (xc - 2) % Lx, (yc + 5) % Ly
        xA6c, yA6c =  xc      % Lx, (yc + 6) % Ly
    else:
        xA1c, yA1c = (xc - 1) % Lx, (yc - 1) % Ly
        xA2c, yA2c = (xc - 3) % Lx, yc % Ly
        xA3c, yA3c = (xc - 3) % Lx, (yc + 2) % Ly
        xA4c, yA4c = (xc - 3) % Lx, (yc + 4) % Ly
        xA5c, yA5c = (xc - 1) % Lx, (yc + 5) % Ly
        xA6c, yA6c =  xc      % Lx, (yc + 6) % Ly

    A1c = idx(xA1c, yA1c, Lx, Ly)   
    A1 = set().union(*[p_ind[i] for i in plaq7(A1c, Lx, Ly)])
    A2c = idx(xA2c, yA2c, Lx, Ly)   
    A2 = set().union(*[p_ind[i] for i in plaq7(A2c, Lx, Ly)])   
    A3c = idx(xA3c, yA3c, Lx, Ly)   
    A3 = set().union(*[p_ind[i] for i in plaq7(A3c, Lx, Ly)]) 
    A4c = idx(xA4c, yA4c, Lx, Ly)
    A4 = set().union(*[p_ind[i] for i in plaq7(A4c, Lx, Ly)])
    A5c = idx(xA5c, yA5c, Lx, Ly)
    A5 = set().union(*[p_ind[i] for i in plaq7(A5c, Lx, Ly)])
    A6c = idx(xA6c, yA6c, Lx, Ly)
    A6 = set().union(*[p_ind[i] for i in plaq7(A6c, Lx, Ly)])

    print(f"A1c={A1c} at (x={xA1c}, y={yA1c}), plaquettes in A1: {sorted(A1)}", flush=True)
    print(f"A2c={A2c} at (x={xA2c}, y={yA2c}), plaquettes in A2: {sorted(A2)}", flush=True)
    print(f"A3c={A3c} at (x={xA3c}, y={yA3c}), plaquettes in A3: {sorted(A3)}", flush=True)
    print(f"A4c={A4c} at (x={xA4c}, y={yA4c}), plaquettes in A4: {sorted(A4)}", flush=True)
    print(f"A5c={A5c} at (x={xA5c}, y={yA5c}), plaquettes in A5: {sorted(A5)}", flush=True)
    print(f"A6c={A6c} at (x={xA6c}, y={yA6c}), plaquettes in A6: {sorted(A6)}", flush=True)

    # subsystem B is two overlapping plaq7 regions, centered at Bc1 and Bc2
    if yc % 2 == 0:
        xB1c, yB1c = (xc + 1) % Lx, (yc - 1) % Ly
        xB2c, yB2c = (xc + 1) % Lx, (yc - 3) % Ly
        xB3c, yB3c =  xc      % Lx, (yc - 4) % Ly
        xB4c, yB4c = (xc - 2) % Lx, (yc - 5) % Ly
        xB5c, yB5c = (xc - 3) % Lx, (yc - 4) % Ly
        xB6c, yB6c = (xc - 5) % Lx, (yc - 3) % Ly
    else:
        xB1c, yB1c = (xc + 2) % Lx, (yc - 1) % Ly
        xB2c, yB2c = (xc + 2) % Lx, (yc - 3) % Ly
        xB3c, yB3c = xc % Lx, (yc - 4) % Ly
        xB4c, yB4c = (xc - 1) % Lx, (yc - 5) % Ly
        xB5c, yB5c = (xc - 3) % Lx, (yc - 4) % Ly
        xB6c, yB6c = (xc - 4) % Lx, (yc - 3) % Ly

    B1c = idx(xB1c, yB1c, Lx, Ly)
    B1 = set().union(*[p_ind[i] for i in plaq7(B1c, Lx, Ly)])
    B2c = idx(xB2c, yB2c, Lx, Ly)
    B2 = set().union(*[p_ind[i] for i in plaq7(B2c, Lx, Ly)])
    B3c = idx(xB3c, yB3c, Lx, Ly)
    B3 = set().union(*[p_ind[i] for i in plaq7(B3c, Lx, Ly)])
    B4c = idx(xB4c, yB4c, Lx, Ly)
    B4 = set().union(*[p_ind[i] for i in plaq7(B4c, Lx, Ly)])
    B5c = idx(xB5c, yB5c, Lx, Ly)
    B5 = set().union(*[p_ind[i] for i in plaq7(B5c, Lx, Ly)])
    B6c = idx(xB6c, yB6c, Lx, Ly)
    B6 = set().union(*[p_ind[i] for i in plaq7(B6c, Lx, Ly)])

    print(f"B1c={B1c} at (x={xB1c}, y={yB1c}), plaquettes in B1: {sorted(B1)}", flush=True)
    print(f"B2c={B2c} at (x={xB2c}, y={yB2c}), plaquettes in B2: {sorted(B2)}", flush=True)
    print(f"B3c={B3c} at (x={xB3c}, y={yB3c}), plaquettes in B3: {sorted(B3)}", flush=True)
    print(f"B4c={B4c} at (x={xB4c}, y={yB4c}), plaquettes in B4: {sorted(B4)}", flush=True)
    print(f"B5c={B5c} at (x={xB5c}, y={yB5c}), plaquettes in B5: {sorted(B5)}", flush=True)
    print(f"B6c={B6c} at (x={xB6c}, y={yB6c}), plaquettes in B6: {sorted(B6)}", flush=True)

    # subsystem C is two overlapping plaq7 regions, centered at Cc1 and Cc2
    if yc % 2 == 0:
        xC1c, yC1c = xc    , (yc + 2) % Ly
        xC2c, yC2c = (xc + 1) % Lx, (yc + 3) % Ly
        xC3c, yC3c = (xc + 3) % Lx, (yc + 2) % Ly
        xC4c, yC4c = (xc + 4) % Lx, (yc + 1) % Ly
        xC5c, yC5c = (xc + 4) % Lx, (yc - 1) % Ly
        xC6c, yC6c = (xc + 4) % Lx, (yc - 3) % Ly
    else:
        xC1c, yC1c = xc    , (yc + 2) % Ly
        xC2c, yC2c = (xc + 2) % Lx, (yc + 3) % Ly
        xC3c, yC3c = (xc + 3) % Lx, (yc + 2) % Ly
        xC4c, yC4c = (xc + 5) % Lx, (yc + 1) % Ly
        xC5c, yC5c = (xc + 5) % Lx, (yc - 1) % Ly
        xC6c, yC6c = (xc + 5) % Lx, (yc - 3) % Ly

    C1c = idx(xC1c, yC1c, Lx, Ly)
    C1 = set().union(*[p_ind[i] for i in plaq7(C1c, Lx, Ly)])
    C2c = idx(xC2c, yC2c, Lx, Ly)
    C2 = set().union(*[p_ind[i] for i in plaq7(C2c, Lx, Ly)])
    C3c = idx(xC3c, yC3c, Lx, Ly)
    C3 = set().union(*[p_ind[i] for i in plaq7(C3c, Lx, Ly)])
    C4c = idx(xC4c, yC4c, Lx, Ly)
    C4 = set().union(*[p_ind[i] for i in plaq7(C4c, Lx, Ly)])
    C5c = idx(xC5c, yC5c, Lx, Ly)
    C5 = set().union(*[p_ind[i] for i in plaq7(C5c, Lx, Ly)])
    C6c = idx(xC6c, yC6c, Lx, Ly)
    C6 = set().union(*[p_ind[i] for i in plaq7(C6c, Lx, Ly)])

    print(f"C1c={C1c} at (x={xC1c}, y={yC1c}), plaquettes in C1: {sorted(C1)}", flush=True)
    print(f"C2c={C2c} at (x={xC2c}, y={yC2c}), plaquettes in C2: {sorted(C2)}", flush=True)
    print(f"C3c={C3c} at (x={xC3c}, y={yC3c}), plaquettes in C3: {sorted(C3)}", flush=True)
    print(f"C4c={C4c} at (x={xC4c}, y={yC4c}), plaquettes in C4: {sorted(C4)}", flush=True)
    print(f"C5c={C5c} at (x={xC5c}, y={yC5c}), plaquettes in C5: {sorted(C5)}", flush=True)
    print(f"C6c={C6c} at (x={xC6c}, y={yC6c}), plaquettes in C6: {sorted(C6)}", flush=True)

    sys.exit(0)



    # plaq 7
    # A = A1 
    # B = B1 
    # C = C1  

    # plaq12
    A = A1 | A2
    B = B1 | B2
    C = C1 | C2 

    # plaq17
    A = A1 | A2 | A3
    B = B1 | B2 | B3
    C = C1 | C2 | C3

    # plaq22
    A = A1 | A2 | A3 | A4
    B = B1 | B2 | B3 | B4
    C = C1 | C2 | C3 | C4

    # plaq27
    A = A1 | A2 | A3 | A4 | A5
    B = B1 | B2 | B3 | B4 | B5
    C = C1 | C2 | C3 | C4 | C5

    # plaq32
    A = A1 | A2 | A3 | A4 | A5 | A6
    B = B1 | B2 | B3 | B4 | B5 | B6
    C = C1 | C2 | C3 | C4 | C5 | C6

    B = B - A
    C = C - (A | B)

    AB = A | B
    BC = B | C
    CA = C | A
    ABC = A | B | C

    A_idx = np.array(sorted(A), dtype=np.int64)
    B_idx = np.array(sorted(B), dtype=np.int64)
    C_idx = np.array(sorted(C), dtype=np.int64)
    AB_idx = np.array(sorted(AB), dtype=np.int64)
    BC_idx = np.array(sorted(BC), dtype=np.int64)
    CA_idx = np.array(sorted(CA), dtype=np.int64)
    ABC_idx = np.array(sorted(ABC), dtype=np.int64)

    G_csr = symplectic_metric(Lv)
    link_colors = selected_link_colors()
    directions = selected_directions()

    string_lengths = list(range(1, MAX_STRING_LENGTH + 1))

    # if rank == 0:
    #     print("selected link colors:", link_colors, flush=True)
    #     print("selected directions:", directions, flush=True)
    #     for link_color in link_colors:
    #         for dname in directions:
    #             plaquette_name = plaquette_color_for_link_direction(link_color, dname)
    #             print(
    #                 f"  link={link_color:5s} dir={dname:8s} <- plaquette={plaquette_name}",
    #                 flush=True,
    #             )

    ops = create_string_loop_operators(Lx, Ly, zz_links_by_color, MAX_STRING_LENGTH, link_colors, directions)

    # if DEBUG_PRINT_OPS and rank == 0:
    #     np.set_printoptions(linewidth=200, threshold=np.inf)
    #     for link_color in link_colors:
    #         for dname in directions:
    #             plaquette_name = ops[link_color][dname]["plaquette_color"]
    #             print(f"\nlink={link_color}, direction={dname}, plaquette={plaquette_name}")
    #             for name in OP_NAMES:
    #                 M = ops[link_color][dname][name].toarray().T
    #                 for j, vec in enumerate(M):
    #                     print(f"{name}[{j}] nonzero = {np.flatnonzero(vec)}")

    # TEN用
    local_TEN = np.zeros(Np, dtype=np.float64)
    local_TEN_sq = np.zeros(Np, dtype=np.float64)

    # 最大クラスターサイズ用
    local_largest = np.zeros(Np, dtype=np.float64)
    local_largest_sq = np.zeros(Np, dtype=np.float64)

    # string 用
    local_R2 = make_nested_accumulator(R2_string_KEYS, Np, link_colors, directions, string_lengths)
    local_R2_sq = make_nested_accumulator(R2_string_KEYS, Np, link_colors, directions, string_lengths)

    # loop 用
    local_R2_loop = make_loop_accumulator(R2_loop_KEYS, Np, link_colors, directions)
    local_R2_loop_sq = make_loop_accumulator(R2_loop_KEYS, Np, link_colors, directions)

    count_local = np.zeros(Np, dtype=np.int64)

    for isamp in range(rank, Nd, size):
        # rng = np.random.default_rng(BASE_SEED + isamp)
        # u = rng.random(Lh)
        # order = np.argsort(u)
        # u_sorted = u[order]


        chosen_idx, uf, p_occ, largest = explosive_percolation_zz(
        zz_dephasing,
        Lx,
        Ly,
        ntrial = ntrial,
        seed = BASE_SEED + isamp,
        )

        # order = np.array(chosen_idx)
        # u_sorted = np.array(p_occ)

        MR, nsdd = initial_stabilizer_state(
            Lv, Lx, Ly, p_ind,
            zero_x=[Lh - 2, Lh - 1],
            zero_z=[Lh - 2, Lh - 1],
        )
        ptr = 0

        MR0 = MR.copy()
        nsdd0 = nsdd

        # sanity check at p=0
        NA0 = negativity_E_fast(MR0, nsdd0, Lv, A_idx)
        NB0 = negativity_E_fast(MR0, nsdd0, Lv, B_idx)
        NC0 = negativity_E_fast(MR0, nsdd0, Lv, C_idx)
        NAB0 = negativity_E_fast(MR0, nsdd0, Lv, AB_idx)
        NBC0 = negativity_E_fast(MR0, nsdd0, Lv, BC_idx)
        NCA0 = negativity_E_fast(MR0, nsdd0, Lv, CA_idx)
        NABC0 = negativity_E_fast(MR0, nsdd0, Lv, ABC_idx)
        TEN0 = NA0 + NB0 + NC0 - NAB0 - NBC0 - NCA0 + NABC0

        if rank == 0:
            print(f"p=0 sanity check: TEN={TEN0:.2f}", flush=True)

        for ip, p in enumerate(p_list):
            while ptr < len(chosen_idx) and p_occ[ptr] <= p:
                nsdd = dephasing_dlinkZZ_inplace(MR, int(chosen_idx[ptr]), zz_dephasing, nsdd)

                # link_id = int(chosen_idx[ptr])

                # q0, q1 = zz_dephasing[link_id]

                # print(
                #     f"[DEC] ptr={ptr:4d} "
                #     f"link={link_id:4d} "
                #     f"qubits=({q0:4d},{q1:4d})"
                # )

                ptr += 1

            if ptr > 0:
                largest_now = largest[ptr - 1]
            else:
                largest_now = 0.0

            active_MR = MR[:nsdd].copy()

            NA = float(negativity_E_fast(active_MR, nsdd, Lv, A_idx))
            NB = float(negativity_E_fast(active_MR, nsdd, Lv, B_idx))
            NC = float(negativity_E_fast(active_MR, nsdd, Lv, C_idx))
            NAB = float(negativity_E_fast(active_MR, nsdd, Lv, AB_idx))
            NBC = float(negativity_E_fast(active_MR, nsdd, Lv, BC_idx))
            NCA = float(negativity_E_fast(active_MR, nsdd, Lv, CA_idx))
            NABC = float(negativity_E_fast(active_MR, nsdd, Lv, ABC_idx))
            TEN = NA + NB + NC - NAB - NBC - NCA + NABC

            local_TEN[ip] += TEN
            local_TEN_sq[ip] += TEN**2

            local_largest[ip] += largest_now
            local_largest_sq[ip] += largest_now**2

            if rank == 0:
                print(
                    f"rank={rank} isamp={isamp} ip={ip} p={p:.2f} "
                    f"TEN={TEN} nsdd={nsdd}", flush=True,
                )

            if DEBUG_PRINT_MR and rank == 0:
                print(f"\nsample={isamp}, ip={ip}, p={p:.3f}, nsdd={nsdd}, rank={rank_mod2(active_MR)}")
                print(active_MR)

            detail = should_print_details(ip)

            for link_color in link_colors:
                for dname in directions:

                    # ----------------------------
                    # string
                    # ----------------------------
                    for slen in range(1, MAX_STRING_LENGTH + 1):

                        current_ops = ops[link_color][dname][slen]

                        for string_name, dep_string_key in zip(R2_string_KEYS, DEP_string_KEYS):
                            op = current_ops[string_name]

                            value = renyi2_fraction(active_MR, G_csr, op)

                            local_R2[link_color][dname][string_name][slen][ip] += value
                            local_R2_sq[link_color][dname][string_name][slen][ip] += value**2

                            if rank == 0:
                                print(
                                    f"STRING "
                                    f"ip={ip:3d} p={p:.3f} sample={isamp:3d} "
                                    f"link={link_color:6s} dir={dname:8s} len={slen:3d} "
                                    f"{string_name:8s}={value:.4f} ",
                                    #f"{dep_string_key:12s}={frac:.4f} "
                                    #f"dependent={n_dep:3d}/{n_ops:3d}",
                                    flush=True,
                                )

                    # ----------------------------
                    # loop
                    # ----------------------------
                    current_loop_ops = ops[link_color][dname]["loop"]

                    for loop_name, dep_loop_key in zip(R2_loop_KEYS, DEP_loop_KEYS):
                        op = current_loop_ops[loop_name]

                        value = renyi2_fraction(active_MR, G_csr, op)

                        local_R2_loop[link_color][dname][loop_name][ip] += value
                        local_R2_loop_sq[link_color][dname][loop_name][ip] += value**2

                        if rank == 0:
                            print(
                                f"LOOP   "
                                f"ip={ip:3d} p={p:.3f} sample={isamp:3d} "
                                f"link={link_color:6s} dir={dname:8s} "
                                f"{loop_name:8s}={value:.4f} ",
                                flush=True,
                            )

            count_local[ip] += 1

    count_global_float = np.zeros(Np, dtype=np.float64)
    comm.Allreduce(count_local.astype(np.float64), count_global_float, op=MPI.SUM)
    count_global = count_global_float.astype(np.int64)

    TEN_stats = reduce_stats_1d(local_TEN, local_TEN_sq, count_global)

    largest_stats = reduce_stats_1d(local_largest, local_largest_sq, count_global)

    R2_stats = reduce_stats_nested(
        local_R2,
        local_R2_sq,
        count_global,
        link_colors,
        directions,
    )

    R2_loop_stats = reduce_stats_loop(
        local_R2_loop,
        local_R2_loop_sq,
        count_global,
        link_colors,
        directions,
    )

    save_results(
        p_list,
        count_global,
        TEN_stats,
        largest_stats,
        R2_stats,
        R2_loop_stats,
        link_colors,
        directions,
    )

    if rank == 0:
        print(f"total elapsed: {time.time() - start:.2f} s", flush=True)


def reduce_stats_1d(
    local_sum: np.ndarray,
    local_sum_sq: np.ndarray,
    count_global: np.ndarray,
) -> dict:
    count = count_global.astype(np.float64)

    global_sum = np.zeros_like(local_sum, dtype=np.float64)
    global_sum_sq = np.zeros_like(local_sum_sq, dtype=np.float64)

    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(local_sum_sq, global_sum_sq, op=MPI.SUM)

    mean = global_sum / count

    var = np.full_like(mean, np.nan, dtype=np.float64)
    se = np.full_like(mean, np.nan, dtype=np.float64)

    mask = count > 1
    var[mask] = (global_sum_sq[mask] - count[mask] * mean[mask]**2) / (count[mask] - 1.0)
    var[mask] = np.maximum(var[mask], 0.0)
    se[mask] = np.sqrt(var[mask] / count[mask])

    return {
        "mean": mean,
        "var": var,
        "se": se,
    }

def reduce_stats_nested(
    local_sum: dict,
    local_sum_sq: dict,
    count_global: np.ndarray,
    link_colors: list[str],
    directions: list[str],
) -> dict:
    """
    For string observables:
      stats[link_color][dname][key][slen]["mean"/"var"/"se"]
    """

    stats = {}
    count = count_global.astype(np.float64)

    for link_color in link_colors:
        stats[link_color] = {}

        for dname in directions:
            plaquette_name = plaquette_color_for_link_direction(link_color, dname)

            stats[link_color][dname] = {
                "plaquette_color": plaquette_name
            }

            for key in local_sum[link_color][dname].keys():
                stats[link_color][dname][key] = {}

                for slen, arr in local_sum[link_color][dname][key].items():

                    global_sum = np.zeros_like(arr, dtype=np.float64)
                    global_sum_sq = np.zeros_like(arr, dtype=np.float64)

                    comm.Allreduce(arr, global_sum, op=MPI.SUM)
                    comm.Allreduce(
                        local_sum_sq[link_color][dname][key][slen],
                        global_sum_sq,
                        op=MPI.SUM,
                    )

                    mean = global_sum / count

                    var = np.full_like(mean, np.nan, dtype=np.float64)
                    se = np.full_like(mean, np.nan, dtype=np.float64)

                    mask = count > 1
                    var[mask] = (
                        global_sum_sq[mask]
                        - count[mask] * mean[mask] ** 2
                    ) / (count[mask] - 1.0)

                    var[mask] = np.maximum(var[mask], 0.0)
                    se[mask] = np.sqrt(var[mask] / count[mask])

                    stats[link_color][dname][key][slen] = {
                        "mean": mean,
                        "var": var,
                        "se": se,
                    }

    return stats

def reduce_stats_loop(
    local_sum: dict,
    local_sum_sq: dict,
    count_global: np.ndarray,
    link_colors: list[str],
    directions: list[str],
) -> dict:

    stats = {}
    count = count_global.astype(np.float64)

    for link_color in link_colors:
        stats[link_color] = {}

        for dname in directions:
            plaquette_name = plaquette_color_for_link_direction(link_color, dname)

            stats[link_color][dname] = {
                "plaquette_color": plaquette_name
            }

            for key, arr in local_sum[link_color][dname].items():

                global_sum = np.zeros_like(arr, dtype=np.float64)
                global_sum_sq = np.zeros_like(arr, dtype=np.float64)

                comm.Allreduce(arr, global_sum, op=MPI.SUM)
                comm.Allreduce(
                    local_sum_sq[link_color][dname][key],
                    global_sum_sq,
                    op=MPI.SUM,
                )

                mean = global_sum / count

                var = np.full_like(mean, np.nan, dtype=np.float64)
                se = np.full_like(mean, np.nan, dtype=np.float64)

                mask = count > 1
                var[mask] = (
                    global_sum_sq[mask]
                    - count[mask] * mean[mask] ** 2
                ) / (count[mask] - 1.0)

                var[mask] = np.maximum(var[mask], 0.0)
                se[mask] = np.sqrt(var[mask] / count[mask])

                stats[link_color][dname][key] = {
                    "mean": mean,
                    "var": var,
                    "se": se,
                }

    return stats

def save_results(
    p_list: np.ndarray,
    count_global: np.ndarray,

    TEN_stats: dict,
    largest_stats: dict,

    R2_stats: dict,

    R2_loop_stats: dict,

    link_colors: list[str],
    directions: list[str],
) -> None:

    stem = f"Nd_{Nd}_Lx_{Lx}_Ly_{Ly}_Np_{Np}_ntrial_{ntrial}_ps_{ps}_pl_{pl}_{cname}"

    base = Path(stem)

    if rank == 0:
        base.mkdir(exist_ok=True)

    comm.Barrier()

    if rank != 0:
        return

    save_dict = {
        "p": p_list,
        "count_global": count_global,

        "Nd": Nd,
        "Np": Np,
        "Lx": Lx,
        "Ly": Ly,

        "MAX_STRING_LENGTH": MAX_STRING_LENGTH,

        "DEPHASING_LINK_COLOR": DEPHASING_LINK_COLOR,

        "SELECT_LINK_COLORS": np.array(link_colors),
        "SELECT_DIRECTIONS": np.array(directions),

        # ------------------------
        # TEN
        # ------------------------
        "TEN_mean": TEN_stats["mean"],
        "TEN_var": TEN_stats["var"],
        "TEN_se": TEN_stats["se"],

        "largest_mean": largest_stats["mean"],
        "largest_var": largest_stats["var"],
        "largest_se": largest_stats["se"],
    }

    # =====================================================
    # STRING and LOOP
    # =====================================================
    for link_color in link_colors:
        for dname in directions:

            plaquette_name = plaquette_color_for_link_direction(
                link_color,
                dname,
            )

            prefix = (
                f"link_{link_color}"
                f"_dir_{dname}"
                f"_plaquette_{plaquette_name}"
            )

            # ------------------------
            # R2 string
            # ------------------------
            for key in R2_string_KEYS:

                for slen in range(1, MAX_STRING_LENGTH + 1):

                    for stat in ["mean", "var", "se"]:

                        save_dict[
                            f"{prefix}_{key}_len{slen}_{stat}"
                        ] = (
                            R2_stats[link_color][dname][key][slen][stat]
                        )

            # ------------------------
            # R2 loop
            # ------------------------
            for key in R2_loop_KEYS:

                for stat in ["mean", "var", "se"]:

                    save_dict[
                        f"{prefix}_{key}_{stat}"
                    ] = (
                        R2_loop_stats[link_color][dname][key][stat]
                    )


    out = base / f"{stem}_avg.npz"

    np.savez(out, **save_dict)

    print(f"saved: {out}", flush=True)

def print_summary(
    p_list: np.ndarray,
    R2_stats: dict,
    #dep_stats: dict,
    link_colors: list[str],
    directions: list[str],
) -> None:
    for ip, p in enumerate(p_list):
        print(f"\np={p:.3f}", flush=True)
        for link_color in link_colors:
            for dname in directions:
                plaquette_name = plaquette_color_for_link_direction(link_color, dname)
                vals = R2_stats[link_color][dname]
                print(
                    f"    link={link_color:5s} dir={dname:8s} plaquette={plaquette_name:5s} "
                    f"R2x={vals['R2x']['mean'][ip]:.4f}±{vals['R2x']['se'][ip]:.4f} "
                    f"R2z={vals['R2z']['mean'][ip]:.4f}±{vals['R2z']['se'][ip]:.4f} "
                    f"R2x_loop={vals['R2x_loop']['mean'][ip]:.4f}±{vals['R2x_loop']['se'][ip]:.4f} "
                    f"R2z_loop={vals['R2z_loop']['mean'][ip]:.4f}±{vals['R2z_loop']['se'][ip]:.4f} ",
                    flush=True,
                )

if __name__ == "__main__":
    main()
