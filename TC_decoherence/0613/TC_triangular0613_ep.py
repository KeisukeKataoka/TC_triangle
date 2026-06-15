# initial Z px=0 (px,pg)=(1,0) to (px,pg)=(0,1)

from __future__ import print_function, division

import time
from dataclasses import dataclass

import numpy as np
from numba import njit
from mpi4py import MPI
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


# ============================================================
# MPI / random seed / print options
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#seed = int(time.time()) + rank * 10000
#np.random.seed(seed)
GLOBAL_SEED = 123456  # 一貫性を持たせるため任意の定数
np.random.seed(GLOBAL_SEED + rank)

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


# ============================================================
# Parameters
# ============================================================
@dataclass(frozen=True)
class Params:
    ps: float = 0.0
    pl: float = 1.0
    Np: int = 21
    Nd: int = 100  # sample number

    # case1 = X, case2 = Z, case3 = ground state of TC
    case: int = 3

    Lx: int = 12
    Ly: int = 12

    # TEN subsystem origin
    Ax: int = 2
    Ay: int = 2

    # EP order construction の試行回数。大きいほどループを避けるが、計算コストも上がる。
    ntrial: int = 120

# ============================================================
# Union-Find (for loop detection)
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

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return False

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def score_product(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return self.size[ra] ** 2

        return self.size[ra] * self.size[rb]

    def largest(self):
        roots = [i for i in range(len(self.parent)) if self.find(i) == i]
        return max(self.size[r] for r in roots)




def link_endpoints(q, Lx, Ly):
    Lv = Lx * Ly

    if q < Lv:
        s = q
        x, y = inv_idx(s, Lx, Ly)
        a = idx(x, y, Lx, Ly)
        b = idx(x + 1, y, Lx, Ly)

    elif q < 2 * Lv:
        s = q - Lv
        x, y = inv_idx(s, Lx, Ly)
        a = idx(x, y, Lx, Ly)
        b = idx(x, y + 1, Lx, Ly)

    else:
        s = q - 2 * Lv
        x, y = inv_idx(s, Lx, Ly)
        a = idx(x, y, Lx, Ly)
        b = idx(x + 1, y + 1, Lx, Ly)

    return a, b


def make_ep_order_triangular(Lx, Ly, ntrial=2):
    Lv = Lx * Ly
    L = 3 * Lv

    uf = UnionFind(Lv)

    remaining = list(range(L))
    order = []
    largest_cluster = []

    for step in range(L):
        m = min(ntrial, len(remaining))
        cand_pos = np.random.choice(len(remaining), size=m, replace=False)

        best_pos = None
        best_score = None

        for pos in cand_pos:
            q = remaining[pos]
            a, b = link_endpoints(q, Lx, Ly)
            score = uf.score_product(a, b)

            if best_score is None or score < best_score:
                best_score = score
                best_pos = pos

        q_best = remaining.pop(best_pos)
        a, b = link_endpoints(q_best, Lx, Ly)
        uf.union(a, b)

        order.append(q_best)
        largest_cluster.append(uf.largest())

    return np.array(order), np.array(largest_cluster)

def link_xy(q, Lx, Ly):
    Lv = Lx * Ly

    if q < Lv:
        s = q
        x, y = inv_idx(s, Lx, Ly)
        return x, y, x + 1, y

    elif q < 2 * Lv:
        s = q - Lv
        x, y = inv_idx(s, Lx, Ly)
        return x, y, x, y + 1

    else:
        s = q - 2 * Lv
        x, y = inv_idx(s, Lx, Ly)
        return x, y, x + 1, y + 1


def plot_ep_snapshot_cluster(order, Lx, Ly, p, filename=None):
    Lv = Lx * Ly
    L = 3 * Lv

    n_occ = int(np.floor(p * L))
    occ_links = order[:n_occ]

    # occupied links だけでクラスターを再構成
    uf = UnionFind(Lv)
    for q in occ_links:
        a, b = link_endpoints(int(q), Lx, Ly)
        uf.union(a, b)

    # root ごとに色番号を割り当てる
    roots = sorted(set(uf.find(i) for i in range(Lv)))
    root_to_color = {r: k for k, r in enumerate(roots)}

    plt.figure(figsize=(6, 6))

    # background
    for q in range(L):
        x1, y1, x2, y2 = link_xy(q, Lx, Ly)
        plt.plot([x1, x2], [y1, y2], "-", lw=0.4, alpha=0.15, color="gray")

    # occupied links: cluster root ごとに色
    for q in occ_links:
        q = int(q)
        a, b = link_endpoints(q, Lx, Ly)
        root = uf.find(a)
        c = root_to_color[root]

        x1, y1, x2, y2 = link_xy(q, Lx, Ly)
        plt.plot([x1, x2], [y1, y2], "-", lw=2.5, color=f"C{c % 10}")

    plt.scatter(
        [i % Lx for i in range(Lv)],
        [i // Lx for i in range(Lv)],
        s=12,
        color="k",
        zorder=3,
    )

    plt.gca().set_aspect("equal")
    plt.xlim(-0.5, Lx + 0.5)
    plt.ylim(-0.5, Ly + 0.5)
    plt.title(f"EP cluster snapshot, p={p:.2f}, n_occ={n_occ}/{L}")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()



# ============================================================
# Lattice utilities
# ============================================================
def idx(x, y, Lx, Ly):
    return (x % Lx) + Lx * (y % Ly)


def inv_idx(i, Lx, Ly):
    return i % Lx, i // Lx


def dir_x(x, y, Lx, Lv):
    return x + Lx * y


def dir_y(x, y, Lx, Lv):
    return x + Lx * y + Lv


def dir_xy(x, y, Lx, Lv):
    return x + Lx * y + 2 * Lv


def create_transformation_pv_link(Lv, Lx, Ly):
    #############################################
    ############ plaquette part #################
    #############################################
    p_indd = np.zeros((2 * Lx * Ly, 3), dtype="uint32")
    for iy in range(Ly):
        for ix in range(Lx):
            # 下三角
            ic = (ix % Lx) + Lx * (iy % Ly)
            ipx = (ix + 1) % Lx
            ipy = (iy + 1) % Ly
            p_indd[ic][0] = dir_x(ix, iy, Lx, Lv)
            p_indd[ic][1] = dir_y(ipx, iy, Lx, Lv)
            p_indd[ic][2] = dir_xy(ix, iy, Lx, Lv)

            # 上三角
            p_indd[ic + Lx * Ly][0] = dir_xy(ix, iy, Lx, Lv)
            p_indd[ic + Lx * Ly][1] = dir_x(ix, ipy, Lx, Lv)
            p_indd[ic + Lx * Ly][2] = dir_y(ix, iy, Lx, Lv)

    ############################################
    ############ vertex part ###################
    ############################################
    v_indd = np.zeros((Lx * Ly, 6), dtype="uint32")
    for iy in range(Ly):
        for ix in range(Lx):
            ic = (ix % Lx) + Lx * (iy % Ly)
            imx = (ix - 1) % Lx
            imy = (iy - 1) % Ly
            v_indd[ic][0] = dir_x(ix, iy, Lx, Lv)
            v_indd[ic][1] = dir_xy(ix, iy, Lx, Lv)
            v_indd[ic][2] = dir_y(ix, iy, Lx, Lv)
            v_indd[ic][3] = dir_x(imx, iy, Lx, Lv)
            v_indd[ic][4] = dir_xy(imx, imy, Lx, Lv)
            v_indd[ic][5] = dir_y(ix, imy, Lx, Lv)

    #print(v_indd)
    #print(p_indd)
    return p_indd, v_indd


# ============================================================
# Stabilizer construction / logical operators
# ============================================================
def initial_stabilizer_state(case, Ld, Lxd, Lyd, v_ind, p_ind):
    # Ld: total # of qubits
    MRi = np.zeros((Ld, 2 * Ld), dtype="uint8")

    if case == 1:
        # Case I: X product state
        for ii in range(Ld):
            MRi[ii][ii] = 1  # X

    if case == 2:
        # Case II: Z product state
        for ii in range(Ld):
            MRi[ii][Ld + ii] = 1  # Z

    if case == 3:
        # Case III: unique g.s. of TC
        # plaquette stabilizers ZZZ
        for ip in range(2 * Lxd * Lyd - 1):
            MRi[ip][Ld + p_ind[ip][0]] = 1  # Z0
            MRi[ip][Ld + p_ind[ip][1]] = 1  # Z1
            MRi[ip][Ld + p_ind[ip][2]] = 1  # Z2
            # 独立ではない最後のBpだけ取らない

        # vertex stabilizers XXXXXX
        for iv in range(Lxd * Lyd - 1):
            ivs = iv + 2 * Lxd * Lyd - 1  # note that iv and ivs are different.
            MRi[ivs][v_ind[iv][0]] = 1  # X0
            MRi[ivs][v_ind[iv][1]] = 1  # X1
            MRi[ivs][v_ind[iv][2]] = 1  # X2
            MRi[ivs][v_ind[iv][3]] = 1  # X3
            MRi[ivs][v_ind[iv][4]] = 1  # X4
            MRi[ivs][v_ind[iv][5]] = 1  # X5
            # 独立ではない最後のAvだけ取らない

        MRi[Ld - 2] = 0
        MRi[Ld - 1] = 0

        # Add logical operators
        # x-direction loop logical X
        # for kk in range(Lxd):
        #     iv=(kk%Lx)+Lx*(0%Lyd)
        #     MRi[Ld-2][v_ind[iv][0]] = 1
        # y-direction loop logical X
        # for kk in range(Lyd):
        #     iv=(0%Lx)+Lx*(kk%Lyd)
        #     MRi[Ld-1][v_ind[iv][2]] = 0
        # x-direction loop logical Z
        # for kk in range(Lxd):
        #     iv=(kk%Lx)+Lx*(0%Lyd)
        #     MRi[Ld-2][Ld+v_ind[iv][0]] = 1
        # y-direction loop logical Z
        # for kk in range(Lyd):
        #     iv=(0%Lx)+Lx*(kk%Lyd)
        #     MRi[Ld-1][Ld+v_ind[iv][2]] = 1

    #print(MRi)
    return MRi


def make_logical_ops(Ld, Lxd, Lyd, v_ind):
    ops = {}

    op = np.zeros(2 * Ld, dtype=np.uint8)
    for kk in range(Lxd):
        iv = kk + Lxd * 0
        op[v_ind[iv][1]] = 1
        op[v_ind[iv][2]] = 1
    ops["Xx"] = op

    op = np.zeros(2 * Ld, dtype=np.uint8)
    for kk in range(Lyd):
        iv = 0 + Lxd * kk
        op[v_ind[iv][0]] = 1
        op[v_ind[iv][1]] = 1
    ops["Xy"] = op

    op = np.zeros(2 * Ld, dtype=np.uint8)
    for kk in range(Lxd):
        iv = kk + Lxd * 0
        op[Ld + v_ind[iv][0]] = 1
    ops["Zx"] = op

    op = np.zeros(2 * Ld, dtype=np.uint8)
    for kk in range(Lyd):
        iv = 0 + Lxd * kk
        op[Ld + v_ind[iv][2]] = 1
    ops["Zy"] = op

    return ops


# ============================================================
# Dephasing update
# ============================================================
def dephasing_linkZ_inplace(dMR, q, nsdd):
    active = dMR[:nsdd]

    # Z_q と反可換する stabilizer は X_q=1 のもの
    bad = np.flatnonzero(active[:, q] == 1)

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


def dephasing_linkZ(MR, q, nsdd):
    """Copy版。debugしやすいように、行削除後のMRとnsddを返す。"""
    active = MR[:nsdd].copy()
    L2 = active.shape[1] // 2

    bad = np.flatnonzero(active[:, q] == 1)

    # print("q =", q)
    # print("bad =", bad)
    # for b in bad:
    #     print(
    #         b,
    #         "X=", np.where(active[b, :L])[0],
    #         "Z=", np.where(active[b, L:])[0],
    #         "active[b,q]=", active[b, q],
    #     )

    loop_rows = {nsdd - 2, nsdd - 1}
    flag = (bad.size > 0) and set(bad).issubset(loop_rows)

    # if flag:
    #     print("Xx = 0")
    # else:
    #     print("Xx = 1")
    # print("q =", q)
    # print("bad =", bad)
    # for b in bad:
    #     print(
    #         "bad row", b,
    #         "X=", np.where(active[b, :L2])[0],
    #         "Z=", np.where(active[b, L2:])[0],
    #     )

    if bad.size == 0:
        return active, nsdd

    k0 = int(bad[0])
    pivot = active[k0].copy()

    #print("delete pivot row =", k0)

    if bad.size > 1:
        active[bad[1:]] ^= pivot

    active = np.delete(active, k0, axis=0)
    nsdd1 = nsdd - 1

    return active, nsdd1


def print_stabilizers(MR, title="MR"):
    L2 = MR.shape[1] // 2
    print(title)
    for i, row in enumerate(MR):
        x = np.where(row[:L2])[0].tolist()
        z = np.where(row[L2:])[0].tolist()
        print(f"{i:2d}: X={x} Z={z}")


def dephasing_linkZ_debug(MR, q):
    active = MR.copy()
    L2 = active.shape[1] // 2

    bad = np.flatnonzero(active[:, q] == 1)

    print("=" * 60)
    print(f"dephase link q = {q}")
    print(f"bad rows before = {bad.tolist()}")

    print_stabilizers(active, "BEFORE")

    if bad.size == 0:
        print("no stabilizer anticommutes with Z_q")
        print_stabilizers(active, "AFTER")
        return active

    k0 = int(bad[0])
    pivot = active[k0].copy()

    print(f"pivot row deleted = {k0}")
    print(f"pivot X={np.where(pivot[:L2])[0].tolist()} Z={np.where(pivot[L2:])[0].tolist()}")

    # どの行がどう変わるか
    for b in bad[1:]:
        before = active[b].copy()
        after = before ^ pivot
        print(f"row {b} updated: row {b} ^= pivot")
        print(f"  before X={np.where(before[:L2])[0].tolist()} Z={np.where(before[L2:])[0].tolist()}")
        print(f"  after  X={np.where(after[:L2])[0].tolist()} Z={np.where(after[L2:])[0].tolist()}")

    if bad.size > 1:
        active[bad[1:]] ^= pivot

    active = np.delete(active, k0, axis=0)

    print_stabilizers(active, "AFTER")
    return active


# def dephasing_linkZ(MR, q):
#     active = MR.copy()
#
#     bad = np.flatnonzero(active[:, q] == 1)
#
#     if bad.size == 0:
#         return active
#
#     k0 = int(bad[0])
#     pivot = active[k0].copy()
#
#     if bad.size > 1:
#         active[bad[1:]] ^= pivot
#
#     active = np.delete(active, k0, axis=0)
#
#     return active


# ============================================================
# Fast GF(2) linear algebra
# ============================================================
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
# Negativity / TEN
# ============================================================
def negativity_E_fast(MRdn: np.ndarray, nsdd2: int, Ld: int, sub_idx: np.ndarray) -> float:
    # Ld は total # of qubits (= L) を渡す
    active = MRdn[:nsdd2]

    X = active[:, sub_idx].astype(np.uint16, copy=False)
    Z = active[:, Ld + sub_idx].astype(np.uint16, copy=False)

    ComM = ((X @ Z.T) + (Z @ X.T)) & 1
    rankJ = rank_mod2_numba(ComM.astype(np.uint8, copy=False))
    return 0.5 * rankJ


def calc_TEN(MR, nsdd, L, regions):
    A_idx, B_idx, C_idx, AB_idx, BC_idx, CA_idx, ABC_idx = regions

    NA = negativity_E_fast(MR, nsdd, L, A_idx)
    NB = negativity_E_fast(MR, nsdd, L, B_idx)
    NC = negativity_E_fast(MR, nsdd, L, C_idx)
    NAB = negativity_E_fast(MR, nsdd, L, AB_idx)
    NBC = negativity_E_fast(MR, nsdd, L, BC_idx)
    NCA = negativity_E_fast(MR, nsdd, L, CA_idx)
    NABC = negativity_E_fast(MR, nsdd, L, ABC_idx)

    TEN = NA + NB + NC - NAB - NBC - NCA + NABC
    return TEN, (NA, NB, NC, NAB, NBC, NCA, NABC)


# https://arxiv.org/abs/2204.08489
def EE_cal(MRd, Asub_set, Ld):
    # partition
    #LA=Ld//2
    Asub = list(Asub_set)
    LA = len(Asub)
    MRdA = np.zeros((Ld, 2 * len(Asub)), dtype="uint8")
    #print("new",Asub)
    for k2 in range(len(Asub)):
        MRdA[:, k2] = MRd[:, Asub[k2]].copy()
        MRdA[:, LA + k2] = MRd[:, Ld + Asub[k2]].copy()
    #print(MRdA.shape)
    #print(MRdA)
    rankA = rank_mod2_numba(MRdA)
    #print(rankA)
    EEd = rankA - LA
    return EEd


# ============================================================
# Subsystems
# ============================================================
def subset2(Ld, Ax, Ay, Lx, Ly, p_indd):
    i = idx(Ax, Ay, Lx, Ly)
    Asub_set = set()
    for j in range(3):
        Asub_set.add(p_indd[i][j])  # 三角形
        Asub_set.add(p_indd[i + Lx * Ly][j])  # 逆三角形
        print(p_indd[i][j])
        print(p_indd[i + Lx * Ly][j])
    return Asub_set


def subset6(Ld, Ax, Ay, Lx, Ly, p_indd):
    Asub_set = set()
    Amx = (Ax - 1) % Lx
    Amy = (Ay - 1) % Ly
    i0 = idx(Ax, Ay, Lx, Ly)
    i1 = idx(Amx, Amy, Lx, Ly)
    i2 = idx(Amx, Ay, Lx, Ly)
    i3 = idx(Ax, Amy, Lx, Ly)
    for j in range(3):
        Asub_set.add(p_indd[i0][j])
        Asub_set.add(p_indd[i0 + Lx * Ly][j])
        Asub_set.add(p_indd[i1][j])
        Asub_set.add(p_indd[i1 + Lx * Ly][j])
        Asub_set.add(p_indd[i2][j])
        Asub_set.add(p_indd[i3 + Lx * Ly][j])
    return Asub_set


def make_regions(L, Lx, Ly, p_indd, Ax, Ay):
    Bx, By = Ax + 2, Ay + 1
    Cx, Cy = Ax + 1, Ay + 2

    A = subset6(L, Ax, Ay, Lx, Ly, p_indd)
    B = subset6(L, Bx, By, Lx, Ly, p_indd)
    C = subset6(L, Cx, Cy, Lx, Ly, p_indd)

    B = B - A
    C = C - (A | B)

    if rank == 0:
        print("A =", sorted(map(int, A)))
        print("B =", sorted(map(int, B)))
        print("C =", sorted(map(int, C)))

    AB = A | B
    BC = B | C
    CA = C | A
    ABC = A | B | C

    return (
        np.array(sorted(A), dtype=np.int64),
        np.array(sorted(B), dtype=np.int64),
        np.array(sorted(C), dtype=np.int64),
        np.array(sorted(AB), dtype=np.int64),
        np.array(sorted(BC), dtype=np.int64),
        np.array(sorted(CA), dtype=np.int64),
        np.array(sorted(ABC), dtype=np.int64),
    )


# ============================================================
# Renyi2 correlators / loop diagnostics
# ============================================================
def Renyi2_create(Ld, Lv, Lxd, Lyd):
    STx = np.zeros((2 * Ld, (Lxd - 1) * Lyd), dtype="uint8")
    STz = np.zeros((2 * Ld, (Lxd - 1) * Lyd), dtype="uint8")
    i = 0
    for piy in range(Lyd):
        # zig-zag
        for leng in range(1, Lxd):
            for ld in range(leng):
                ipd = ld + piy * Lxd
                ix, iy = inv_idx(ipd, Lxd, Lyd)
                di1 = dir_y(ix, iy, Lxd, Lv)
                di2 = dir_xy(ix, iy, Lxd, Lv)
                #print(piy,leng,diz1+Ld,diz2+Ld)
                STx[di1, i] = 1
                STx[di2, i] = 1
                STz[di1 + Ld, i] = 1
                STz[di2 + Ld, i] = 1
            i += 1
    return csr_matrix(STx), csr_matrix(STz)


def Renyi2_loop_create(Ld, Lv, Lxd, Lyd):
    STx = np.zeros((2 * Ld, Lyd), dtype="uint8")
    STz = np.zeros((2 * Ld, Lyd), dtype="uint8")
    i = 0
    for piy in range(Lyd):
        # zig-zag
        for ld in range(Lxd):
            ipd = ld + piy * Lxd
            ix, iy = inv_idx(ipd, Lxd, Lyd)
            di1 = dir_y(ix, iy, Lxd, Lv)
            di2 = dir_xy(ix, iy, Lxd, Lv)
            #print(piy,leng,diz1+Ld,diz2+Ld)
            STx[di1, i] = 1
            STx[di2, i] = 1
            STz[di1 + Ld, i] = 1
            STz[di2 + Ld, i] = 1
        i += 1
    return csr_matrix(STx), csr_matrix(STz)


def Gc(L):
    Gc_mat = np.zeros((2 * L, 2 * L), dtype="uint8")  # 2L * 2L matrix
    for k in range(L):
        Gc_mat[k][L + k] = 1
        Gc_mat[L + k][k] = 1
    return csr_matrix(Gc_mat)


def Renyi2_csr(dMR, Gcd_csr, ST_csr, Lxd, Lyd):
    dMR_csr = csr_matrix(dMR)
    Mgs = (dMR_csr.dot(Gcd_csr.dot(ST_csr))).toarray()

    R2_element = np.all((Mgs % 2) < 0.1, axis=0)
    #print(R2_element)
    R2sum = np.sum(R2_element.astype(int))
    return R2sum / ((Lxd - 1) * Lyd)


def calc_dep_and_R2(MR, Gc_csr, strings, L, Lx, Ly):
    STx_csr, STz_csr, STx_loop_csr, STz_loop_csr = strings

    rank_MR = rank_mod2_numba(MR.copy().astype(np.uint8, copy=False))

    # x-loop
    STx_loop_dense_T = STx_loop_csr.transpose().toarray()
    combined_x = np.vstack([MR, STx_loop_dense_T])
    rank_combined_x = rank_mod2_numba(combined_x.astype(np.uint8, copy=False))

    # z-loop
    STz_loop_dense_T = STz_loop_csr.transpose().toarray()
    combined_z = np.vstack([MR, STz_loop_dense_T])
    rank_combined_z = rank_mod2_numba(combined_z.astype(np.uint8, copy=False))

    dep_x = rank_combined_x - rank_MR
    dep_z = rank_combined_z - rank_MR

    R2x = Renyi2_csr(MR, Gc_csr, STx_csr, Lx, Ly)
    R2z = Renyi2_csr(MR, Gc_csr, STz_csr, Lx, Ly)
    R2x_loop = Renyi2_csr(MR, Gc_csr, STx_loop_csr, Lx, Ly)
    R2z_loop = Renyi2_csr(MR, Gc_csr, STz_loop_csr, Lx, Ly)

    R2x_loop = R2x_loop * (Lx - 1)
    R2z_loop = R2z_loop * (Lx - 1)

    return R2x, R2z, R2x_loop, R2z_loop, dep_x, dep_z


# ============================================================
# Logical checks
# ============================================================
def logical_in_MR(MR, op):
    op = op.reshape(1, -1).astype(MR.dtype, copy=False)

    r1 = rank_mod2_numba(MR.copy())
    r2 = rank_mod2_numba(np.vstack([MR, op]).copy())

    return int(r1 == r2)  # 1: MRに含まれる, 0: 含まれない


def anticommutes(row, op):
    L = len(row) // 2

    x1 = row[:L]
    z1 = row[L:]

    x2 = op[:L]
    z2 = op[L:]

    symp = (np.dot(x1, z2) + np.dot(z1, x2)) % 2
    return symp


def calc_logicals(MR, nsdd, logical_ops):
    active_MR = MR[:nsdd]

    alive_Xx = logical_in_MR(active_MR, logical_ops["Xx"])
    alive_Xy = logical_in_MR(active_MR, logical_ops["Xy"])
    alive_Zx = logical_in_MR(active_MR, logical_ops["Zx"])
    alive_Zy = logical_in_MR(active_MR, logical_ops["Zy"])

    # 元コードの観測量: nsdd-2行とlogical_Zyの反可換性
    logical_Xx_val = anticommutes(active_MR[nsdd - 2], logical_ops["Zy"])
    # anti_last1 = anticommutes(active_MR[nsdd-1], logical_ops["Zy"])

    return alive_Xx, alive_Xy, alive_Zx, alive_Zy, logical_Xx_val


# ============================================================
# Sampling / aggregation
# ============================================================
def simulate_one_sample(ids, params, geometry, regions, strings):
    Lx, Ly, Lv, L, p_indd, v_indd, logical_ops, Gc_csr = geometry

    p_list = np.linspace(params.ps, params.pl, params.Np)
    sample_results = {}

    # stabilizer matrix
    MR0 = initial_stabilizer_state(params.case, L, Lx, Ly, v_indd, p_indd)
    #MR0[L - 2] = logical_ops["Xx"]
    #MR0[L-1] = logical_ops["Zy"]
    nsdd0 = L

    # for name in ["Xx", "Xy", "Zx", "Zy"]:
    #     op = logical_ops[name]
    #     L_dummy = op.shape[0] // 2
    #     x = np.where(op[:L_dummy])[0]
    #     z = np.where(op[L_dummy:])[0]
    #     print(f"{name}:")
    #     print(f"  X={list(x)}")
    #     print(f"  Z={list(z)}")

    # alive_Xx = logical_in_MR(MR0, logical_ops["Xx"])
    # alive_Xy = logical_in_MR(MR0, logical_ops["Xy"])
    # alive_Zx = logical_in_MR(MR0, logical_ops["Zx"])
    # alive_Zy = logical_in_MR(MR0, logical_ops["Zy"])
    # val = f"alive_Xx={alive_Xx}, alive_Xy={alive_Xy}, alive_Zx={alive_Zx}, alive_Zy={alive_Zy}"
    # print(val)
    # print(np.where(MR0[L-2, :L])[0])

    # debug
    # L0 = MR0.shape[1] // 2
    # for i, row in enumerate(MR0):
    #     x = [int(v) for v in np.where(row[:L0])[0]]
    #     z = [int(v) for v in np.where(row[L0:])[0]]
    #     print(f"{i:2d}: X={x} Z={z}")
    # print(val)

    # sanity check at p=0
    TEN0, neg_parts0 = calc_TEN(MR0, nsdd0, L, regions)
    #print("NA=", neg_parts0[0], "NB=", neg_parts0[1], "NC=", neg_parts0[2],
    #      "NAB=", neg_parts0[3], "NBC=", neg_parts0[4], "NCA=", neg_parts0[5],
    #      "NABC=", neg_parts0[6], "TEN=", TEN0)

    MR = MR0.copy()
    nsdd = L

    ##repair

    u = np.random.random(L)
    #order = np.argsort(u)

    # ntrial=1 は random order と同じ
    order, largest_cluster = make_ep_order_triangular(Lx, Ly, ntrial=params.ntrial)

    u_sorted = u[order]

    #order, largest_cluster = make_ep_order_triangular(params.Lx, params.Ly, ntrial=params.ntrial)

    ptr = 0

    # for ip, p in enumerate(p_list):
    #     while ptr < L and u_sorted[ptr] < p:
    #         q = int(order[ptr])

    #         MR, nsdd = dephasing_linkZ(MR, q, nsdd)
    #         ptr += 1

    for ip, p in enumerate(p_list):
        n_target = int(np.floor(p * L))

        while ptr < n_target:
            q = int(order[ptr])  # EPで選ばれたtriangle link番号

            # nsdd = dephasing_linkZ_inplace(MR, q, nsdd)
            MR, nsdd = dephasing_linkZ(MR, q, nsdd)

            # print("ip=", ip, "p=", p, "ptr=", ptr, "q=", q, "nsdd=", nsdd)

            ptr += 1



            #print(MR)
            # L_dummy = MR.shape[1] // 2
            # for i, row in enumerate(MR):
            #     x = [int(v) for v in np.where(row[:L_dummy])[0]]
            #     z = [int(v) for v in np.where(row[L_dummy:])[0]]
            #     print(f"{i:2d}: X={x} Z={z}")


            # debug: dephasing直後にTENを見たいとき
            # TEN, neg_parts = calc_TEN(MR, nsdd, L, regions)
            # print(f"ptr={ptr:3d}, TEN={TEN:.3f}")
            # print("NA=", neg_parts[0], "NB=", neg_parts[1], "NC=", neg_parts[2],
            #       "NAB=", neg_parts[3], "NBC=", neg_parts[4], "NCA=", neg_parts[5],
            #       "NABC=", neg_parts[6], "TEN=", TEN)
            # input("Enter を押すと続行します... ptr=" + str(ptr))

            # active_MR = MR[:nsdd].copy()
            # L_dummy = active_MR.shape[1] // 2
            # print("active_MR")
            # for i, row in enumerate(active_MR):
            #     x = [int(v) for v in np.where(row[:L_dummy])[0]]
            #     z = [int(v) for v in np.where(row[L_dummy:])[0]]
            #     print(f"{i:2d}: X={x} Z={z}")

            # for i in range(max(0, nsdd-2), nsdd):
            #     row = active_MR[i]
            #     x = np.where(row[:L_dummy])[0]
            #     z = np.where(row[L_dummy:])[0]
            #     anti = anticommutes(row, logical_ops["Zy"])
            #     print(f"{i:2d}: X={list(x)} Z={list(z)} anti_Zy={anti}")
            # input("Enter を押すと続行します... " + "ptr=" + str(ptr) + " " + "q=" + str(q))

        active_MR = MR[:nsdd]
        TEN, neg_parts = calc_TEN(active_MR, nsdd, L, regions)

        largest = (
            largest_cluster[n_target - 1] if n_target > 0 else 1
        ) / Lv

        alive_Xx, alive_Xy, alive_Zx, alive_Zy, logical_Xx_val = calc_logicals(
            active_MR, nsdd, logical_ops
        )

        # for i in range(max(0, nsdd-2), nsdd):
        #     row = active_MR[i]
        #     L_dummy = active_MR.shape[1] // 2
        #     x = np.where(row[:L_dummy])[0]
        #     z = np.where(row[L_dummy:])[0]
        #     anti = anticommutes(row, logical_ops["Zy"])
        #     print(f"{i:2d}: X={list(x)} Z={list(z)} anti_Zy={anti}")

        # print(f"logical_Xx_val={logical_Xx_val}")

        # debug
        #print(f"ptr={ptr:3d}, TEN={TEN:.3f}")

        n_dephased = ptr
        #print(f"p={p:.3f}, dephased links={n_dephased}/{L}")

        # print("NA=", neg_parts[0], "NB=", neg_parts[1], "NC=", neg_parts[2],
        #       "NAB=", neg_parts[3], "NBC=", neg_parts[4], "NCA=", neg_parts[5],
        #       "NABC=", neg_parts[6], "TEN=", TEN,
        #       "logical_Xx_val=", logical_Xx_val, flush=True)
        # print(f"p={p:.3f}, TEN={TEN:.1f}, logical_Xx_val={logical_Xx_val:.1f}", flush=True)
        #input(f"Enter を押すと続行します... ")

        R2x, R2z, R2x_loop, R2z_loop, dep_x, dep_z = calc_dep_and_R2(
            active_MR, Gc_csr, strings, L, Lx, Ly
        )

        # print(
        #     f"rank={rank:>3} Lx=Ly={Lx:>2} p={p:>6.3f} ids={ids:>4} "
        #     f"TEN={TEN:>2} R2x={R2x:>6.3f} R2z={R2z:>6.3f} "
        #     f"R2x_loop={R2x_loop:>6.3f} R2z_loop={R2z_loop:>6.3f} "
        #     f"dep_x={dep_x:>6.3f} dep_z={dep_z:>6.3f} "
        #     f"logical_Xx_val={logical_Xx_val:>6.3f}",
        #     flush=True,
        # )

        print(
            f"rank={rank:>3} Lx=Ly={Lx:>2} p={p:>6.3f} ids={ids:>4} "
            f"TEN={TEN:>2} R2z={R2z:>6.3f} "
            f"logical_Xx_val={logical_Xx_val:>6.3f}, "
            f"largest={largest:>6.3f}",
            flush=True,
        )


        key = round(float(p), 6)
        sample_results.setdefault(key, []).append(
            (TEN, R2x, R2z, R2x_loop, R2z_loop, dep_x, dep_z, logical_Xx_val,largest)
        )

    return sample_results


def ave_err_var(x):
    x = np.asarray(x)
    ave = np.mean(x)
    var = np.var(x)
    err = np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return ave, err, var


def merge_local_results(all_results):
    merged_results = {}
    for res in all_results:
        for p, val_list in res.items():
            merged_results.setdefault(p, []).extend(val_list)
    return merged_results


def make_dataset(merged_results):
    keys = [
        "p",
        "TEN_ave", "TEN_err", "TEN_var",
        "R2x_ave", "R2x_err", "R2x_var",
        "R2z_ave", "R2z_err", "R2z_var",
        "R2x_loop_ave", "R2x_loop_err", "R2x_loop_var",
        "R2z_loop_ave", "R2z_loop_err", "R2z_loop_var",
        "dep_x_ave", "dep_x_err", "dep_x_var",
        "dep_z_ave", "dep_z_err", "dep_z_var",
        "logical_Xx_val_ave", "logical_Xx_val_err", "logical_Xx_val_var",
        "largest_ave", "largest_err", "largest_var",
    ]
    data_set = {key: [] for key in keys}

    for p in sorted(merged_results):
        vals = merged_results[p]

        TEN_vals = np.array([v[0] for v in vals])
        R2x_vals = np.array([v[1] for v in vals])
        R2z_vals = np.array([v[2] for v in vals])
        R2x_loop_vals = np.array([v[3] for v in vals])
        R2z_loop_vals = np.array([v[4] for v in vals])
        dep_x_vals = np.array([v[5] for v in vals])
        dep_z_vals = np.array([v[6] for v in vals])
        logical_Xx_val_vals = np.array([v[7] for v in vals])
        largest_vals = np.array([v[8] for v in vals])

        TEN_ave, TEN_err, TEN_var = ave_err_var(TEN_vals)
        R2x_ave, R2x_err, R2x_var = ave_err_var(R2x_vals)
        R2z_ave, R2z_err, R2z_var = ave_err_var(R2z_vals)
        R2x_loop_ave, R2x_loop_err, R2x_loop_var = ave_err_var(R2x_loop_vals)
        R2z_loop_ave, R2z_loop_err, R2z_loop_var = ave_err_var(R2z_loop_vals)
        dep_x_ave, dep_x_err, dep_x_var = ave_err_var(dep_x_vals)
        dep_z_ave, dep_z_err, dep_z_var = ave_err_var(dep_z_vals)
        logical_Xx_val_ave, logical_Xx_val_err, logical_Xx_val_var = ave_err_var(logical_Xx_val_vals)
        largest_ave, largest_err, largest_var = ave_err_var(largest_vals)

        print(
            f"p={p:.3f} : "
            f"TEN={TEN_ave:.4f}, "
            f"x-string={R2x_ave:.4f}, "
            f"z-string={R2z_ave:.4f}, "
            f"x-loop={R2x_loop_ave:.4f}, "
            f"z-loop={R2z_loop_ave:.4f}, "
            f"dep_x={dep_x_ave:.4f}, "
            f"dep_z={dep_z_ave:.4f}, "
            f"logical_Xx_val={logical_Xx_val_ave:.4f}, "
            f"largest={largest_ave:.4f}"
        )

        data_set["p"].append(p)
        for name, val in [
            ("TEN", (TEN_ave, TEN_err, TEN_var)),
            ("R2x", (R2x_ave, R2x_err, R2x_var)),
            ("R2z", (R2z_ave, R2z_err, R2z_var)),
            ("R2x_loop", (R2x_loop_ave, R2x_loop_err, R2x_loop_var)),
            ("R2z_loop", (R2z_loop_ave, R2z_loop_err, R2z_loop_var)),
            ("dep_x", (dep_x_ave, dep_x_err, dep_x_var)),
            ("dep_z", (dep_z_ave, dep_z_err, dep_z_var)),
            ("logical_Xx_val", (logical_Xx_val_ave, logical_Xx_val_err, logical_Xx_val_var)),
            ("largest", (largest_ave, largest_err, largest_var)),
        ]:
            data_set[f"{name}_ave"].append(val[0])
            data_set[f"{name}_err"].append(val[1])
            data_set[f"{name}_var"].append(val[2])

    return data_set


def save_dataset(data_set, params):
    output = f"Lx_{params.Lx}_Nd_{params.Nd}_Np_{params.Np}_ps_{params.ps}_pl_{params.pl}_ntrial_{params.ntrial}.npz"
    np.savez_compressed(
        output,
        **{key: np.array(value) for key, value in data_set.items()},
    )
    return output


def main():
    time_start = time.time()
    params = Params()

    Lx, Ly = params.Lx, params.Ly
    Lv = Lx * Ly       # total # of vertex
    L = 3 * Lv         # total # of link qubits
    Lp = 2 * Lv        # total # of plaquette

    p_indd, v_indd = create_transformation_pv_link(Lv, Lx, Ly)
    logical_ops = make_logical_ops(L, Lx, Ly, v_indd)
    regions = make_regions(L, Lx, Ly, p_indd, params.Ax, params.Ay)

    Gc_csr = Gc(L)
    STx_csr, STz_csr = Renyi2_create(L, Lv, Lx, Ly)
    STx_loop_csr, STz_loop_csr = Renyi2_loop_create(L, Lv, Lx, Ly)
    strings = (STx_csr, STz_csr, STx_loop_csr, STz_loop_csr)

    geometry = (Lx, Ly, Lv, L, p_indd, v_indd, logical_ops, Gc_csr)

    tasks = list(range(params.Nd))
    my_tasks = [tasks[i] for i in range(len(tasks)) if i % size == rank]

    local_results = {}
    for ids in my_tasks:
        sample_results = simulate_one_sample(ids, params, geometry, regions, strings)
        for p, val_list in sample_results.items():
            local_results.setdefault(p, []).extend(val_list)

    # MPI集約
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        merged_results = merge_local_results(all_results)
        data_set = make_dataset(merged_results)
        output = save_dataset(data_set, params)
        print(f"saved: {output}")

        time_end = time.time()
        elapsed = int(time_end - time_start)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"time = {h:02d}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    main()
