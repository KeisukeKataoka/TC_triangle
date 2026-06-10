# initial Z px=0 (px,pg)=(1,0) to (px,pg)=(0,1)

from __future__ import print_function, division
import sys,os
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import multi_dot
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from scipy.sparse.linalg import inv, eigs
from scipy.linalg import svdvals
from scipy.stats import unitary_group
from scipy import linalg
from numpy import linalg as LA
import pylab
import time
import numba
from numba import njit,c16,i8,u2
import sys
import math
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#seed = int(time.time()) + rank * 10000
#np.random.seed(seed)
global_seed = 123456  # 一貫性を持たせるため任意の定数
seed = global_seed + rank
np.random.seed(seed)

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
time_start = time.time()

def idx(x,y,Lx,Ly):
    return (x % Lx) + Lx * (y % Ly)
def inv_idx(i,Lx,Ly):
    return i % Lx, i // Lx


def dir_x(x,y,Lx,Lv):
    return x+Lx*y
def dir_y(x,y,Lx,Lv):
    return x+Lx*y+Lv
def dir_xy(x,y,Lx,Lv):
    return x+Lx*y+2*Lv


def create_transformation_pv_link(Lv, Lx, Ly):
    #############################################
    ############ plaquette part #################
    #############################################
    p_indd = np.zeros((2*Lx*Ly,3),dtype='uint32')
    for iy in range(Ly):
        for ix in range(Lx):
            #下三角
            ic=(ix%Lx)+Lx*(iy%Ly)
            ipx=(ix+1)%Lx
            ipy=(iy+1)%Ly
            p_indd[ic][0]=dir_x(ix,iy,Lx,Lv)
            p_indd[ic][1]=dir_y(ipx,iy,Lx,Lv)
            p_indd[ic][2]=dir_xy(ix,iy,Lx,Lv)
            #上三角
            p_indd[ic+Lx*Ly][0]=dir_xy(ix,iy,Lx,Lv)
            p_indd[ic+Lx*Ly][1]=dir_x(ix,ipy,Lx,Lv)
            p_indd[ic+Lx*Ly][2]=dir_y(ix,iy,Lx,Lv)

    ############################################
    ############ vertex part ###################
    ############################################
    v_indd = np.zeros((Lx*Ly,6),dtype='uint32')
    for iy in range(Ly):
        for ix in range(Lx):
            ic=(ix%Lx)+Lx*(iy%Ly)
            imx=(ix-1)%Lx
            imy=(iy-1)%Ly
            v_indd[ic][0]=dir_x(ix,iy,Lx,Lv)
            v_indd[ic][1]=dir_xy(ix,iy,Lx,Lv)
            v_indd[ic][2]=dir_y(ix,iy,Lx,Lv)
            v_indd[ic][3]=dir_x(imx,iy,Lx,Lv)
            v_indd[ic][4]=dir_xy(imx,imy,Lx,Lv)
            v_indd[ic][5]=dir_y(ix,imy,Lx,Lv)
    #print(v_indd)
    #print(p_indd)
    return p_indd, v_indd


####################################################
def initial_stabilizer_state(case,Ld,Lxd,Lyd,v_ind,p_ind):
    # Ld: total # of qubits
    MRi=np.zeros((Ld,2*Ld),dtype='uint32')
    if case == 1:
        #Case I: X product state
        for ii in range(Ld):
            MRi[ii][ii]=1 # X
    if case == 2:
        #Case I: Z product state
        for ii in range(Ld):
            MRi[ii][Ld+ii]=1 # Z
    if case ==3:
        ##Case III: unique g.s. of TC 
        ### plaquette stabilizers ZZZ ###
        for ip in range(2*Lxd*Lyd-1):
            MRi[ip][Ld+p_ind[ip][0]]=1 # Z0
            MRi[ip][Ld+p_ind[ip][1]]=1 # Z1
            MRi[ip][Ld+p_ind[ip][2]]=1 # Z2
            #独立ではない最後のBpだけ取らない
        ### vertex stabilizers  XXXXXX ###
        for iv in range((Lxd)*(Lyd)-1):
            ivs=iv+2*Lxd*Lyd-1  # note that iv and ivs are different.
            MRi[ivs][v_ind[iv][0]]=1 # X0
            MRi[ivs][v_ind[iv][1]]=1 # X1
            MRi[ivs][v_ind[iv][2]]=1 # X2
            MRi[ivs][v_ind[iv][3]]=1 # X3
            MRi[ivs][v_ind[iv][4]]=1 # X4
            MRi[ivs][v_ind[iv][5]]=1 # X5
            #独立ではない最後のAvだけ取らない
        ### Add logical operators###
            # x-direction loop logical Z
            for kk in range(Lxd):
                iv=(kk%Lx)+Lx*(0%Lyd)
                MRi[Ld-2][Ld+v_ind[iv][0]]=0 #Z
            # x-direction loop logical Z
            for kk in range(Lyd):
                iv=(0%Lx)+Lx*(kk%Lyd)
                MRi[Ld-1][Ld+v_ind[iv][2]]=0 #Z
    #print(MRi)
    return MRi

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

def dephasing_linkZ(MR, q):
    active = MR.copy()
    L2 = active.shape[1] // 2

    bad = np.flatnonzero(active[:, q] == 1)

    # print("q =", q)
    # print("bad =", bad)
    # for b in bad:
    #     print(
    #         "bad row", b,
    #         "X=", np.where(active[b, :L2])[0],
    #         "Z=", np.where(active[b, L2:])[0],
    #     )

    if bad.size == 0:
        return active

    k0 = int(bad[0])
    pivot = active[k0].copy()

    #print("delete pivot row =", k0)

    if bad.size > 1:
        active[bad[1:]] ^= pivot

    active = np.delete(active, k0, axis=0)

    return active

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

#     bad = np.flatnonzero(active[:, q] == 1)

#     if bad.size == 0:
#         return active

#     k0 = int(bad[0])
#     pivot = active[k0].copy()

#     if bad.size > 1:
#         active[bad[1:]] ^= pivot

#     active = np.delete(active, k0, axis=0)

#     return active


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
def subset2(Ld,Ax,Ay,Lx,Ly,p_indd):
    i = idx(Ax,Ay,Lx,Ly)
    Asub_set = set()
    for j in range(3):
        Asub_set.add(p_indd[i][j]) #三角形
        Asub_set.add(p_indd[i + Lx*Ly][j]) #逆三角形
        print(p_indd[i][j])
        print(p_indd[i + Lx*Ly][j])
    return Asub_set

def subset6(Ld,Ax,Ay,Lx,Ly,p_indd):
    Asub_set = set()
    Amx=(Ax - 1) % Lx
    Amy=(Ay - 1) % Ly
    i0 = idx(Ax,Ay,Lx,Ly)
    i1 = idx(Amx,Amy,Lx,Ly)
    i2 = idx(Amx,Ay,Lx,Ly)
    i3 = idx(Ax,Amy,Lx,Ly)
    for j in range(3):
        Asub_set.add(p_indd[i0][j])
        Asub_set.add(p_indd[i0 + Lx*Ly][j])
        Asub_set.add(p_indd[i1][j])
        Asub_set.add(p_indd[i1 + Lx*Ly][j])
        Asub_set.add(p_indd[i2][j])
        Asub_set.add(p_indd[i3+Lx*Ly][j])
    return Asub_set

# https://arxiv.org/abs/2204.08489
def EE_cal(MRd,Asub_set,Ld):
    # partition #
    #LA=Ld//2
    Asub=list(Asub_set)
    LA=len(Asub)
    MRdA=np.zeros((Ld,2*(len(Asub))),dtype='uint8')
    #print("new",Asub)
    for k2 in range(len(Asub)):
        MRdA[:,k2]=MRd[:,Asub[k2]].copy()
        MRdA[:,(LA)+k2]=MRd[:,(Ld)+Asub[k2]].copy()
    #print(MRdA.shape)
    #print(MRdA)
    rankA=rank_mod2_v3(MRdA)
    #print(rankA)
    EEd=rankA-(LA)
    return EEd


#####    Renyi2-correlator-csr  ######
def Renyi2_create(Ld,Lv,Lxd,Lyd):
    STx = np.zeros((2*Ld,(Lxd-1)*Lyd),dtype='uint8')
    STz = np.zeros((2*Ld,(Lxd-1)*Lyd),dtype='uint8')
    i=0
    for piy in range(Lyd):

        #zig-zag
        for leng in range(Lxd-1):
            leng = leng +1
            for ld in range(leng):
                ipd=ld+piy*Lxd
                ix,iy = inv_idx(ipd,Lxd,Lyd)
                di1=dir_y(ix,iy,Lxd,Lv)    
                di2=dir_xy(ix,iy,Lxd,Lv)
                #print(piy,leng,diz1+Ld,diz2+Ld)
                STx[di1,i]=1
                STx[di2,i]=1    
                STz[di1+Ld,i]=1
                STz[di2+Ld,i]=1            
            i+=1
    STx_list= csr_matrix(STx)
    STz_list= csr_matrix(STz)
    return STx_list, STz_list

def Renyi2_loop_create(Ld,Lv,Lxd,Lyd):
    STx = np.zeros((2*Ld,Lyd),dtype='uint8')
    STz = np.zeros((2*Ld,Lyd),dtype='uint8')
    i=0
    for piy in range(Lyd):
        #zig-zag
        for ld in range(Lxd):
            ipd=ld+piy*Lxd
            ix,iy = inv_idx(ipd,Lxd,Lyd)
            di1=dir_y(ix,iy,Lxd,Lv)    
            di2=dir_xy(ix,iy,Lxd,Lv)
            #print(piy,leng,diz1+Ld,diz2+Ld)
            STx[di1,i]=1
            STx[di2,i]=1    
            STz[di1+Ld,i]=1
            STz[di2+Ld,i]=1            
        i+=1
    STx_list= csr_matrix(STx)
    STz_list= csr_matrix(STz)
    return STx_list, STz_list

def Gc(L):
    Gc=np.zeros((2*L,2*L),dtype='uint8') # 2L * 2L matrix
    for k in range(L):
        Gc[k][L+k]=1
        Gc[L+k][k]=1  
    Gc_csr=csr_matrix(Gc)
    return Gc_csr

def Renyi2_csr(dMR,Gcd_csr,ST_csr,Lxd,Lyd):
    dMR_csr = csr_matrix(dMR)
    Mgs=(dMR_csr.dot(Gcd_csr.dot(ST_csr))).toarray()

    R2_element=np.all((Mgs%2)<0.1,axis=0)
    #print(R2_element)
    R2sum=np.sum(R2_element.astype(int))#       
    return R2sum/((Lxd-1)*Lyd)



##### Main simulation loop #####


# parameters
ps, pl = 0.0, 1.0
Np=21
Nd = 100 #800 sample number

#case1= X,case2= Z, case3= ground state of TC
case=3

Lx, Ly = 12,12

Lv=Lx*Ly # total # of vertex
L=3*Lv # total # of link qubits
Lp=2*Lv  # total # of plaquette


data_set = {}

tasks = list(range(Nd))

my_tasks = [tasks[i] for i in range(len(tasks)) if i % size == rank]

# --- 3. 計算結果をためる辞書 ---
local_results = {}  # key = (Lx, pg), value = [TEE値のリスト]

fig, ax = plt.subplots(1,2,figsize=(12,4))

p_indd, v_indd=create_transformation_pv_link(Lv, Lx, Ly)

#Ax,Ay=Lx//2,Ly//2
Ax,Ay = 2,2
Bx,By=Ax+2,Ay+1
Cx,Cy=Ax+1,Ay+2     

A = subset6(L,Ax,Ay,Lx,Ly,p_indd)
B = subset6(L,Bx,By,Lx,Ly,p_indd)
C = subset6(L,Cx,Cy,Lx,Ly,p_indd)


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

A_idx = np.array(sorted(A), dtype=np.int64)
B_idx = np.array(sorted(B), dtype=np.int64)
C_idx = np.array(sorted(C), dtype=np.int64)
AB_idx = np.array(sorted(AB), dtype=np.int64)
BC_idx = np.array(sorted(BC), dtype=np.int64)
CA_idx = np.array(sorted(CA), dtype=np.int64)
ABC_idx = np.array(sorted(ABC), dtype=np.int64)



for ids in my_tasks:
    
    
    
    Gc_csr=Gc(L)
    Gc_array = Gc_csr.toarray()
    #print("Gc",Gc.shape)
    
    
    p_list = np.linspace(ps, pl, Np)
    TEN_list=[]
    
    R2x_list=[]
    R2z_list=[]
    
    R2x_loop_list=[]
    R2z_loop_list=[]
    
    dep_x_list=[]
    dep_z_list=[]

    TEN_ave=[]
    TEN_error=[]
    TEN_var=[]

    R2x_ave=[]
    R2x_error=[]
    R2x_var=[]

    R2z_ave=[]
    R2z_error=[]
    R2z_var=[]

    R2x_loop_ave=[]
    R2x_loop_error=[]
    R2x_loop_var=[]

    R2z_loop_ave=[]
    R2z_loop_error=[]
    R2z_loop_var=[]

    dep_x_ave=[]
    dep_x_error=[]
    dep_x_var=[]

    dep_z_ave=[]
    dep_z_error=[]
    dep_z_var=[]

    # stabilizer matrix
    MR=np.zeros((L,2*L),dtype='uint8') # L * 2L matrix

    # sample set of physical quantity
    MR0=initial_stabilizer_state(case,L,Lx,Ly,v_indd,p_indd)
    nsdd0 = L


    # debug
    # L0 = MR0.shape[1] // 2
    # for i, row in enumerate(MR0):
    #     x = [int(v) for v in np.where(row[:L0])[0]]
    #     z = [int(v) for v in np.where(row[L0:])[0]]
    #     print(f"{i:2d}: X={x} Z={z}")

    




    # sanity check at p=0
    NA0 = negativity_E_fast(MR0, nsdd0, Lv, A_idx)
    NB0 = negativity_E_fast(MR0, nsdd0, Lv, B_idx)
    NC0 = negativity_E_fast(MR0, nsdd0, Lv, C_idx)
    NAB0 = negativity_E_fast(MR0, nsdd0, Lv, AB_idx)
    NBC0 = negativity_E_fast(MR0, nsdd0, Lv, BC_idx)
    NCA0 = negativity_E_fast(MR0, nsdd0, Lv, CA_idx)
    NABC0 = negativity_E_fast(MR0, nsdd0, Lv, ABC_idx)
    TEN0 = NA0 + NB0 + NC0 - NAB0 - NBC0 - NCA0 + NABC0

    #print("NA=", NA0, "NB=", NB0, "NC=", NC0, "NAB=", NAB0, "NBC=", NBC0, "NCA=", NCA0, "NABC=", NABC0, "TEN=", TEN0)

    MR = MR0.copy()
    nsdd = L


    u = np.random.random(L)
    order = np.argsort(u)
    u_sorted = u[order]
    ptr = 0

    for ip, p in enumerate(p_list):
        while ptr < L and u_sorted[ptr] < p:
            q = int(order[ptr])   # triangle のリンク番号
            #nsdd = dephasing_linkZ_inplace(MR, q, nsdd)

            MR = dephasing_linkZ(MR, q)
            #print("ip=",ip,"p=",p,"q=",q,"nsdd=",nsdd)

            #print(MR)
            #L_dummy = MR.shape[1] // 2
            # for i, row in enumerate(MR):
            #     x = [int(v) for v in np.where(row[:L_dummy])[0]]
            #     z = [int(v) for v in np.where(row[L_dummy:])[0]]
            #     print(f"{i:2d}: X={x} Z={z}")

            ptr += 1

            active_MR = MR.copy()
            L_dummy = active_MR.shape[1] // 2

            NA = negativity_E_fast(active_MR, nsdd0, Lv, A_idx)
            NB = negativity_E_fast(active_MR, nsdd0, Lv, B_idx)
            NC = negativity_E_fast(active_MR, nsdd0, Lv, C_idx)
            NAB = negativity_E_fast(active_MR, nsdd0, Lv, AB_idx)
            NBC = negativity_E_fast(active_MR, nsdd0, Lv, BC_idx)
            NCA = negativity_E_fast(active_MR, nsdd0, Lv, CA_idx)
            NABC = negativity_E_fast(active_MR, nsdd0, Lv, ABC_idx)
            TEN = NA + NB + NC - NAB - NBC - NCA + NABC

            # debug
            # print(f"ptr={ptr:3d}, TEN={TEN:.3f}")
            # print("NA=", NA, "NB=", NB, "NC=", NC, "NAB=", NAB, "NBC=", NBC, "NCA=", NCA, "NABC=", NABC, "TEN=", TEN)
            # input("Enter を押すと続行します... ptr=" + str(ptr))

        active_MR = MR.copy()
        #active_MR = MR[:nsdd].copy()
        L_dummy = active_MR.shape[1] // 2
        # print("active_MR")
        # for i, row in enumerate(active_MR):
        #     x = [int(v) for v in np.where(row[:L_dummy])[0]]
        #     z = [int(v) for v in np.where(row[L_dummy:])[0]]
        #     print(f"{i:2d}: X={x} Z={z}")

        NA = negativity_E_fast(active_MR, nsdd0, Lv, A_idx)
        NB = negativity_E_fast(active_MR, nsdd0, Lv, B_idx)
        NC = negativity_E_fast(active_MR, nsdd0, Lv, C_idx)
        NAB = negativity_E_fast(active_MR, nsdd0, Lv, AB_idx)
        NBC = negativity_E_fast(active_MR, nsdd0, Lv, BC_idx)
        NCA = negativity_E_fast(active_MR, nsdd0, Lv, CA_idx)
        NABC = negativity_E_fast(active_MR, nsdd0, Lv, ABC_idx)
        TEN = NA + NB + NC - NAB - NBC - NCA + NABC

        # debug
        # print(f"ptr={ptr:3d}, TEN={TEN:.3f}")

        n_dephased = ptr
        # print(f"p={p:.3f}, dephased links={n_dephased}/{L}")

        #print("NA=", NA, "NB=", NB, "NC=", NC, "NAB=", NAB, "NBC=", NBC, "NCA=", NCA, "NABC=", NABC, "TEN=", TEN)

        STx_csr,STz_csr=Renyi2_create(L,Lv,Lx,Ly)
        STx_loop_csr,STz_loop_csr=Renyi2_loop_create(L,Lv,Lx,Ly)
        #ST_dense = ST_list.toarray()
        #print(ST_dense.shape)
        #print(ST_dense)

        rank_MR = rank_mod2_numba(MR.copy().astype(np.uint8, copy=False))
        
        ### x-loop ###
        STx_loop_csr_T = STx_loop_csr.transpose()
        STx_loop_dense_T = STx_loop_csr_T.toarray()
        
        rank_STx_loop = rank_mod2_numba(STx_loop_dense_T.astype(np.uint8, copy=False))   
        
        combined_x = np.vstack([MR, STx_loop_dense_T])
        rank_combined_x = rank_mod2_numba(combined_x.astype(np.uint8, copy=False))

        ### z-loop ###
        STz_loop_csr_T = STz_loop_csr.transpose()
        STz_loop_dense_T = STz_loop_csr_T.toarray()
        
        rank_STz_loop = rank_mod2_numba(STz_loop_dense_T.astype(np.uint8, copy=False))      
        
        combined_z = np.vstack([MR, STz_loop_dense_T])
        rank_combined_z = rank_mod2_numba(combined_z.astype(np.uint8, copy=False))
    
        dep_x=rank_combined_x-rank_MR
        dep_z=rank_combined_z-rank_MR        
                
        R2x=Renyi2_csr(MR,Gc_csr,STx_csr,Lx,Ly)
        R2z=Renyi2_csr(MR,Gc_csr,STz_csr,Lx,Ly)
        R2x_loop=Renyi2_csr(MR,Gc_csr,STx_loop_csr,Lx,Ly)
        R2z_loop=Renyi2_csr(MR,Gc_csr,STz_loop_csr,Lx,Ly)

        R2x_loop = R2x_loop*(Lx-1)
        R2z_loop = R2z_loop*(Lx-1)
        #print("pg=",pg,"ids=",ids,"TEN=",TEN0,"Renyi2=",Renyi2_corr_csr)
        TEN_list=np.append(TEN_list,TEN)
        R2x_list=np.append(R2x_list,R2x)
        R2z_list=np.append(R2z_list,R2z)
        R2x_loop_list=np.append(R2x_loop_list,R2x_loop)
        R2z_loop_list=np.append(R2z_loop_list,R2z_loop)
        dep_x_list=np.append(dep_x_list,dep_x)
        dep_z_list=np.append(dep_z_list,dep_z)
        print(f"rank={rank:>3} Lx=Ly={Lx:>2} p={p:>6.3f} ids={ids:>4} TEN={TEN:>2} R2x={R2x:>6.3f} R2z={R2z:>6.3f} R2x_loop={R2x_loop:>6.3f} R2z_loop={R2z_loop:>6.3f}  dep_x={dep_x:>6.3f} dep_z={dep_z:>6.3f}", flush=True)

    
        if len(TEN_list) > 1:
            TEN_aved = np.mean(TEN_list)
            TEN_vard = np.var(TEN_list)
            TEN_errord = np.std(TEN_list, ddof=1) / np.sqrt(len(TEN_list))

            R2x_aved = np.mean(R2x_list)
            R2x_vard = np.var(R2x_list)
            R2x_errord = np.std(R2x_list, ddof=1) / np.sqrt(len(R2x_list))

            R2z_aved = np.mean(R2z_list)
            R2z_vard = np.var(R2z_list)
            R2z_errord = np.std(R2z_list, ddof=1) / np.sqrt(len(R2z_list))

            R2x_loop_aved = np.mean(R2x_loop_list)
            R2x_loop_vard = np.var(R2x_loop_list)
            R2x_loop_errord = np.std(R2x_loop_list, ddof=1) / np.sqrt(len(R2x_loop_list))

            R2z_loop_aved = np.mean(R2z_loop_list)
            R2z_loop_vard = np.var(R2z_loop_list)
            R2z_loop_errord = np.std(R2z_loop_list, ddof=1) / np.sqrt(len(R2z_loop_list))
            
        elif len(TEN_list) == 1:
            TEN_aved = TEN_list[0]
            TEN_vard = 0.0
            TEN_errord = 0.0

            R2x_aved = R2x_list[0]
            R2x_vard = 0.0
            R2x_errord = 0.0

            R2z_aved = R2z_list[0]
            R2z_vard = 0.0
            R2z_errord = 0.0
            
            R2x_loop_aved = R2x_loop_list[0]
            R2x_loop_vard = 0.0
            R2x_loop_errord = 0.0

            R2z_loop_aved = R2z_loop_list[0]
            R2z_loop_vard = 0.0
            R2z_loop_errord = 0.0
            
        else:
            TEN_aved = np.nan
            TEN_vard = np.nan
            TEN_errord = np.nan
            
            R2x_aved = np.nan
            R2x_vard = np.nan
            R2x_errord = np.nan

            R2z_aved = np.nan
            R2z_vard = np.nan
            R2z_errord = np.nan
            
            R2x_loop_aved = np.nan
            R2x_loop_vard = np.nan
            R2x_loop_errord = np.nan

            R2z_loop_aved = np.nan
            R2z_loop_vard = np.nan
            R2z_loop_errord = np.nan 
                    
        key = round(float(p), 6)

        if key not in local_results:
            local_results[key] = []

        local_results[key].append(
            (TEN, R2x, R2z, R2x_loop, R2z_loop, dep_x, dep_z)
        )

# --- 4. MPI集約 ---
all_results = comm.gather(local_results, root=0)

if rank == 0:
    merged_results = {}

    for res in all_results:
        for p, val_list in res.items():
            merged_results.setdefault(p, []).extend(val_list)

    data_set = {
        "p": [],
        "TEN_ave": [],
        "TEN_err": [],
        "TEN_var": [],
        "R2x_ave": [],
        "R2x_err": [],
        "R2x_var": [],
        "R2z_ave": [],
        "R2z_err": [],
        "R2z_var": [],
        "R2x_loop_ave": [],
        "R2x_loop_err": [],
        "R2x_loop_var": [],
        "R2z_loop_ave": [],
        "R2z_loop_err": [],
        "R2z_loop_var": [],
        "dep_x_ave": [],
        "dep_x_err": [],
        "dep_x_var": [],
        "dep_z_ave": [],
        "dep_z_err": [],
        "dep_z_var": [],
    }

    for p in sorted(merged_results):
        vals = merged_results[p]

        TEN_vals = np.array([v[0] for v in vals])
        R2x_vals = np.array([v[1] for v in vals])
        R2z_vals = np.array([v[2] for v in vals])
        R2x_loop_vals = np.array([v[3] for v in vals])
        R2z_loop_vals = np.array([v[4] for v in vals])
        dep_x_vals = np.array([v[5] for v in vals])
        dep_z_vals = np.array([v[6] for v in vals])

        def ave_err_var(x):
            ave = np.mean(x)
            var = np.var(x)
            err = np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
            return ave, err, var

        TEN_ave, TEN_err, TEN_var = ave_err_var(TEN_vals)
        R2x_ave, R2x_err, R2x_var = ave_err_var(R2x_vals)
        R2z_ave, R2z_err, R2z_var = ave_err_var(R2z_vals)
        R2x_loop_ave, R2x_loop_err, R2x_loop_var = ave_err_var(R2x_loop_vals)
        R2z_loop_ave, R2z_loop_err, R2z_loop_var = ave_err_var(R2z_loop_vals)
        dep_x_ave, dep_x_err, dep_x_var = ave_err_var(dep_x_vals)
        dep_z_ave, dep_z_err, dep_z_var = ave_err_var(dep_z_vals)

        print(
            f"p={p:.3f} : "
            f"TEN={TEN_ave:.4f}, "
            f"x-string={R2x_ave:.4f}, "
            f"z-string={R2z_ave:.4f}, "
            f"x-loop={R2x_loop_ave:.4f}, "
            f"z-loop={R2z_loop_ave:.4f}, "
            f"dep_x={dep_x_ave:.4f}, "
            f"dep_z={dep_z_ave:.4f}"
        )

        data_set["p"].append(p)
        data_set["TEN_ave"].append(TEN_ave)
        data_set["TEN_err"].append(TEN_err)
        data_set["TEN_var"].append(TEN_var)

        data_set["R2x_ave"].append(R2x_ave)
        data_set["R2x_err"].append(R2x_err)
        data_set["R2x_var"].append(R2x_var)

        data_set["R2z_ave"].append(R2z_ave)
        data_set["R2z_err"].append(R2z_err)
        data_set["R2z_var"].append(R2z_var)

        data_set["R2x_loop_ave"].append(R2x_loop_ave)
        data_set["R2x_loop_err"].append(R2x_loop_err)
        data_set["R2x_loop_var"].append(R2x_loop_var)

        data_set["R2z_loop_ave"].append(R2z_loop_ave)
        data_set["R2z_loop_err"].append(R2z_loop_err)
        data_set["R2z_loop_var"].append(R2z_loop_var)

        data_set["dep_x_ave"].append(dep_x_ave)
        data_set["dep_x_err"].append(dep_x_err)
        data_set["dep_x_var"].append(dep_x_var)

        data_set["dep_z_ave"].append(dep_z_ave)
        data_set["dep_z_err"].append(dep_z_err)
        data_set["dep_z_var"].append(dep_z_var)
    
    np.savez_compressed(
    f"Lx_{Lx}_Nd_{Nd}_Np_{Np}_ps_{ps}_pl_{pl}.npz",
    p=np.array(data_set["p"]),
    TEN_ave=np.array(data_set["TEN_ave"]),
    TEN_err=np.array(data_set["TEN_err"]),
    TEN_var=np.array(data_set["TEN_var"]),
    R2x_ave=np.array(data_set["R2x_ave"]),
    R2x_err=np.array(data_set["R2x_err"]),
    R2x_var=np.array(data_set["R2x_var"]),
    R2z_ave=np.array(data_set["R2z_ave"]),
    R2z_err=np.array(data_set["R2z_err"]),
    R2z_var=np.array(data_set["R2z_var"]),
    R2x_loop_ave=np.array(data_set["R2x_loop_ave"]),
    R2x_loop_err=np.array(data_set["R2x_loop_err"]),
    R2x_loop_var=np.array(data_set["R2x_loop_var"]),
    R2z_loop_ave=np.array(data_set["R2z_loop_ave"]),
    R2z_loop_err=np.array(data_set["R2z_loop_err"]),
    R2z_loop_var=np.array(data_set["R2z_loop_var"]),
    dep_x_ave=np.array(data_set["dep_x_ave"]),
    dep_x_err=np.array(data_set["dep_x_err"]),
    dep_x_var=np.array(data_set["dep_x_var"]),
    dep_z_ave=np.array(data_set["dep_z_ave"]),
    dep_z_err=np.array(data_set["dep_z_err"]),
    dep_z_var=np.array(data_set["dep_z_var"]),
)
    
    # 必要ならグラフ描画もここで可能
    time_end = time.time()

    elapsed = int(time_end - time_start)

    h = elapsed // 3600
    m = (elapsed % 3600) // 60
    s = elapsed % 60

    print(f"time = {h:02d}:{m:02d}:{s:02d}")