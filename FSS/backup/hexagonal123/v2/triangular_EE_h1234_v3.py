# initial state produxt X pz+pg=1-px
# h1234

from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and ca be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
from quspin.operators import hamiltonian # Hamiltonians and operators
#from quspin.basis import spinless_fermion_basis_1d # Hilbert space fermion basis
from quspin.basis import spin_basis_1d # Hilbert space spin basis
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
import copy
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
            for kk in range(Lx):
                iv=(kk%Lx)+Lx*(0%Ly)
                MRi[Ld-2][Ld+v_ind[iv][0]]=0 #Z
            # x-direction loop logical Z
            for kk in range(Ly):
                iv=(0%Lx)+Lx*(kk%Ly)
                MRi[Ld-1][Ld+v_ind[iv][1]]=0 #Z
    #print(MRi)
    return MRi

####################################################
def measurement_op(dMR,Ld,Lxd,Lyd,v_ind,p_ind,Gcd,mtyped):
    Meo=np.zeros(2*Ld,dtype='uint32')
    #
    if mtyped==1: #plaquette or star operator
        todo=[0,1]
        prob_list = [0.5,0.5]
        vorp = np.random.choice(todo,size=None,replace=True, p=prob_list)
        if vorp==0:
            todo=np.arange(Lxd*Lyd)
            vp=np.random.choice(todo,size=None,replace=True,p=None)
            #please check
            Meo[v_ind[vp][0]]=1 #X0
            Meo[v_ind[vp][1]]=1 #X1
            Meo[v_ind[vp][2]]=1 #X2
            Meo[v_ind[vp][3]]=1 #X3
            Meo[v_ind[vp][4]]=1 #X4
            Meo[v_ind[vp][5]]=1 #X5
        else:
            todo=np.arange(2*Lxd*Lyd)
            pp=np.random.choice(todo,size=None,replace=True,p=None)
            #please check
            Meo[Ld+p_ind[pp][0]]=1 #Z0
            Meo[Ld+p_ind[pp][1]]=1 #Z1
            Meo[Ld+p_ind[pp][2]]=1 #Z2
            
    if mtyped==2: #local X
        todo=np.arange(Ld)
        mp=np.random.choice(todo,size=None,replace=True,p=None)
        #please check
        Meo[mp]=1 #X0
        
    if mtyped==3: #local Z
        todo=np.arange(Ld)
        mp=np.random.choice(todo,size=None,replace=True,p=None)
        #please check
        Meo[Ld+mp]=1 #X0
    
    # Meo without outcome sign
    MeoF=np.zeros(2*Ld,dtype='uint32')
    MeoF=Meo
    
    # dMR without foctor
    dMRF=np.zeros((Ld,2*Ld),dtype='uint32')
    dMRF=dMR
    #check anti commtation
    Mgs=np.dot(dMRF,np.dot(Gcd,MeoF.T))
    Mgs=Mgs%2  ## check: reduce Z2 value 
    
    antic_index_list=[]
    aid=np.where(Mgs[:]!=0)
    antic_index_list=aid[0]
    #######
    lenMe=len(antic_index_list)
    if lenMe !=0:
        kc=antic_index_list[0]
        dMR_prev=dMR[kc].copy()
        #replace stabilizer for kc
        dMR[kc]=Meo
        update_list=np.delete(antic_index_list,0)
        dMR[update_list,:]=np.mod(dMR[update_list,:]+dMR_prev,2)
    return dMR

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
    MRdA=np.zeros((Ld,2*(len(Asub))),dtype='uint32')
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

#############################
def rank_mod2_v3(MRdd):
    # MRdA construct
    i = 0
    Ic = MRdd.shape[0]
    Jc = MRdd.shape[1]
    for j in range(Jc):
        check_ind=np.where(MRdd[i:,j]==1)
        if len(check_ind[0]) !=0:
            if len(check_ind[0]) >1:
                elim_ind=np.delete(check_ind,0)
                MRdd[elim_ind+i,:]=np.mod(MRdd[elim_ind+i,:]+MRdd[check_ind[0][0]+i,:],2)
            MRdd[i,:],MRdd[check_ind[0][0]+i,:] = MRdd[check_ind[0][0]+i,:],MRdd[i,:].copy()
            i=i+1
    return i
############################

#case1= X,case2= Z, case3= ground state of TC
case=2


print("test")
data_set = {}

Np=11
Nd=100 #800 sample number

# --- 1. 計算タスクの全リスト作成 ---
#L_list=[10,12,14,16]
L_list=[10]
tasks = []
for tmp_i, Lx in enumerate(L_list):
    Ly = Lx
    for ids in range(Nd):
        tasks.append((tmp_i, ids))

# --- 2. MPI rankに均等割り当て ---
my_tasks = [tasks[i] for i in range(len(tasks)) if i % size == rank]

# --- 3. 計算結果をためる辞書 ---
local_results = {}  # key = (Lx, pg), value = [TEE値のリスト]



for (tmp_i, ids) in my_tasks:
    Lx=L_list[tmp_i]
    Ly=L_list[tmp_i]

    Lv=Lx*Ly # total # of vertex
    L=3*Lv # total # of link qubits
    Lp=2*Lv  # total # of plaquette
    NT=4*L #4*Lでなくて6*L?? total time step
    
    p_indd, v_indd=create_transformation_pv_link(Lv, Lx, Ly)

    # stabilizer matrix
    MR0=np.zeros((L,2*L),dtype='uint32') # L * 2L matrix
    # sample set of physical quantity
    MR=initial_stabilizer_state(case,L,Lx,Ly,v_indd,p_indd)
    
    p_list=[]

    #ベンゼン
    TEE1_ave=[]
    TEE1_error=[]
    TEE1_var=[]

    #ナフタレン
    TEE2_ave=[]
    TEE2_error=[]
    TEE2_var=[]

    #アントラセン
    TEE3_ave=[]
    TEE3_error=[]
    TEE3_var=[]

    #アントラセン
    TEE4_ave=[]
    TEE4_error=[]
    TEE4_var=[]
    
    #for ip in range(len(p_list)):
    ps=0.7 # ps < pg < pl
    pl=0.9
    px=0.0
    for ip in range(Np):  # pg を順に変えながらMR引き継ぎ    
        #pg=0.5*(1-px)*ip/(Np-1)+0.5
        pg=ps+(pl-px-ps)*ip/(Np-1)
        pz=1-px-pg
        
        p_list=np.append(p_list,pg)
    
    
        ### check matrix ###
        Gc=np.zeros((2*L,2*L),dtype='uint32') # 2L * 2L matrix
        for k in range(L):
            Gc[k][L+k]=1
            Gc[L+k][k]=1  
        Gc_csr=csr_matrix(Gc)
        
        TEE1_list=[]
        TEE2_list=[]
        TEE3_list=[]
        TEE4_list=[]
        #MR=MR0 #-0.85
        #MR = np.copy(MR0) #-0.86
        #MR =copy.deepcopy(MR0) #-0.85
        #measurement dynamics
    
        prob_list=[pg,px,pz]
        todo=[1,2,3]
        for it in range(NT):
            mtype= np.random.choice(todo,size=None,replace=True,p=prob_list)
            MR=measurement_op(MR,L,Lx,Ly,v_indd,p_indd,Gc,mtype)
            if it==NT-1:
                TEE1_sum = 0.0
                TEE2_sum = 0.0
                TEE3_sum = 0.0
                TEE4_sum = 0.0
                count = 0
                
                for Ax in range(Lx):
                    for Ay in range(Ly):
                        Bx, By = (Ax - 2) % Lx, (Ay - 1) % Ly
                        Cx, Cy = (Ax - 1) % Lx, (Ay - 2) % Ly
                
                        # ベンゼン型
                        A1 = subset6(L, Ax, Ay, Lx, Ly, p_indd)
                        B1 = subset6(L, Bx, By, Lx, Ly, p_indd) - A1
                        C1 = subset6(L, Cx, Cy, Lx, Ly, p_indd) - A1 - B1
                
                        AB1 = A1 | B1
                        BC1 = B1 | C1
                        CA1 = C1 | A1
                        ABC1 = AB1 | C1
                
                        SA1 = EE_cal(MR, A1, L)
                        SB1 = EE_cal(MR, B1, L)
                        SC1 = EE_cal(MR, C1, L)
                        SAB1 = EE_cal(MR, AB1, L)
                        SBC1 = EE_cal(MR, BC1, L)
                        SCA1 = EE_cal(MR, CA1, L)
                        SABC1 = EE_cal(MR, ABC1, L)
                
                        TEE1 = SA1 + SB1 + SC1 - SAB1 - SBC1 - SCA1 + SABC1
                
                        # ナフタレン型
                        A2 = subset6(L, Ax, Ay, Lx, Ly, p_indd) | subset6(L, (Ax - 1) % Lx, (Ay + 1) % Ly, Lx, Ly, p_indd)
                        B2 = subset6(L, Bx, By, Lx, Ly, p_indd) | subset6(L, (Bx - 1) % Lx, (By - 2) % Ly, Lx, Ly, p_indd) - A2
                        C2 = subset6(L, Cx, Cy, Lx, Ly, p_indd) | subset6(L, (Cx + 2) % Lx, (Cy + 1) % Ly, Lx, Ly, p_indd) - A2 - B2
                
                        AB2 = A2 | B2
                        BC2 = B2 | C2
                        CA2 = C2 | A2
                        ABC2 = AB2 | C2
                
                        SA2 = EE_cal(MR, A2, L)
                        SB2 = EE_cal(MR, B2, L)
                        SC2 = EE_cal(MR, C2, L)
                        SAB2 = EE_cal(MR, AB2, L)
                        SBC2 = EE_cal(MR, BC2, L)
                        SCA2 = EE_cal(MR, CA2, L)
                        SABC2 = EE_cal(MR, ABC2, L)
                
                        TEE2 = SA2 + SB2 + SC2 - SAB2 - SBC2 - SCA2 + SABC2
                
                        # アントラセン型
                        A3 = (subset6(L, Ax, Ay, Lx, Ly, p_indd) |
                              subset6(L, (Ax - 1) % Lx, (Ay + 1) % Ly, Lx, Ly, p_indd) |
                              subset6(L, (Ax - 3) % Lx, Ay, Lx, Ly, p_indd))
                        B3 = (subset6(L, Bx, By, Lx, Ly, p_indd) |
                              subset6(L, (Bx - 1) % Lx, (By - 2) % Ly, Lx, Ly, p_indd) |
                              subset6(L, Bx, (By - 3) % Ly, Lx, Ly, p_indd)) - A3
                        C3 = (subset6(L, Cx, Cy, Lx, Ly, p_indd) |
                              subset6(L, (Cx + 2) % Lx, (Cy + 1) % Ly, Lx, Ly, p_indd) |
                              subset6(L, (Cx + 3) % Lx, (Cy + 1) % Ly, Lx, Ly, p_indd)) - A3 - B3
                
                        AB3 = A3 | B3
                        BC3 = B3 | C3
                        CA3 = C3 | A3
                        ABC3 = AB3 | C3
                
                        SA3 = EE_cal(MR, A3, L)
                        SB3 = EE_cal(MR, B3, L)
                        SC3 = EE_cal(MR, C3, L)
                        SAB3 = EE_cal(MR, AB3, L)
                        SBC3 = EE_cal(MR, BC3, L)
                        SCA3 = EE_cal(MR, CA3, L)
                        SABC3 = EE_cal(MR, ABC3, L)
                
                        TEE3 = SA3 + SB3 + SC3 - SAB3 - SBC3 - SCA3 + SABC3
                
                        # アントラセン+ベンゼン型
                        A4 = (subset6(L, Ax, Ay, Lx, Ly, p_indd) |
                              subset6(L, (Ax - 1) % Lx, (Ay + 1) % Ly, Lx, Ly, p_indd) |
                              subset6(L, (Ax - 3) % Lx, Ay, Lx, Ly, p_indd) |
                              subset6(L, (Ax - 4) % Lx, (Ay - 2) % Ly, Lx, Ly, p_indd))
                        B4 = (subset6(L, Bx, By, Lx, Ly, p_indd) |
                              subset6(L, (Bx - 1) % Lx, (By - 2) % Ly, Lx, Ly, p_indd) |
                              subset6(L, Bx, (By - 3) % Ly, Lx, Ly, p_indd) |
                              subset6(L, (Bx + 2) % Lx, (By - 2) % Ly, Lx, Ly, p_indd)) - A4
                        C4 = (subset6(L, Cx, Cy, Lx, Ly, p_indd) |
                              subset6(L, (Cx + 2) % Lx, (Cy + 1) % Ly, Lx, Ly, p_indd) |
                              subset6(L, (Cx + 3) % Lx, (Cy + 3) % Ly, Lx, Ly, p_indd) |
                              subset6(L, (Cx + 2) % Lx, (Cy + 4) % Ly, Lx, Ly, p_indd)) - A4 - B4
                
                        AB4 = A4 | B4
                        BC4 = B4 | C4
                        CA4 = C4 | A4
                        ABC4 = AB4 | C4
                
                        SA4 = EE_cal(MR, A4, L)
                        SB4 = EE_cal(MR, B4, L)
                        SC4 = EE_cal(MR, C4, L)
                        SAB4 = EE_cal(MR, AB4, L)
                        SBC4 = EE_cal(MR, BC4, L)
                        SCA4 = EE_cal(MR, CA4, L)
                        SABC4 = EE_cal(MR, ABC4, L)
                
                        TEE4 = SA4 + SB4 + SC4 - SAB4 - SBC4 - SCA4 + SABC4
                
                        TEE1_sum += TEE1
                        TEE2_sum += TEE2
                        TEE3_sum += TEE3
                        TEE4_sum += TEE4
                
                        count += 1


                        print(f"rank={rank:>3} Lx=Ly={Lx:>2} pg={pg:>6.3f} ids={ids:>4}", 
                              f"TEE1={TEE1:>6.3f} TEE2={TEE2:>6.3f} TEE3={TEE3:>6.3f}  TEE4={TEE4:>6.3f}", flush=True)
                # 平均を計算
                TEE1 = TEE1_sum / count
                TEE2 = TEE2_sum / count
                TEE3 = TEE3_sum / count
                TEE4 = TEE4_sum / count
            
                            
                print(f"rank={rank:>3} Lx=Ly={Lx:>2} pg={pg:>6.3f} ids={ids:>4}", 
                      f"TEE1={TEE1:>6.3f} TEE2={TEE2:>6.3f} TEE3={TEE3:>6.3f}  TEE4={TEE4:>6.3f}", flush=True)
                TEE1_list=np.append(TEE1_list,TEE1)
                TEE2_list=np.append(TEE2_list,TEE2)
                TEE3_list=np.append(TEE3_list,TEE3)
                TEE4_list=np.append(TEE3_list,TEE4)
                
                # trajectryの最後のMRの保存用
                #if ids==Nd-1:
                    #np.save(f"MR_Lx_{Lx}_pg_{pg:.2f}.npy", MR)
    
        key = (Lx, round(pg, 3))
        if key not in local_results:
            local_results[key] = []
        local_results[key].append((TEE1, TEE2, TEE3, TEE4)) 

# --- 4. MPI集約 ---

all_results = comm.gather(local_results, root=0)

if rank == 0:
    merged_results = {}
    for res in all_results:
        for key, val_list in res.items():
            if key not in merged_results:
                merged_results[key] = []
            merged_results[key].extend(val_list)

    # Lxごとにまとめて結果計算・表示・保存
    data_set = {}
    for key in merged_results:
        Lx, pg = key
        vals = np.array(merged_results[key])  # shape: (num_samples, 3)
        TEE1_list = vals[:, 0]
        TEE2_list = vals[:, 1]
        TEE3_list = vals[:, 2]
        TEE4_list = vals[:, 3]

        def ave_err_var(arr):
            ave = np.mean(arr)
            err = np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            var = np.var(arr)
            return ave, err, var

        ave1, err1, var1 = ave_err_var(TEE1_list)
        ave2, err2, var2 = ave_err_var(TEE2_list)
        ave3, err3, var3 = ave_err_var(TEE3_list)
        ave4, err4, var4 = ave_err_var(TEE4_list)

        print(f"Lx={Lx}, pg={pg:.3f} :")
        print(f"  TEE1: ave={ave1:.4f} ± {err1:.4f}, var={var1:.4f}")
        print(f"  TEE2: ave={ave2:.4f} ± {err2:.4f}, var={var2:.4f}")
        print(f"  TEE3: ave={ave3:.4f} ± {err3:.4f}, var={var3:.4f}")
        print(f"  TEE3: ave={ave4:.4f} ± {err4:.4f}, var={var4:.4f}")

        # 結果をLx単位で格納
        data_set.setdefault(Lx, {}).setdefault("pg", []).append(pg)
        for label, a, e, v in zip(
            ["TEE1", "TEE2", "TEE3", "TEE4"],
            [ave1, ave2, ave3, ave4],
            [err1, err2, err3, err4],
            [var1, var2, var3, var4],
            ):
            data_set[Lx].setdefault(f"{label}_ave", []).append(a)
            data_set[Lx].setdefault(f"{label}_err", []).append(e)
            data_set[Lx].setdefault(f"{label}_var", []).append(v)

    # npzファイル保存
    for Lx in data_set:
        np.savez_compressed(f"Lx_{Lx}_Nd_{Nd}_NT_{NT}_px_{px}_MPI_hexagonal1234.npz", 
                            pg=data_set[Lx]["pg"],
                            TEE1_ave=data_set[Lx]["TEE1_ave"],
                            TEE1_err=data_set[Lx]["TEE1_err"],
                            TEE1_var=data_set[Lx]["TEE1_var"],
                            TEE2_ave=data_set[Lx]["TEE2_ave"],
                            TEE2_err=data_set[Lx]["TEE2_err"],
                            TEE2_var=data_set[Lx]["TEE2_var"],
                            TEE3_ave=data_set[Lx]["TEE3_ave"],
                            TEE3_err=data_set[Lx]["TEE3_err"],
                            TEE3_var=data_set[Lx]["TEE3_var"],
                            TEE4_ave=data_set[Lx]["TEE4_ave"],
                            TEE4_err=data_set[Lx]["TEE4_err"],
                            TEE4_var=data_set[Lx]["TEE4_var"])

    # タイム出力など
    time_end = time.time()
    print("px=",px,"Np=",Np,"Nd=",Nd,"NT=",NT,"time=",int(time_end-time_start))

rank_final = MPI.COMM_WORLD.Get_rank()
print(f"[Rank {rank_final}] Reached end of script.")
MPI.Finalize()