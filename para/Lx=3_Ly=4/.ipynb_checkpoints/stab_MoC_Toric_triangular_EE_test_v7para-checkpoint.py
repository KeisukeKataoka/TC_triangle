#v7:initial stateをproduct xにしてpx=0,pg+pz=1でgammaを調べる



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
from numba import njit,c16,i8,u2
import sys
import math


import multiprocessing as mp

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.random.seed(12345)

start = time.time()

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
                MRi[Ld-1][Ld+v_ind[iv][1]]=1 #Z
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

####################
# Asub_set Asub_setを合わせていく
####################
def subset(Ld,Ax, Ay):
    i = idx(Ax, Ay)
    Asub_set = set()
    for j in range(3):
        Asub_set.add(p_e_indd[i][j])
        Asub_set.add(p_e_indd[i + Ld // 3][j])
    return Asub_set

####################
#EE
####################
# https://arxiv.org/abs/2204.08489
def EE_cal(MRd,Asub_set,Ld,p_ind):
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

def single_sample_EE(args):
    MR0, L, Lx, Ly, v_e_indd, p_e_indd, Gc, prob_list, NT, Asub_set, seed = args
    np.random.seed(seed)  
    MR = MR0.copy()
    todo = [1, 2, 3]
    for it in range(NT):
        mtype = np.random.choice(todo, size=None, replace=True, p=prob_list)
        MR = measurement_op(MR, L, Lx, Ly, v_e_indd, p_e_indd, Gc, mtype)
    EE_val = EE_cal(MR, Asub_set, L, p_e_indd)
    return EE_val

############################

### Main part ###
## triangular lattice under PBC##
## parameter
#set the vertex lattice system size 
Lx=3
Ly=4

Lv=Lx*Ly # total # of vertex
L=3*Lv # total # of link qubits
Lp=2*Lv  # total # of plaquette

case=1 # case=1:X product state, case=2:Z product state, case=3:unique g.s. of TC  

###############################################
# Create 3-directed edge index for each vertex
###############################################
#v_3e_indd = np.zeros((Lx*Ly,3),dtype='uint32')
def idx(x, y):
    return (x % Lx)+Lx*(y % Ly)
def inv_idx(i):
    return i % Lx, i // Lx
# 方向 0: (x+1, y), 1: (x, y+1), 2: (x-1, y+1)
directions = [(+1, 0), (0, +1), (-1, +1)]
# 一意な辺のリスト（方向0,1,2のみ使って構築）
edge_set = set()
for y in range(Ly):
    for x in range(Lx):
        i = idx(x, y)
        for dx, dy in directions:
            j = idx(x + dx, y + dy)
            edge = tuple(sorted((i, j)))
            edge_set.add(edge)
# 辞書順に並べてedge indexを定義
sorted_edges = sorted(edge_set)
edge_to_index = {e: n for n, e in enumerate(sorted_edges)}

####################################
# 関数：vertex index i に対して方向 0, 1, 2 の edge_index を返す
def get_3edge_indices_from_vertex(i):
    x, y = inv_idx(i)
    indices = []
    for dx, dy in directions:
        j = idx(x + dx, y + dy)
        edge = tuple(sorted((i, j)))
        indices.append(edge_to_index[edge])
    return indices
####################################
# Create vertex operator edge-labels
v_e_indd = np.zeros((Lx*Ly,6),dtype='uint32')
for iy in range(Ly):
    for ix in range(Lx):
        ic=(ix%Lx)+Lx*(iy%Ly)
        icmx=((ix-1)%Lx)+Lx*(iy%Ly)
        icmy=((ix)%Lx)+Lx*((iy-1)%Ly)
        icpxmy=((ix+1)%Lx)+Lx*((iy-1)%Ly)       
        v_e_indd[ic][0]=get_3edge_indices_from_vertex(ic)[0]
        v_e_indd[ic][1]=get_3edge_indices_from_vertex(ic)[1]
        v_e_indd[ic][2]=get_3edge_indices_from_vertex(ic)[2]
        v_e_indd[ic][3]=get_3edge_indices_from_vertex(icmx)[0]
        v_e_indd[ic][4]=get_3edge_indices_from_vertex(icmy)[1]
        v_e_indd[ic][5]=get_3edge_indices_from_vertex(icpxmy)[2]
#######################################
# Create plaquette operator edge-labels
p_e_indd = np.zeros((2*Lx*Ly,3),dtype='uint32')
for iy in range(Ly):
    for ix in range(Lx):
        #上三角
        ic=(ix%Lx)+Lx*(iy%Ly)
        icpx=((ix+1)%Lx)+Lx*(iy%Ly)
        p_e_indd[ic][0]=get_3edge_indices_from_vertex(ic)[0]
        p_e_indd[ic][1]=get_3edge_indices_from_vertex(icpx)[2]
        p_e_indd[ic][2]=get_3edge_indices_from_vertex(ic)[1]
        #下三角
        ic=(ix%Lx)+Lx*(iy%Ly)
        icpy=((ix)%Lx)+Lx*((iy+1)%Ly)
        p_e_indd[ic+Lx*Ly][0]=get_3edge_indices_from_vertex(icpx)[1]
        p_e_indd[ic+Lx*Ly][1]=get_3edge_indices_from_vertex(icpy)[0]
        p_e_indd[ic+Lx*Ly][2]=get_3edge_indices_from_vertex(icpx)[2]
### check matrix ###
Gc=np.zeros((2*L,2*L),dtype='uint32') # 2L * 2L matrix
for k in range(L):
    Gc[k][L+k]=1
    Gc[L+k][k]=1  
Gc_csr=csr_matrix(Gc)


NT=4*L

### Main part ###
Asub_set = set()
for Ay in range(Ly-1):
    for Ax in range(Lx-1):
        Np=11
        print("Ax,Ay=",Ax,Ay)
        Asub_set |= subset(L,Ax, Ay)  # ← Asub_setのedgeの和集合
        ip=10
        Nd = 2 #800 sample number
        #for ip in range(len(p_list)):
        base_seed = 12345
        args_list = []
        px=0.0
        pg=(1-px)*ip/(Np-1)
        pz=1-px-pg
        print("pg=",pg)
        prob_list=[pg,px,pz]
        # stabilizer matrix
        MR=np.zeros((L,2*L),dtype='uint32') # L * 2L matrix
        # sample set of physical quantity
        MR0=initial_stabilizer_state(case,L,Lx,Ly,v_e_indd,p_e_indd)#

        for i in range(Nd):
            seed = base_seed + i + int(pg*1e6)  # pgによってseed変化
            args_list.append((MR0, L, Lx, Ly, v_e_indd, p_e_indd, Gc, prob_list, NT, Asub_set, seed))

        SA_list=[]
         # 並列計算
        with mp.Pool(processes=mp.cpu_count()) as pool:
            SA_list = pool.map(single_sample_EE, args_list)
            # EE_list にNd個のEEが入る
            print("EE average:", np.mean(SA_list))
            print("Asubset:", Asub_set)
            SA_aved=np.mean(SA_list)
            SA_vard=np.var(SA_list,axis=0)
            SA_error=np.std(SA_list,axis=0, ddof=1)/np.sqrt(len(SA_list))
        if Ay ==0:
            with open("Ax="+str(Ax)+"_Ay="+str(Ay)+"_P="+str(2*(Ax+1)+2)+"_L="+str(L)+"_T="+str(NT)+".lst", 'a') as f:
                f.writelines(str(pg)+'\t'+str(SA_aved)+'\t'+str(SA_vard)+'\t'+str(SA_error)+"\n")
            print("P=",2*(Ax+1)+2,"S_A=",SA_aved)
        if Ay >0:
            with open("Ax="+str(Ax)+"_Ay="+str(Ay)+"_P="+str(2*(Lx-1)+2*(Ay+1))+"_L="+str(L)+"_T="+str(NT)+".lst", 'a') as f:
                f.writelines(str(pg)+'\t'+str(SA_aved)+'\t'+str(SA_vard)+'\t'+str(SA_error)+"\n")
            print("P=",2*(Lx+1)+2,"S_A=",SA_aved)

end = time.time()
print("並列計算にかかった時間: {:.2f} 秒".format(end - start))


