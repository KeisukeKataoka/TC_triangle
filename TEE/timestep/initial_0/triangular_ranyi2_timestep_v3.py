# initial Z px=0 (px,pg)=(1,0) to (px,pg)=(0,1)

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
    if case == 0:
        #Case 0: 0
        for ii in range(Ld):
            MRi[ii][ii]=0
    if case == 1:
        #Case I: X product state
        for ii in range(Ld):
            MRi[ii][ii]=1 # X
    if case == 2:
        #Case II: Z product state
        for ii in range(Ld):
            MRi[ii][Ld+ii]=1 # Z
    if case == 3:
        #Case III: Y product state
        for ii in range(Ld):
            MRi[ii][ii]=1 
            MRi[ii][Ld+ii]=1 
    if case ==4:
        ##Case IV: unique g.s. of TC 
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
                MRi[Ld-2][Ld+v_ind[iv][0]]=1 #Z
            # x-direction loop logical Z
            for kk in range(Ly):
                iv=(0%Lx)+Lx*(kk%Ly)
                MRi[Ld-1][Ld+v_ind[iv][2]]=1 #Z
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

    #print("このオペレーターが")
    #print(MeoF)    
    # dMR without foctor
    dMRF=np.zeros((Ld,2*Ld),dtype='uint32')
    dMRF=dMR

    #print("この状態に作用")   
    #print(dMR)    
    
    #check anti commtation
    Mgs=np.dot(dMRF,np.dot(Gcd,MeoF.T))      
    Mgs=Mgs%2  ## check: reduce Z2 value 

    #print("反交換するスタビライザーは１を返すのでその場所を探す")     
    #print(Mgs)  
    
    antic_index_list=[]
    aid=np.where(Mgs[:]!=0)
    antic_index_list=aid[0]

    #print(aid[0],"番目のスタビライザーと反交換する")
    
    #######
    lenMe=len(antic_index_list)
    if lenMe !=0:
        kc=antic_index_list[0]
        dMR_prev=dMR[kc].copy()
        #replace stabilizer for kc
        
        dMR[kc]=Meo
        #print(kc,"番目のスタビライザーとオペレーターを交換")
        #print(dMR) 
        
        update_list=np.delete(antic_index_list,0)
        
        dMR[update_list,:]=np.mod(dMR[update_list,:]+dMR_prev,2)
        #print("反交換するスタビライザーの残りは最初の反交換するスタビライザーをかけて交換するように変形する")
        #print(update_list,"番目のスタビライザーを更新")
        #print(dMR) 
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

#####    Renyi2-correlator-csr  ######
def Renyi2_create(Ld,Lv,Lxd,Lyd):
    STx = np.zeros((2*Ld,(Lxd-1)*Lyd),dtype='uint32')
    STz = np.zeros((2*Ld,(Lxd-1)*Lyd),dtype='uint32')
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
    STx = np.zeros((2*Ld,Lyd),dtype='uint32')
    STz = np.zeros((2*Ld,Lyd),dtype='uint32')
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
    Gc=np.zeros((2*L,2*L),dtype='uint32') # 2L * 2L matrix
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
################################

    
def dephasing_dlinkXZ(dMR,ipd,dired,nsdd2,Ld,Lxd,Lyd,LOcheckd):
    #target dephasing links (please check 9/9)
    ix,iy = ipd%Lxd,ipd//Lxd
    if dired==0:
        if iy==0:
            diz=ix+(Ld-Lxd)
            dix=((ix+1)%Lxd)
        else:
            diz=ix+(Lxd)+(2*Lxd)*(iy-1)
            dix=(ix+1)%Lxd+(Lxd)+(2*Lxd)*(iy-1)+Lxd
    else:
        diz = ix+(Lxd)+(2*Lxd)*(iy-1)+Lxd
        dix = ix+(Lxd)+(2*Lxd)*(iy-1)+(2*Lxd)
    #print(888,ipd,dired,diz,dix)
    
    #di=di.astype('int64')
    MRXZ_bzbx=np.zeros((Ld),dtype='uint32')
    for kk in range(nsdd2):
        MRXZ_bzbx[kk]=dMR[kk,diz]+dMR[kk,dix+Ld] #(9/23,checked)
    MRXZ_bzbx=np.mod(MRXZ_bzbx,2)
    #################
    aid=np.where(MRXZ_bzbx[:]==1) # anticommute X search (9/9,please check)
    di_index_list=aid[0]
    lendi=len(di_index_list)
    # logical X string anticommute check
    if lendi !=0:
        if (di_index_list[0]==nsdd2-1) or (di_index_list[0]==nsdd2-2):
            #print(ipd,diz,dix)
            LOcheckd=1
            #print("logical X lost")

    if lendi !=0:
        k0=di_index_list[0]
        dMR_prev=dMR[k0,:].copy()
        #replace stabilizer for kc
        update_list=np.delete(di_index_list,0)
        dMR[update_list,:]=np.mod(dMR[update_list,:]+dMR_prev,2)
        dMR = np.delete(dMR,k0,0)
        nsdd2-=1
    #print("dephasingZZ2-a",(dMR2[:,di]+dMR2[:,(di+1)%Ld]))
    #print("------------------")
    return dMR,nsdd2,LOcheckd








Lx,Ly=12,12

Lv=Lx*Ly # total # of vertex
L=3*Lv # total # of link qubits
Lp=2*Lv  # total # of plaquette
#NT=6*L #*L #4*L total time step

#sample number
Nd = 10
#total timestep
Np = 11

alpha=4

NT=alpha*L #*L #4*L total time step

####setting####
#case0= 0, case1= X,case2= Z, case3= Y, case4= ground state of TC
case=3

px=0.0

# --- 1. 計算タスクの全リスト作成 ---
data_set = {}
tasks = []
for ip in range(Np):
    for ids in range(Nd):
        tasks.append((ip, ids))

# --- 2. MPI rankに均等割り当て ---
my_tasks = [tasks[i] for i in range(len(tasks)) if i % size == rank]

# --- 3. 計算結果をためる辞書 ---
local_results = {}


p_indd, v_indd=create_transformation_pv_link(Lv, Lx, Ly)

Gc_csr=Gc(L)
Gc = Gc_csr.toarray()

p_list=[]
t_list=[]

TEE_ave=[]
TEE_error=[]
TEE_var=[]

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


# stabilizer matrix
MR=np.zeros((L,2*L),dtype='uint32') # L * 2L matrix
# sample set of physical quantity
MR0=initial_stabilizer_state(case,L,Lx,Ly,v_indd,p_indd)

NT0 = 10

NT1 = NT//NT0

LOQx_list = np.zeros((Np, Nd, NT1+1))
LOQz_list = np.zeros((Np, Nd, NT1+1))
TEE_list = np.zeros((Np, Nd, NT1+1))
R2x_list = np.zeros((Np, Nd, NT1+1))
R2z_list = np.zeros((Np, Nd, NT1+1))
R2x_loop_list = np.zeros((Np, Nd, NT1+1))
R2z_loop_list = np.zeros((Np, Nd, NT1+1))

for (ip, ids) in my_tasks:
    MR=MR0.copy()
    
    ps=0.0 # ps < pg < 1
    pl=1.0
        
    pg=ps+(pl-px-ps)*ip/(Np-1)
    pz=1-px-pg

    prob_list=[pg,px,pz]
    todo=[1,2,3]

    p_list=np.append(p_list,pg)

    Ax,Ay=Lx//2,Ly//2
    Bx,By=Ax-2,Ay-1
    Cx,Cy=Ax-1,Ay-2     

    #ベンゼン型
    A = subset6(L,Ax,Ay,Lx,Ly,p_indd)
    B = subset6(L,Bx,By,Lx,Ly,p_indd) - A
    C = subset6(L,Cx,Cy,Lx,Ly,p_indd) - A -B

    #ナフタレン型
    #A = subset6(L,Ax,Ay,Lx,Ly,p_indd)|subset6(L,Ax-1,Ay+1,Lx,Ly,p_indd)
    #B = subset6(L,Bx,By,Lx,Ly,p_indd)|subset6(L,Bx-1,By-2,Lx,Ly,p_indd) - A
    #C = subset6(L,Cx,Cy,Lx,Ly,p_indd)|subset6(L,Cx+2,Cy+1,Lx,Ly,p_indd) - A -B

    #アントラセン型
    #A = subset6(L,Ax,Ay,Lx,Ly,p_indd)|subset6(L,Ax-1,Ay+1,Lx,Ly,p_indd)|subset6(L,Ax-3,Ay,Lx,Ly,p_indd)
    #B = subset6(L,Bx,By,Lx,Ly,p_indd)|subset6(L,Bx-1,By-2,Lx,Ly,p_indd)|subset6(L,Bx,By-3,Lx,Ly,p_indd) - A
    #C = subset6(L,Cx,Cy,Lx,Ly,p_indd)|subset6(L,Cx+2,Cy+1,Lx,Ly,p_indd)|subset6(L,Cx+3,Cy+1,Lx,Ly,p_indd) - A -B


    AB= A|B
    BC= B|C
    CA= C|A
    ABC=AB|C
    
    for it in range(NT):
        t_list=np.append(t_list,it)
        mtype= np.random.choice(todo,size=None,replace=True,p=prob_list)
        MR=measurement_op(MR,L,Lx,Ly,v_indd,p_indd,Gc,mtype)
        
        rank_MR=rank_mod2_v3(MR)
        #print(it,rank_MR)
            
        SA=EE_cal(MR,A,L)
        SB=EE_cal(MR,B,L)
        SC=EE_cal(MR,C,L)
        SAB=EE_cal(MR,AB,L)
        SBC=EE_cal(MR,BC,L)
        SCA=EE_cal(MR,CA,L)
        SABC=EE_cal(MR,ABC,L)

        TEE=SA+SB+SC-SAB-SBC-SCA+SABC

        STx_csr,STz_csr=Renyi2_create(L,Lv,Lx,Ly)
        STx_loop_csr,STz_loop_csr=Renyi2_loop_create(L,Lv,Lx,Ly)


        ### x-loop ###
        STx_loop_csr_T = STx_loop_csr.transpose()
        STx_loop_dense_T = STx_loop_csr_T.toarray()
        
        rank_STx_loop=rank_mod2_v3(STx_loop_dense_T)      
        
        combined_x = np.vstack([MR, STx_loop_dense_T])
        rank_combined_x = rank_mod2_v3(combined_x.copy())

        ### z-loop ###
        STz_loop_csr_T = STz_loop_csr.transpose()
        STz_loop_dense_T = STz_loop_csr_T.toarray()
        
        rank_STz_loop=rank_mod2_v3(STz_loop_dense_T)      
        
        combined_z = np.vstack([MR, STz_loop_dense_T])
        rank_combined_z = rank_mod2_v3(combined_z.copy())
  
        dep_x=rank_combined_x-rank_MR
        dep_z=rank_combined_z-rank_MR        
        
        R2x=Renyi2_csr(MR,Gc_csr,STx_csr,Lx,Ly)
        R2z=Renyi2_csr(MR,Gc_csr,STz_csr,Lx,Ly)
        R2x_loop=Renyi2_csr(MR,Gc_csr,STx_loop_csr,Lx,Ly)
        R2z_loop=Renyi2_csr(MR,Gc_csr,STz_loop_csr,Lx,Ly)

        R2x_loop = R2x_loop*(Lx-1)
        R2z_loop = R2z_loop*(Lx-1)
        #print("ids=",ids,"it=",it,"TEE=",TEE)
        #print("ids=",ids,"it=",it,"R2z_loop=",R2z_loop)
        if it % NT0 == 0:
            it1 = it // NT0
            TEE_list[ip, ids, it1] = TEE
            R2x_list[ip, ids, it1] = R2x
            R2z_list[ip, ids, it1] = R2z
            R2x_loop_list[ip, ids, it1] = R2x_loop
            R2z_loop_list[ip,ids, it1] = R2z_loop
            LOQx_list[ip, ids, it1] = dep_x
            LOQz_list[ip, ids, it1] = dep_z
            #print(f"rank={rank:>2} Lx=Ly={Lx:>2} it={it:>4} pg={pg:>6.3f} ids={ids:>4} R2z_loop={R2z_loop}", flush=True)
            #print(f"ip={ip}, ids={ids}, it={it}, shape={TEE_list.shape}")
        if it == 0 or it == NT - 1:
            #print("ids=",ids,"it=",it,"TEE=",TEE,"R2x=",R2x,"R2z=",R2z,"R2x_loop=",R2x_loop,"R2z_loop=",R2z_loop,"LOQx=",dep_x,"LOQz=",dep_z)
            print(f"Lx=Ly={Lx:>2} rank={rank:>2} ids={ids:>4} pg={pg:>6.3f} it={it:>4} TEE={TEE}", flush=True)

        key = (round(pg, 3),it,ids)
        if key not in local_results:
            local_results[key] = []
        local_results[key].append([pg, it, TEE, R2x, R2z, R2x_loop, R2z_loop, dep_x, dep_z])

time_end = time.time()
#print("pg=","{:.2f}".format(pg),"Nd=",Nd,"NT=",NT,"time=",int(time_end-time_start))

# --- 4. MPI集約 ---

all_results = comm.gather(local_results, root=0)

if rank == 0:
    merged_results = {}
    for res in all_results:
        for key, val_list in res.items():
            if key not in merged_results:
                merged_results[key] = []
            merged_results[key].extend(val_list)

    pg_values = sorted(set([key[0] for key in merged_results.keys()]))
    pg_to_idx = {pg: i for i, pg in enumerate(pg_values)}

    for pg in pg_values:
        pz=1-pg-px
        TEE_arr = np.zeros((NT, Nd))
        R2x_arr = np.zeros((NT, Nd))
        R2z_arr = np.zeros((NT, Nd))
        R2x_loop_arr = np.zeros((NT, Nd))
        R2z_loop_arr = np.zeros((NT, Nd))
        dep_x_arr = np.zeros((NT, Nd))
        dep_z_arr = np.zeros((NT, Nd))

        for (pg_val, it, ids), vals_list in merged_results.items():
            if pg_val != pg:
                continue
            for (_, it, TEE, R2x, R2z, R2x_loop, R2z_loop, dep_x, dep_z) in vals_list:
                TEE_arr[it, ids] = TEE
                R2x_arr[it, ids] = R2x
                R2z_arr[it, ids] = R2z
                R2x_loop_arr[it, ids] = R2x_loop
                R2z_loop_arr[it, ids] = R2z_loop
                dep_x_arr[it, ids] = dep_x
                dep_z_arr[it, ids] = dep_z

        # ファイル名にpgの値を含めて保存
        np.savez_compressed(
            f"results_Lx_{Lx}_Nd_{Nd}_NT_{alpha}L_px_{px:.3f}_pz_{pz:.3f}_pg_{pg:.3f}.npz",  # 例: results_pg=0.100.npz
            pg=pg,
            TEE=TEE_arr,
            R2x=R2x_arr,
            R2z=R2z_arr,
            R2x_loop=R2x_loop_arr,
            R2z_loop=R2z_loop_arr,
            dep_x=dep_x_arr,
            dep_z=dep_z_arr,
        )



