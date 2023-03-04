import math
from tkinter import X
import matplotlib.pyplot as plt
import numpy as np
from Lab1 import chunks, open_file
import navpy
import pymap3d.ecef as pym
import pymap3d.aer as pa
from itertools import permutations
from functools import reduce

def load_file(input,num):
    reffile=open_file(input,num)
    reffile=chunks(reffile,12)
    sat_phase=[[],[],[],[],[],[],[],[],[],[],[],[]]
    sat_doppler=[[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in reffile:
        for x in range(0,12):
            sat_phase[x].append(0)
            sat_doppler[x].append(0)
        for j in i:
            if j[0]==7.0:
                sat_phase[0].pop()
                sat_phase[0].append(j[3])
                sat_doppler[0].pop()
                sat_doppler[0].append(j[4])
            if j[0]==8.0:
                sat_phase[1].pop()
                sat_phase[1].append(j[3])
                sat_doppler[1].pop()
                sat_doppler[1].append(j[4])
            if j[0]==9.0:
                sat_phase[2].pop()
                sat_phase[2].append(j[3])
                sat_doppler[2].pop()
                sat_doppler[2].append(j[4])
            if j[0]==11.0:
                sat_phase[3].pop()
                sat_phase[3].append(j[3])
                sat_doppler[3].pop()
                sat_doppler[3].append(j[4])
            if j[0]==15.0:
                sat_phase[4].pop()
                sat_phase[4].append(j[3])
                sat_doppler[4].pop()
                sat_doppler[4].append(j[4])
            if j[0]==17.0:
                sat_phase[5].pop()
                sat_phase[5].append(j[3])
                sat_doppler[5].pop()
                sat_doppler[5].append(j[4])
            if j[0]==18.0:
                sat_phase[6].pop()
                sat_phase[6].append(j[3])
                sat_doppler[6].pop()
                sat_doppler[6].append(j[4])
            if j[0]==19.0:
                sat_phase[7].pop()
                sat_phase[7].append(j[3])
                sat_doppler[7].pop()
                sat_doppler[7].append(j[4])
            if j[0]==24.0:
                sat_phase[8].pop()
                sat_phase[8].append(j[3])
                sat_doppler[8].pop()
                sat_doppler[8].append(j[4])
            if j[0]==26.0:
                sat_phase[9].pop()
                sat_phase[9].append(j[3])
                sat_doppler[9].pop()
                sat_doppler[9].append(j[4])
            if j[0]==27.0:
                sat_phase[10].pop()
                sat_phase[10].append(j[3])
                sat_doppler[10].pop()
                sat_doppler[10].append(j[4])
            if j[0]==28.0:
                sat_phase[11].pop()
                sat_phase[11].append(j[3])
                sat_doppler[11].pop()
                sat_doppler[11].append(j[4])
    sat_cycleslips=[[],[],[],[],[],[],[],[],[],[],[],[]]
    
    for y in range(0,12):
        psedo=sat_phase[y]
        dop=sat_doppler[y]
        previous_phase=psedo[0]
        previous_dop=dop[0]
        for q in range(1,3600):
            current_phase=psedo[q]
            current_dop=dop[q]
            if current_phase!=0 and previous_phase==0:
                sat_cycleslips[y].append((q,(y+1)))
            else:
                k=previous_phase-(current_dop+previous_dop)/2
                value=abs(current_phase-k)
                if value>=1:
                    sat_cycleslips[y].append((q,(y+1)))
            previous_phase=current_phase
            previous_dop=current_dop
    return sat_cycleslips

def plot_cycleslip(input):
    ax=plt.subplot(111)
    for i in range(0,12):
        temp=input[i]
        x=[x[0] for x in temp]
        y=[x[1] for x in temp]
        ax.scatter(x,y)
    plt.legend(['PRN 7','PRN 8','PRN 9','PRN 11','PRN 15','PRN 17','PRN 18','PRN 19','PRN 24','PRN 26','PRN 27','PRN 28'],loc='best')
    ax.set_xlabel('epoch (s)(Time from beginning on test)')
    ax.set_ylabel('satellites prn(according to the legend)')
    ax.set_title('cycle slips as a function of time')
    plt.show()

def checkcommon():
    satfile=open_file("Satellites.sat",8)
    satfile=chunks(satfile,12)
    satprn=[]
    for i in satfile:
        temp=[]
        for j in i:
            if j[0]!=0:
                temp.append(j[0])
        satprn.append(temp)

    recfile=open_file("RemoteL1L2.obs",6)
    recfile=chunks(recfile,12)
    recprn=[]
    for i in recfile:
        temp=[]
        for j in i:
            if j[0]!=0:
                temp.append(j[0])
        recprn.append(temp)

    reffile=open_file("BaseL1L2.obs",6)
    reffile=chunks(reffile,12)
    refprn=[]
    for i in reffile:
        temp=[]
        for j in i:
            if j[0]!=0:
                temp.append(j[0])
        refprn.append(temp)

    result=[]
    for w in range(0,3600):
        set1=set(satprn[w])
        set2=set(recprn[w])
        set3=set(refprn[w])
        result.append(list(set1&set2&set3))
    return result

def cl_form():
    cl1=np.ones((9,1))
    cl2=np.diag([1,1,1,1,1,1,1,1,1])
    cl3=-cl1
    cl4=np.diag([-1,-1,-1,-1,-1,-1,-1,-1,-1])
    Bm=np.hstack((cl1,cl2))
    Bm=np.hstack((Bm,cl3))
    Bm=np.hstack((Bm,cl4))
    BmT=np.transpose(Bm)
    cl_code=np.diag([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    cl_phase=np.diag([0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2,0.02**2])
    cl_code=Bm@cl_code@BmT
    cl_phase=Bm@cl_phase@BmT
    cl_code=np.hstack((cl_code,np.zeros((9,9))))
    cl_phase=np.hstack((np.zeros((9,9)),cl_phase))
    cl=np.vstack((cl_code,cl_phase))
    return cl

def extract_data():
    target_prn=[7.0, 8.0, 11.0, 15.0, 17.0, 19.0, 24.0, 26.0, 27.0, 28.0]
    satfile=open_file("Satellites.sat",8)
    satfile=chunks(satfile,12)
    recfile=open_file("RemoteL1L2.obs",6)
    recfile=chunks(recfile,12)
    reffile=open_file("BaseL1L2.obs",6)
    reffile=chunks(reffile,12)
    newsat=[]
    for i in satfile[:1800]:
        temp=[]
        for j in i:
            if j[0] in target_prn:
                temp.append(j)
        newsat.append(temp)
    newremote=[]
    for i in recfile[:1800]:
        temp=[]
        for j in i:
            if j[0] in target_prn:
                temp.append(j)
        newremote.append(temp)
    newbase=[]
    for i in reffile[:1800]:
        temp=[]
        for j in i:
            if j[0] in target_prn:
                temp.append(j)
        newbase.append(temp)
    return newsat,newremote,newbase

def least_square(sat,remote,base):
    psedorange=[]
    phase=[]
    ep=0
    for i in remote:
        temp=[]
        temphase=[]
        for j in i:
            temp.append(j[2])
            temphase.append(-j[3])
        psedorange.append(temp)
        phase.append(temphase)
        ep+=1
    bpsedorange=[]
    bphase=[]
    ep=0
    for a in base:
        temp=[]
        temphase=[]
        for b in a:
            temp.append(b[2])
            temphase.append(-b[3])
        bpsedorange.append(temp)
        bphase.append(temphase)
        ep+=1
    result=[]
    residuals=[]
    stdv_x=[]
    stdv_y=[]
    stdv_z=[]
    stdv_ambs=[]
    eleva_angle=[]
    epoch=0 
    x,y,z=-1633489.6534772082, -3651626.453991796, 4952480.547976259
    ambiguity_vector=[14, 1, 12, 3, 4, -7, 21, 11, 13]
    ambiguity_vector=np.array(ambiguity_vector)
    ambiguity_vectors=[]
    x_base=51.27704
    y_base=-113.9832
    z_base=1090.833 
    xb,yb,zb=navpy.lla2ecef(x_base,y_base,z_base)
    lamd=299.792458/1575.42
    observation_covarience_matrix=cl_form()
    x_prior=np.array([x,y,z])
    x_prior=np.hstack((x_prior,ambiguity_vector))
    dia=[5,5,5,5,5,5,5,5,5,5,5,5]
    cx_matrix=np.diag(dia)
    # cx_matrix[0][0]=0.01
    # cx_matrix[1][1]=0.01
    # cx_matrix[2][2]=0.01

    for i in sat:
        observation=np.array(psedorange[epoch])
        other_observation=np.delete(observation,[6])
        bobservation=np.array(bpsedorange[epoch])
        other_bobservation=np.delete(bobservation,[6])
        bsat_pseudorange=observation[6]
        bsat_bpseudorange=bobservation[6]

        L1phase=np.array(phase[epoch])
        other_phase=np.delete(L1phase,[6])
        bsat_phase=L1phase[6]
        base_L1phase=np.array(bphase[epoch])
        other_baseL1phase=np.delete(base_L1phase,[6])
        bsat_baseL1phase=base_L1phase[6]
        
        H=[]
        Hd=[]
        Hr=np.zeros((9,9))
        diagonal=[1,1,1,1,1,1,1,1,1]
        Hrd=np.diag(diagonal)
        geodetic_differencing=[]
        for j in i:
            if j[0]==24.0:
                x_basesat=j[2]
                y_basesat=j[3]
                z_basesat=j[4]
                P_basesat2remote=math.sqrt((x_basesat-x)**2+(y_basesat-y)**2+(z_basesat-z)**2)
                P_basesat2base=math.sqrt((x_basesat-xb)**2+(y_basesat-yb)**2+(z_basesat-zb)**2)
        
        for j in i:
            if j[0]!=24.0:
                xs=j[2]
                ys=j[3]
                zs=j[4]
                P=math.sqrt((xs-x)**2+(ys-y)**2+(zs-z)**2)
                P_base=math.sqrt((xs-xb)**2+(ys-yb)**2+(zs-zb)**2)
                H.append([((x_basesat-x)/P_basesat2remote-(xs-x)/P),((y_basesat-y)/P_basesat2remote-(ys-y)/P),((z_basesat-z)/P_basesat2remote-(zs-z)/P)])
                Hd.append([((x_basesat-x)/P_basesat2remote-(xs-x)/P),((y_basesat-y)/P_basesat2remote-(ys-y)/P),((z_basesat-z)/P_basesat2remote-(zs-z)/P)])
                geodetic_differencing.append((P-P_base)-(P_basesat2remote-P_basesat2base))

        H=np.hstack((H,Hr))
        Hd=np.hstack((Hd,Hrd))
        H=np.vstack((H,Hd))
        H=np.array(H)
        HT=np.transpose(H)
        K=cx_matrix@HT@(np.linalg.inv(H@cx_matrix@HT+observation_covarience_matrix))
        Z_code=((other_observation-other_bobservation)-(bsat_pseudorange-bsat_bpseudorange))
        Z_phase=((other_phase-other_baseL1phase)-(bsat_phase-bsat_baseL1phase))*(lamd)
        
        misclosure_code=Z_code-geodetic_differencing
        misclosure_phase=Z_phase-geodetic_differencing-ambiguity_vector
        residuals.append(misclosure_phase/lamd)
        misclosure=np.hstack((misclosure_code,misclosure_phase))
        temp=K@misclosure
        x_current=x_prior+temp
        result.append(x_current)
        x=x_current[0]
        y=x_current[1]
        z=x_current[2]
        ambiguity_vector=np.array(x_current[3:12])
        ambiguity_vectors.append(ambiguity_vector/lamd)
        x_prior=x_current
        cx_matrix=cx_matrix-K@H@cx_matrix 
        CN=cx_matrix[3:12,3:12]
        epoch+=1 
        cx=cx_matrix[:3,:3]

        rotate=[[-math.sin(-114.1004916333),math.cos(-114.1004916333),0],[-math.cos(-114.1004916333)*math.sin(51.25864328333),-math.sin(-114.1004916333)*math.sin(51.25864328333),math.cos(51.25864328333)],[math.cos(-114.1004916333)*math.cos(51.25864328333),math.sin(-114.1004916333)*math.cos(51.25864328333),math.sin(51.25864328333)]]
        rotateT=np.transpose(rotate)
        cx=rotate@cx@rotateT
        stdv_x.append(math.sqrt(cx[0][0]))
        stdv_y.append(math.sqrt(cx[1][1]))
        stdv_z.append(math.sqrt(cx[2][2]))
        ambs_stdv=[]
        for o in range (3,12):
            ambs_stdv.append(math.sqrt(cx_matrix[o][o]))
        stdv_ambs.append(ambs_stdv)

    return x_current,misclosure,result,stdv_x,stdv_y,stdv_z,ambiguity_vector,CN,ambiguity_vectors,stdv_ambs,residuals

def cal_error(input):
    tempx=[]
    tempy=[]
    tempz=[]
    for i in input:
        x,y,z=pym.ecef2enu(i[0],i[1],i[2],51.25864328333,-114.1004916333,1127.345)
        tempx.append(x)
        tempy.append(y)
        tempz.append(z)
    return tempx,tempy,tempz

def plot_error(inputx,inputy,inputz,stdv_x,stdv_y,stdv_z):
    inputx=np.array(inputx)
    inputy=np.array(inputy)
    inputz=np.array(inputz)
    stdv_x=np.array(stdv_x)
    stdv_y=np.array(stdv_y)
    stdv_z=np.array(stdv_z)
    ax1=plt.subplot(2,3,1)
    plt.plot(range(0,1800),inputx,linewidth=0.3,color='g')
    ax1.set_xlabel('epoch (s)(Time from beginning on test)')
    ax1.set_ylabel('error (m)')
    ax1.set_title('East error time series plot (m)')
    ax2=plt.subplot(2,3,2)
    plt.plot(range(0,1800),inputy,linewidth=0.3,color='y')
    ax2.set_xlabel('epoch (s)(Time from beginning on test)')
    ax2.set_ylabel('error (m)')
    ax2.set_title('North error time series plot (m)')
    ax3=plt.subplot(2,3,3)
    plt.plot(range(0,1800),inputz,linewidth=0.3)
    ax3.set_xlabel('epoch (s)(Time from beginning on test)')
    ax3.set_ylabel('error (m)')
    ax3.set_title('Up error time series plot (m)')
    ax3=plt.subplot(2,3,4)
    ax3.plot(range(0,1800),stdv_x,linewidth=0.3)
    ax3.plot(range(0,1800),-(stdv_x),linewidth=0.3)
    ax3.set_xlabel('epoch (s)(Time from beginning on test)')
    ax3.set_ylabel('standard deviation (m)')
    ax3.set_title('East standard deviation time series plot (m)')
    ax3.legend(['positive','negative'],loc='best')
    ax3=plt.subplot(2,3,5)
    ax3.plot(range(0,1800),stdv_y,linewidth=0.3)
    ax3.plot(range(0,1800),-(stdv_y),linewidth=0.3)
    ax3.set_xlabel('epoch (s)(Time from beginning on test)')
    ax3.set_ylabel('standard deviation (m)')
    ax3.set_title('North standard deviation time series plot (m)')
    ax3.legend(['positive','negative'],loc='best')
    ax3=plt.subplot(2,3,6)
    ax3.plot(range(0,1800),stdv_z,linewidth=0.3)
    ax3.plot(range(0,1800),-(stdv_z),linewidth=0.3)
    ax3.set_xlabel('epoch (s)(Time from beginning on test)')
    ax3.set_ylabel('standard deviation (m)')
    ax3.set_title('Up standard deviation time series plot (m)')
    ax3.legend(['positive','negative'],loc='best')
    plt.show()

def find_integer(input,dif):
    output=[]
    for i in range(math.ceil(input-dif),int(input+dif)+1):
        output.append(i)
    return output

def cal_value(ambs):
    integer_ambs=[]
    for i in ambs:
        temp=find_integer(i,2)
        # value=np.transpose(temp)@np.linalg.inv(CN)@temp
        integer_ambs.append(temp)
    return integer_ambs

def lists_com(lists,CN,ambs):
    sos_group=[]
    combination=[[q,w,e,r,t,y,u,i,o] for q in lists[0]for w in lists[1] for e in lists[2]for r in lists[3]for t in lists[4]for y in lists[5] for u in lists[6]for i in lists[7]for o in lists[8]]
    print(combination[218410])
    combination=combination-ambs
    for i in combination:
        i=np.array(i)
        iT=np.transpose(i)
        sos=iT@np.linalg.inv(CN)@i
        sos_group.append(sos)
    return sos_group,combination

def plot_ambs(input,input2): 
    ax=plt.subplot(111)
    ax.plot(range(0,1800),input)
    ax.plot(range(0,1800),input2)
    # plt.legend(['PRN 7','PRN 8','PRN 9','PRN 11','PRN 15','PRN 17','PRN 18','PRN 19','PRN 24','PRN 26','PRN 27','PRN 28'],loc='best')
    ax.set_xlabel('epoch (s)(Time from beginning on test)')
    ax.set_ylabel('ambiguity (m)')
    ax.set_title('Float ambiguities and fixed ambiguities')
    plt.show()

def plot_stdv(input): 
    input=np.array(input)
    ax=plt.subplot(111)
    ax.plot(range(0,1800),input)
    ax.plot(range(0,1800),-(input))
    # plt.legend(['PRN 7','PRN 8','PRN 9','PRN 11','PRN 15','PRN 17','PRN 18','PRN 19','PRN 24','PRN 26','PRN 27','PRN 28'],loc='best')
    ax.set_xlabel('epoch (s)(Time from beginning on test)')
    ax.set_ylabel('residuals (m)')
    ax.set_title('standard deviations of ambiguities')
    plt.show()

def plot_res(input): 
    input=np.array(input)
    ax=plt.subplot(111)
    ax.plot(range(0,1800),input)
    ax.set_xlabel('epoch (s)(Time from beginning on test)')
    ax.set_ylabel('residuals (m)')
    ax.set_title('residuals time series plot')
    plt.show()

def main():
    # cycleslip=load_file("BaseL1L2.obs",6)
    # plot_cycleslip(cycleslip)
    # cycleslip=load_file("RemoteL1L2.obs",6)
    # plot_cycleslip(cycleslip)
    # common=checkcommon()
    # for i in common:
    #     if i ==[7.0, 8.0, 9.0, 11.0, 15.0, 17.0, 18.0, 19.0, 24.0, 26.0, 27.0, 28.0]:
    #         print(common.index(i))
    #         break
    sat,remote,base=extract_data()
    # print(sat[0],remote[0],base[0])
    x_vector,misclosure,error,stdv_x,stdv_y,stdv_z,ambs,CN,ambs_epoch,stdv_ambs,res=least_square(sat,remote,base)
    print(res)
    plot_res(res)
    # ints=[]
    # for i in range(0,1800):
    #     ints.append([14, 1, 12, 3, 4, -7, 21, 11, 13])
    # print(x_vector)
    # plot_ambs(ambs_epoch,ints)
    # plot_res(stdv_ambs)
    # ambs=ambs/(299.792458/1575.42)
    # Xerror,yerror,zerror=cal_error(error)
    # plot_error(Xerror,yerror,zerror,stdv_x,stdv_y,stdv_z)
    # int=cal_value(ambs)
    # a,com=lists_com(int,CN,ambs)
    # a.sort()
    # print(a[1]/a[0])
main()
