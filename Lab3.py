import math
from tkinter import X
import matplotlib.pyplot as plt
import numpy as np
from Lab1 import chunks, open_file
import navpy
import pymap3d.ecef as pym
import pymap3d.aer as pa

def load_file():
    satfile=open_file("Satellites.sat",8)
    satfile=chunks(satfile,12)
    recfile=open_file("RemoteL1L2.obs",6)
    recfile=chunks(recfile,12)
    reffile=open_file("BaseL1L2.obs",6)
    reffile=chunks(reffile,12)
    common=checkcommon()
    psedorange=[]
    bpsedorange=[]
    ep=0
    for i in recfile:
        temp=[]
        for j in i:
            if j[0] in common[ep]:
                temp.append(j[2])
        psedorange.append(temp)
        ep+=1
    ep=0
    for a in reffile:
        temp=[]
        for b in a:
            if b[0] in common[ep]:
                temp.append(b[2])
        bpsedorange.append(temp)
        ep+=1
    result=[]
    residuals=[]
    stdv_x=[]
    stdv_y=[]
    stdv_z=[]
    eleva_angle=[]
    epoch=0
    x,y,z,clock=0,0,0,0
    x_base=51.27704
    y_base=-113.9832
    z_base=1090.833 
    xb,yb,zb=navpy.lla2ecef(x_base,y_base,z_base)
    for i in satfile:
        observation=psedorange[epoch]
        bobservation=bpsedorange[epoch]
        delta=[1,1,1,1]
        iden_matrix=np.identity(len(observation))
        while max(delta)>0.01:
            H=[]
            rou_cdt=[]
            for j in i:
                if j[0] in common[epoch]:
                    xs=j[2]
                    ys=j[3]
                    zs=j[4]
                    P=math.sqrt((xs-x)**2+(ys-y)**2+(zs-z)**2)
                    P_base=math.sqrt((xs-xb)**2+(ys-yb)**2+(zs-zb)**2)
                    rou_cdt.append(P+clock-P_base)
                    H.append([-(xs-x)/P,-(ys-y)/P,-(zs-z)/P,1])
            H=np.array(H)
            HT=np.transpose(H)
            delta=np.matmul(HT,np.linalg.inv(iden_matrix))
            delta=np.linalg.inv(np.matmul(delta,H))
            delta=np.matmul(delta,HT)
            delta=np.matmul(delta,np.linalg.inv(iden_matrix))
            w=(observation-np.array(bobservation))-np.array(rou_cdt)
            delta=np.matmul(delta,w)
            x+=delta[0]
            y+=delta[1]
            z+=delta[2]
            clock+=delta[3]
            x_result=[x,y,z,clock]
        
        epoch+=1 
        result.append(x_result)

        residual=(observation-np.array(bobservation))-np.array(rou_cdt)
        residuals.append(residual)
        
        cx=np.matmul(HT,np.linalg.inv(iden_matrix))
        cx=np.linalg.inv(np.matmul(cx,H))
        stdv_x.append(math.sqrt(cx[0][0]))
        stdv_y.append(math.sqrt(cx[1][1]))
        stdv_z.append(math.sqrt(cx[2][2]))

        al,output,sr=pa.ecef2aer(xs,ys,zs,51.25864,-114.1005,1127.345)
        eleva_angle.append(output)

    sat_residuals=[[],[],[],[],[],[],[],[],[],[],[],[]]
    for l in range(0,3600):
        temp_common=common[l]
        for p in range(0,len(temp_common)):
            if temp_common[p]==7.0:
                sat_residuals[0].append(residuals[l][p])
            elif temp_common[p]==8.0:
                sat_residuals[1].append(residuals[l][p])
            elif temp_common[p]==9.0:
                sat_residuals[2].append(residuals[l][p])
            elif temp_common[p]==11.0:
                sat_residuals[3].append(residuals[l][p])
            elif temp_common[p]==15.0:
                sat_residuals[4].append(residuals[l][p])
            elif temp_common[p]==17.0:
                sat_residuals[5].append(residuals[l][p])
            elif temp_common[p]==18.0:
                sat_residuals[6].append(residuals[l][p])
            elif temp_common[p]==19.0:
                sat_residuals[7].append(residuals[l][p])
            elif temp_common[p]==24.0:
                sat_residuals[8].append(residuals[l][p])
            elif temp_common[p]==26.0:
                sat_residuals[9].append(residuals[l][p])
            elif temp_common[p]==27.0:
                sat_residuals[10].append(residuals[l][p])
            elif temp_common[p]==28.0:
                sat_residuals[11].append(residuals[l][p])
    return result,stdv_x,stdv_y,stdv_z,sat_residuals,residuals

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

def cal_error(input):
    tempx=[]
    tempy=[]
    tempz=[]
    for i in input:
        x,y,z=pym.ecef2enu(i[0],i[1],i[2],51.25864,-114.1005,1127.345)
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
    plt.plot(range(0,3600),inputx,linewidth=0.3,color='g')
    ax1.set_xlabel('epoch (s)(Time from beginning on test)')
    ax1.set_ylabel('error (m)')
    ax1.set_title('East error time series plot (m)')
    ax2=plt.subplot(2,3,2)
    plt.plot(range(0,3600),inputy,linewidth=0.3,color='y')
    ax2.set_xlabel('epoch (s)(Time from beginning on test)')
    ax2.set_ylabel('error (m)')
    ax2.set_title('North error time series plot (m)')
    ax3=plt.subplot(2,3,3)
    plt.plot(range(0,3600),inputz,linewidth=0.3)
    ax3.set_xlabel('epoch (s)(Time from beginning on test)')
    ax3.set_ylabel('error (m)')
    ax3.set_title('Up error time series plot (m)')
    ax3=plt.subplot(2,3,4)
    ax3.plot(range(0,3600),stdv_x,linewidth=0.3)
    ax3.plot(range(0,3600),-(stdv_x),linewidth=0.3)
    ax3.set_xlabel('epoch (s)(Time from beginning on test)')
    ax3.set_ylabel('standard deviation (m)')
    ax3.set_title('East standard deviation time series plot (m)')
    ax3.legend(['positive','negative'],loc='best')
    ax3=plt.subplot(2,3,5)
    ax3.plot(range(0,3600),stdv_y,linewidth=0.3)
    ax3.plot(range(0,3600),-(stdv_y),linewidth=0.3)
    ax3.set_xlabel('epoch (s)(Time from beginning on test)')
    ax3.set_ylabel('standard deviation (m)')
    ax3.set_title('North standard deviation time series plot (m)')
    ax3.legend(['positive','negative'],loc='best')
    ax3=plt.subplot(2,3,6)
    ax3.plot(range(0,3600),stdv_z,linewidth=0.3)
    ax3.plot(range(0,3600),-(stdv_z),linewidth=0.3)
    ax3.set_xlabel('epoch (s)(Time from beginning on test)')
    ax3.set_ylabel('standard deviation (m)')
    ax3.set_title('Up standard deviation time series plot (m)')
    ax3.legend(['positive','negative'],loc='best')
    plt.show()

def plot_residuals(input):
    ax=plt.subplot(111)
    for i in range(0,12):
        ax.plot(range(0,len(input[i])),input[i],linewidth=0.3)
    plt.legend(['PRN 7','PRN 8','PRN 9','PRN 11','PRN 15','PRN 17','PRN 18','PRN 19','PRN 24','PRN 26','PRN 27','PRN 28'],loc='best')
    ax.set_xlabel('epoch (s)(Time from beginning on test)')
    ax.set_ylabel('residuals (m)')
    ax.set_title('residuals time series plot of all satellites')
    plt.show()

def elevation_angle():
    satfile=open_file("Satellites.sat",8)
    satfile=chunks(satfile,12)
    common=checkcommon()
    sat_eleva=[[],[],[],[],[],[],[],[],[],[],[],[]]
    epoch=0
    for i in satfile:
        for j in i:
            if j[0] in common[epoch]:
                xs=j[2]
                ys=j[3]
                zs=j[4]
                al,output,sr=pa.ecef2aer(xs,ys,zs,51.25864,-114.1005,1127.345)
                if j[0]==7.0:
                    sat_eleva[0].append(output)
                elif j[0]==8.0:
                    sat_eleva[1].append(output)
                elif j[0]==9.0:
                    sat_eleva[2].append(output)
                elif j[0]==11.0:
                    sat_eleva[3].append(output)
                elif j[0]==15.0:
                    sat_eleva[4].append(output)
                elif j[0]==17.0:
                    sat_eleva[5].append(output)
                elif j[0]==18.0:
                    sat_eleva[6].append(output)
                elif j[0]==19.0:
                    sat_eleva[7].append(output)
                elif j[0]==24.0:
                    sat_eleva[8].append(output)
                elif j[0]==26.0:
                    sat_eleva[9].append(output)
                elif j[0]==27.0:
                    sat_eleva[10].append(output)
                elif j[0]==28.0:
                    sat_eleva[11].append(output)
        epoch+=1
    return sat_eleva    

def main(): 
    result,stdv_x,stdv_y,stdv_z,sat_residuals,residuals=load_file()
    print(result[0])
    errorx,errory,errorz=cal_error(result)
    plot_error(errorx,errory,errorz,stdv_x,stdv_y,stdv_z)
    plot_residuals(sat_residuals)
    eleva=elevation_angle()
    ax=plt.subplot(111)
    for num in range(0,12):
        ax.scatter(eleva[num],sat_residuals[num],s=1)
    plt.legend(['PRN 7','PRN 8','PRN 9','PRN 11','PRN 15','PRN 17','PRN 18','PRN 19','PRN 24','PRN 26','PRN 27','PRN 28'],loc='best')
    ax.set_xlabel('elevation angle (degree)')
    ax.set_ylabel('residuals (m)')
    ax.set_title('residuals and elevation angle function')
    plt.show()
    
main()

