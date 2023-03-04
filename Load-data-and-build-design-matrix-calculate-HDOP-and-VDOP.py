import math
from winreg import QueryValueEx
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import rotate_matrix as rm
import navpy

def chunks(list:list,num:int):
    return [list[i:i+num] for i in range(0, len(list),num)]

def open_file(name:str,i:int):
    with open(name,'rb') as f:
        data=np.fromfile(f,count=-1,dtype="float64")
        data_reshape=chunks(data,i)
    return data_reshape

def plot_coordinates(input:list,epoch:int,prn:float):
    data=input
    x=[]
    y=[]
    z=[]
    for i in data[0:epoch*12]:
        if i[0]==prn:
            x.append(i[2])
            y.append(i[3])
            z.append(i[4])
    return x,y,z

def main1():
    file=open_file("Satellites.sat",8)
    prnlist=[7,8,11,15,17,18,19,24,26,27,28,0,9,22]
    satellites_x=[[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    satellites_y=[[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    satellites_z=[[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for j in range(0,14):
        xr,yr,zr=plot_coordinates(file,3600,prnlist[j])
        satellites_x[j].extend(xr)
        satellites_y[j].extend(yr)
        satellites_z[j].extend(zr)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
    for i in range(0,14):
        ax.scatter(satellites_x[i],satellites_y[i],satellites_z[i],marker='.')
    ax.set_title('Coordinates of satellites (all epochs)')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.legend(['PRN 7','PRN 8','PRN 11','PRN 15','PRN 17','PRN 18','PRN 19','PRN 24','PRN 26','PRN 27','PRN 28','PRN 9','PRN 22'],loc='best',bbox_to_anchor=(1.42, 1.0))
    plt.show()

def main2():
    file=open_file("RemoteL1L2.obs",6)
    prnlist=[7,8,11,15,17,18,19,24,26,27,28,0]
    psedorange=[[],[],[],[],[],[],[],[],[],[],[],[]]
    Doppler=[[],[],[],[],[],[],[],[],[],[],[],[]]
    L1_carrier_phase=[[],[],[],[],[],[],[],[],[],[],[],[]]
    for j in range(0,12):
        for i in file[0:3600]:
            if i[0]==prnlist[j]:
                psedorange[j].append(i[2])
                Doppler[j].append(i[4])
                L1_carrier_phase[j].append(i[3])
    for y in range(0,12):
        print(psedorange[y][0]-psedorange[y][299])
        print(L1_carrier_phase[y][0]-L1_carrier_phase[y][299])
        print(Doppler[y][0])
        fig=plt.figure()
        fig2=plt.figure()
        fig3=plt.figure()
        ax1=fig.add_subplot(111)
        ax2=fig3.add_subplot(111)
        ax3=fig2.add_subplot(111)
        ax1.scatter(range(0,300),psedorange[y],marker='.',linewidth=0.1)
        ax2.plot(range(0,300),Doppler[y],marker='.',color='red')
        ax3.plot(range(0,300),L1_carrier_phase[y],marker='.',color='green')
        ax1.set_xlabel('epoch(s)(Time from beginning on test)')
        ax1.set_ylabel('pseudorange (m)')
        ax1.set_title('psedorange of satellite'+str(prnlist[y]))
        ax2.set_xlabel('epoch(s)(Time from beginning on test)')
        ax2.set_ylabel('doppler (Hertz)')
        ax2.set_title('Doppler of satellite'+str(prnlist[y]))
        ax3.set_xlabel('epoch(s)(Time from beginning on test)')
        ax3.set_ylabel('L1 carrier phase (cycles)')
        ax3.set_title('L1_carrier_phase of satellite'+str(prnlist[y]))
        # plt.legend(['psedorange','Doppler','L1_carrier_phase'])
        plt.show()

def main3():
    file=open_file("Satellites.sat",8)
    file=chunks(file,12)
    num_satellites=[]
    x_receiver=51.25864
    y_receiver=-114.1005
    z_receiver=1127.345
    x,y,z=navpy.lla2ecef(x_receiver,y_receiver,z_receiver)
    R=[[-math.sin(y_receiver),math.cos(y_receiver),0,0],[-math.cos(y_receiver)*math.sin(x_receiver),-math.sin(y_receiver)*math.sin(x_receiver),math.cos(x_receiver),0],[math.cos(y_receiver)*math.cos(x_receiver),math.sin(y_receiver)*math.cos(x_receiver),math.sin(x_receiver),0],[0,0,0,1]]
    R=np.array(R)
    RT=np.transpose(R)
    HDOP=[]
    VDOP=[]
    for i in file: 
        H=[]
        nums=0
        for j in i:
            if j[0]!=0:
                nums+=1
                xs=j[2]
                ys=j[3]
                zs=j[4]
                P=math.sqrt((xs-x)**2+(ys-y)**2+(zs-z)**2)
                H.append([-(xs-x)/P,-(ys-y)/P,-(zs-z)/P,-1])
        H=np.array(H)
        HT=np.transpose(H)
        Qx=np.matmul(HT,H)
        Qx=np.linalg.inv(Qx)
        Qx=np.matmul(R,Qx)
        Qx=np.matmul(Qx,RT)
        num_satellites.append(nums)
        xx=Qx[0][0]
        yy=Qx[1][1]
        zz=Qx[2][2]
        hdop=math.sqrt(xx+yy)
        vhop=math.sqrt(zz)
        HDOP.append(hdop)
        VDOP.append(vhop)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(range(0,3600),VDOP,marker='.',linewidth=0.1)
    ax.set_xlabel('epoch(s)(Time from beginning on test)')
    ax.set_title('VDOP')
    plt.legend(['VDOP'])
    plt.show()

def checkduplicate():
    file=open_file("RemoteL1L2.obs",6)
    res=[]
    for i in file:
        res.append(i[0])
    result=[]
    for i in res:
        if i not in result:
            result.append(i)
    return result

def test():
    a=np.array([1,2],[3,4])
    a=np.array([1,2],[3,4])



