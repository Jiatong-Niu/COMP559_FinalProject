"""
@author： Jiatong Niu
"""
import numpy as np
from FEM_functions import *
import matplotlib.pyplot as plt
import time

from tkinter import *
def fix_check(num):
    if num == -1:
        return 'Yes'
    else:
        return 'No'

Node_list = np.array([[0, 0], [1, 0], [0.5, 1]])
Element_list = np.array([[1, 2], [2, 3], [3, 1]])

# assume first node is fixed in all direction
Boundary = np.array([[-1, -1], [1, -1], [1, 1]])
Fext = np.array([[0, 0], [0, 0], [0, -20]])
U_u = np.array([[0, 0], [0, 0], [0, 0]])
E = 10**6
A = 0.01

PD = np.size(Node_list,1)
NoN = np.size(Node_list,0)
ENL = np.zeros([NoN,6*PD])

ENL[:,0:PD] = Node_list[:,:]
ENL[:,PD:2*PD] = Boundary[:,:]

(ENL,DOFs,DOCs) = assign_BCs(Node_list,ENL)
K = stiffness(ENL,Element_list,Node_list,E,A)

ENL[:,4*PD:5*PD] = U_u[:,:]
ENL[:,5*PD:6*PD] = Fext[:,:]

U_u = U_u.flatten()
Fext = Fext.flatten()

Up = calculate_dis(ENL,Node_list)
Fp = calculate_force(ENL,Node_list)

K_uu = K[0:DOFs,0:DOFs]
K_up = K[0:DOFs,DOFs:DOFs+DOCs]
K_pu = K[DOFs:DOFs+DOCs,0:DOFs]
K_pp = K[DOFs:DOFs+DOCs,DOFs:DOFs+DOCs]

F = Fp - np.matmul(K_up,Up)
U_u = np.matmul(np.linalg.inv(K_uu),F)

Fu = np.matmul(K_pu,U_u)+np.matmul(K_pp,Up)

ENL = update(ENL,U_u,Node_list,Fu)
#print(ENL)

for scale in range(0,20):
    scale = 50*scale # exaggeration
    coor = []
    dispx_array = []
    dispy_array = []

    for i in range (np.size(Node_list,0)):
        dispx = ENL[i,8]
        dispy = ENL[i,9]

        x = ENL[i,0] + dispx *scale
        y = ENL[i,1] + dispy *scale

        dispx_array.append(dispx)
        dispy_array.append(dispy)
        coor.append(np.array([x,y]))

    coor = np.vstack(coor)
    dispx_array = np.vstack(dispx_array)


    x_scatter = []
    y_scatter = []

    color_x = []

    for i in range (0,np.size(Element_list,0)):
        x1 = coor[Element_list[i,0]-1,0]
        x2 = coor[Element_list[i,1]-1,0]
        y1 = coor[Element_list[i,0]-1,1]
        y2 = coor[Element_list[i,1]-1,1]

        dispx_EL = np.array([dispx_array[Element_list[i,0]-1],dispx_array[Element_list[i,1]-1]])

        if x1 == x2 :
            x = np.linspace(x1,x2,200)
            y = np.linspace(y1,y2,200)
        else:
            slope = (y2-y1)/(x2-x1)
            x = np.linspace(x1, x2, 200)
            y = slope*(x-x1)+y1

        x_scatter.append(x)
        y_scatter.append(y)

        color_x.append(np.linspace(np.abs(dispx_EL[0]),np.abs(dispx_EL[1]),200))


    x_scatter = np.vstack([x_scatter]).flatten()
    y_scatter = np.vstack([y_scatter]).flatten()
    color_x = np.vstack([color_x]).flatten()


    dispFigure = plt.figure(1)
    ax_dispx = dispFigure.add_subplot(111)
    cmap = plt.get_cmap('jet')
    ax_dispx.scatter(x_scatter,y_scatter,c=color_x,cmap = cmap,s=10,edgecolor = 'none')

    plt.xlim([0, 2])
    plt.ylim([-1,1])
    plt.show()

time.sleep(5)
window_r = Tk()
# basic setting for the window
window_r.title("FEM Interface for Final Project")
window_r.geometry("1280x980")

widget_1 = Label(window_r, text="position of Node 1")
widget_1.grid (row=0,column=0)
entry1 =Entry(window_r, width=20,font=("Verdana",10),justify="center",state = "readonly")
entry1.configure(state="normal")
entry1.insert(0,Node_list[0])
entry1.configure(state = "readonly")
entry1.grid(row=0,column=1)
widget_4= Label(window_r, text="New position of Node 1")
widget_4.grid (row=0,column=2)
entry4 =Entry(window_r, width=20,font=("Verdana",10),justify="center",state = "readonly")
entry4.configure(state="normal")
entry4.insert(0,coor[0])
entry4.configure(state = "readonly")
entry4.grid(row=0,column=3)
widget_7= Label(window_r, text="Is position fixed X/Y")
widget_7.grid (row=0,column=4)
entry7 =Entry(window_r, width=10,font=("Verdana",10),state = "readonly")
entry7.configure(state="normal")
entry7.insert(0,(fix_check(Boundary[0][0]),fix_check(Boundary[0][1])))
entry7.configure(state = "readonly")
entry7.grid(row=0,column=5)


widget_2 = Label(window_r, text="position of Node 2")
widget_2.grid (row=1,column=0)
entry2 =Entry(window_r, width=20,font=("Verdana",10),justify="center",state = "readonly")
entry2.configure(state="normal")
entry2.insert(0,Node_list[1])
entry2.configure(state = "readonly")
entry2.grid(row=1,column=1)
widget_5= Label(window_r, text="New position of Node 1")
widget_5.grid (row=1,column=2)
entry5 =Entry(window_r, width=20,font=("Verdana",10),justify="center",state = "readonly")
entry5.configure(state="normal")
entry5.insert(0,coor[1])
entry5.configure(state = "readonly")
entry5.grid(row=1,column=3)
widget_8= Label(window_r, text="Is position fixed X/Y")
widget_8.grid (row=1,column=4)
entry8 =Entry(window_r, width=10,font=("Verdana",10),state = "readonly")
entry8.configure(state="normal")
entry8.insert(0,(fix_check(Boundary[1][0]),fix_check(Boundary[1][1])))
entry8.configure(state = "readonly")
entry8.grid(row=1,column=5)


widget_3 = Label(window_r, text="position of Node 3")
widget_3.grid (row=2,column=0)
entry3 =Entry(window_r, width=20,font=("Verdana",10),justify="center",state = "readonly")
entry3.configure(state="normal")
entry3.insert(0,Node_list[2])
entry3.configure(state = "readonly")
entry3.grid(row=2,column=1)
widget_6= Label(window_r, text="New position of Node 1")
widget_6.grid (row=2,column=2)
entry6 =Entry(window_r, width=20,font=("Verdana",10),justify="center",state = "readonly")
entry6.configure(state="normal")
entry6.insert(0,coor[2])
entry6.configure(state = "readonly")
entry6.grid(row=2,column=3)
widget_8= Label(window_r, text="Is position fixed X/Y")
widget_8.grid (row=2,column=4)
entry8 =Entry(window_r, width=10,font=("Verdana",10),state = "readonly")
entry8.configure(state="normal")
entry8.insert(0,(fix_check(Boundary[2][0]),fix_check(Boundary[2][1])))
entry8.configure(state = "readonly")
entry8.grid(row=2,column=5)

# keep the window open
window_r.mainloop()