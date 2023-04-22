import numpy as np
import math
"""
@author： Jiatong Niu
"""
## Calculate the degree of freedom and degree of constrain and update information in ENL
def assign_BCs(NL, ENL):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

    DOFs = 0
    DOCs = 0

    for i in range(0, NoN):
        for j in range(0, PD):

            if ENL[i, PD+j] == -1:
                DOCs = DOCs - 1
                ENL[i, 2 * PD + j] = DOCs
            else:
                DOFs = DOFs + 1
                ENL[i, 2 * PD + j] = DOFs

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, 2 * PD + j] < 0:
                ENL[i, 3 * PD + j] = abs(ENL[i, 2 * PD + j]) + DOFs
            else:
                ENL[i, 3 * PD + j] = abs(ENL[i, 2 * PD + j])

    DOCs = abs(DOCs)
    return (ENL, DOFs, DOCs)

# Calculate the stiffness , the imeplementation of this part refers to an online FEM tutorial but all hand written by author
def stiffness(ENL, Element_list, Node_list, E, A):
    NOE = np.size(Element_list, 0)
    NPE = np.size(Element_list, 1)
    PD = np.size(Node_list, 1)
    NoN = np.size(Node_list, 0)

    K = np.zeros([NoN * PD, NoN * PD])
    for i in range(0, NOE):
        nl = Element_list[i, 0:NPE]
        k = element_stiffness(nl, ENL, E, A)
        for r in range(0,NPE):
            for p in range (0,PD):
                for q in range(0, NPE):
                    for s in range(0, PD):
                        row = ENL[nl[r]-1,p+3*PD]
                        column = ENL[nl[q]-1,s+3*PD]
                        value = k[r*PD+p,q*PD+s]
                        K[int(row)-1,int(column)-1] =  K[int(row)-1,int(column)-1] + value
    return K

def element_stiffness(nl, ENL, E, A):
    X1 = ENL[nl[0] - 1, 0]
    Y1 = ENL[nl[0] - 1, 1]
    X2 = ENL[nl[1] - 1, 0]
    Y2 = ENL[nl[1] - 1, 1]

    L = math.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2)
    c = (X2 - X1) / L
    s = (Y2 - Y1) / L

    k = (E * A) / L * np.array([[c ** 2, c * s, -c ** 2, -c * s],
                               [c * s, s ** 2, -c * s, -s ** 2],
                               [-c ** 2, -c * s, c ** 2, c * s],
                               [-c * s, -s ** 2, c * s, s ** 2]])
    return k
#############################################################################

def calculate_force(ENL,Node_list):
    PD = np.size(Node_list, 1)
    NoN = np.size(Node_list, 0)
    DOF = 0
    Fp = []

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == 1:
                DOF += 1
                Fp.append(ENL[i,5*PD+j])
    Fp = np.vstack([Fp]).reshape(-1,1)
    return Fp

def calculate_dis(ENL,Node_list):
    PD = np.size(Node_list, 1)
    NoN = np.size(Node_list, 0)
    DOC = 0
    Up = []

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == -1:
                DOC = DOC + 1
                Up.append(ENL[i,4*PD+j])
    Up = np.vstack([Up]).reshape(-1,1)
    return Up

def update(ENL,U_u,Node_list,Fu):
    PD = np.size(Node_list, 1)
    NoN = np.size(Node_list, 0)

    DOFs = 0
    DOCs = 0
    for i in range(0, NoN):
        for j in range(0, PD):

            if ENL[i, PD+j] == 1:
                DOFs = DOFs + 1
                ENL[i, 4 * PD + j] = U_u[DOFs-1]
            else:
                DOCs = DOCs + 1
                ENL[i, 5 * PD + j] = Fu[DOCs-1]
    return ENL
