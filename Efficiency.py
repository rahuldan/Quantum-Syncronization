import numpy as np
from qutip import *
import matplotlib.pyplot as plt

N = 3                                                                           #Dimension of the Hilbert space
t_int = 50                                                                      #The states will be evaluated at these points of time
omega = 2 * np.pi                                                               #Frequency of the driving force
g = 0.5                                                                           #coupling strength

a1 = basis(N,0)
rho0 = ket2dm(a1)


def H1_coeff(t, args):                                                          #Coefficient of the interaction part of the Hamiltonian
    return g * np.sin(omega * t)

def efficiency(gamma1, gamma2, g):                                              #Calculates the efficiency given the parameters
    b = destroy(N)

    T = 2 * np.pi / omega                                                       #Time period of the driving force
    delta_t = T / t_int                                                         #Time interval between 2 successive time at which the states are evaluated
    tlist = np.linspace(0, T, t_int)                                            #Time at which the states are evaluated

    H0 = (b.dag() * b)
    H1 = b + b.dag()
    H = [H0, [H1, H1_coeff]]

    W = 0
    Q_in = 0
    c= [np.sqrt(gamma1) * b.dag(), np.sqrt(gamma2) * (b * b)]

    output = mesolve(H, rho0, tlist, c_ops = c, e_ops=[], args={}, options=None, progress_bar=None, _safe_mode=True)
    print(gamma2)

    for i in range(t_int - 1):
        A = output.states[i] * np.cos(omega * tlist[i]) * omega * H1 * g
        W = W + (A.tr() * delta_t)
        B = ((output.states[i + 1] - output.states[i]) / delta_t) * (H0 + (H1 * g * np.sin(omega * tlist[i])))
        Q = B.tr()
        if np.real(Q) > 0:
            Q_in = Q_in + (Q * delta_t)

    e = -np.real(W / Q_in)
    return e

e_val = np.empty(30)

gamma2 = 0
for j in range(5, 8):
    g = j
    for i in range(30):
        gamma1 = 0.1
        e_val[i] = efficiency(gamma1, gamma2, g)
        gamma2 += 1
    gamma2 = 0
    plt.plot(e_val)


plt.show()
