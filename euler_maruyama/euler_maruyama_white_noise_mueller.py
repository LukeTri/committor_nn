import numpy as np
from matplotlib import pyplot as plt
import torch

MUELLERMIN1 = np.array([0.62347076, 0.02807048])
MUELLERMIN2 = np.array([-0.55821361, 1.44174872])
mumins = np.zeros((2, 2))
mumins[0] = MUELLERMIN1
mumins[1] = MUELLERMIN2

a = np.array([-1, -1, -6.5, 0.7])
b = np.array([0, 0, 11, 0.6])
c = np.array([-10, -10, -6.5, 0.7])
d = np.array([-200, -100, -170, 15])
z=np.array([1, 0, -0.5, -1])
Y=np.array([0, 0.5, 1.5, 1])


def get_updated_gradient_offset_gaussian(x_0, updaters, omega=5, sigma=0.05):
    offset = np.array([0, 0])
    for q in range(len(updaters)):
        xn1 = updaters[q][0]
        xn2 = updaters[q][1]
        exp = omega * np.exp(-((x_0[0] - xn1) ** 2 + (x_0[1] - xn2) ** 2) / (2*sigma))
        offset[0] += exp * (-2 * x_0[0] + 2 * xn1) / (2*sigma)
        offset[1] += exp * (-2 * x_0[1] + 2 * xn2) / (2*sigma)
    return offset

def get_updated_offset_gaussian(x_0, updaters, omega=5, sigma=0.05):
    offset = np.zeros(len(x_0))
    x_0 = x_0.cpu().detach().numpy()
    for q in range(len(updaters)):
        xn1 = updaters[q][0]
        xn2 = updaters[q][1]
        exp = omega * np.exp(-((x_0[:,0] - xn1) ** 2 + (x_0[:,1] - xn2) ** 2) / (2*sigma))
        offset += exp
    return torch.tensor(offset)


def get_updated_gradient_offset_collective_var(x, c, d, z, k=1000):
    alpha = c[0] * x[0] + c[1] * x[1] - z
    x_offset = k * 2 * c[0] * alpha
    y_offset = k * 2 * c[1] * alpha

    return np.array([x_offset, y_offset])


def getNextIteration(x_0, h, offset_func="", updaters=np.array([]), b=1 / 20, omega=5, sigma=0.05, c=np.array([-2, 1]),
                     d=0, k=1000, z=np.array([0, 0])):
    xtemp = np.random.normal() * np.sqrt(2 * b ** -1 * h * np.sqrt(2))
    ytemp = np.random.normal() * np.sqrt(2 * b ** -1 * h * np.sqrt(2))
    offset = 0
    if offset_func == "metadynamics":
        offset = get_updated_gradient_offset_gaussian(x_0, updaters, omega, sigma)
    elif offset_func == "umbrella":
        offset = get_updated_gradient_offset_collective_var(x_0, c, d, z, k=k)

    return x_0 - (MuellerPotentialGradient(x_0) + offset) * h + np.array([xtemp, ytemp])


def get_next_iteration_vectorized(x_0, h, b=1 / 20):
    brownian = np.random.normal(size=(len(x_0), 2)) * np.sqrt(2 * b ** -1 * h * np.sqrt(2))
    return x_0 - mueller_grad_vectorized_numpy(x_0) * h + brownian


def MuellerPotentialNonVectorized(x, y):
    ret = 0
    for i in range(0, 4):
        ret += d[i] * np.e ** (a[i] * (x - z[i]) ** 2 + b[i] * (x - z[i]) * (y - Y[i]) + c[i] * (y - Y[i]) ** 2)
    return ret


def MuellerPotentialVectorized(x):
    ret = 0
    for i in range(0, 4):
        ret += d[i] * np.e ** (a[i] * (x[:,0] - z[i]) ** 2 + b[i] * (x[:,0] - z[i]) * (x[:,1] - Y[i]) + c[i] * (x[:,1] - Y[i]) ** 2)
    return ret


def plot_contours(x_start=-1.5, x_end=1.5, y_start=-0.5, y_end=2, n=100):
    v_func = np.vectorize(MuellerPotentialNonVectorized)  # major key!

    X, Y = np.meshgrid(np.linspace(x_start, x_end, n),
                       np.linspace(y_start, y_end, n))
    Z = v_func(X, Y)
    tics = np.linspace(-150, 250, 20)
    CS = plt.contour(X, Y, Z, tics, colors='grey', linewidth=5)


def mueller_grad_vectorized_torch(x):
    U_1 = torch.zeros(len(x))
    U_2 = torch.zeros(len(x))
    for i in range(4):
        V = d[i] * torch.e ** (
                a[i] * (x[:,0] - z[i]) ** 2 + b[i] * (x[:,0] - z[i]) * (x[:,1] - Y[i]) + c[i] * (x[:,1] - Y[i]) ** 2)
        U_1 += (2 * a[i] * (x[:,0] - z[i]) + b[i] * (x[:,1] - Y[i])) * V
        U_2 += (b[i] * (x[:,0] - z[i]) + 2 * c[i] * (x[:,1] - Y[i])) * V
    return torch.stack([U_1, U_2],dim=1)

def mueller_grad_vectorized_numpy(x):
    U_1 = np.zeros(len(x))
    U_2 = np.zeros(len(x))
    for i in range(4):
        V = d[i] * torch.e ** (
                a[i] * (x[:,0] - z[i]) ** 2 + b[i] * (x[:,0] - z[i]) * (x[:,1] - Y[i]) + c[i] * (x[:,1] - Y[i]) ** 2)
        U_1 += (2 * a[i] * (x[:,0] - z[i]) + b[i] * (x[:,1] - Y[i])) * V
        U_2 += (b[i] * (x[:,0] - z[i]) + 2 * c[i] * (x[:,1] - Y[i])) * V
    return np.vstack([U_1, U_2]).T


def MuellerPotentialGradient(x):
    U_1 = 0
    U_2 = 0
    for i in range(4):
        V = d[i] * np.e ** (
                a[i] * (x[0] - z[i]) ** 2 + b[i] * (x[0] - z[i]) * (x[1] - Y[i]) + c[i] * (x[1] - Y[i]) ** 2)
        U_1 += (2 * a[i] * (x[0] - z[i]) + b[i] * (x[1] - Y[i])) * V
        U_2 += (b[i] * (x[0] - z[i]) + 2 * c[i] * (x[1] - Y[i])) * V
    return np.array([U_1, U_2])