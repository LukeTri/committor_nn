import numpy as np
from matplotlib import pyplot as plt
import torch

xa=-3
ya=3
xb=0
yb=4.5


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

    return x_0 - (face_potential_gradient(x_0) + offset) * h + np.array([xtemp, ytemp])


def getNextIteration(x_0, h, b=1/20):
    brownian_motion = np.random.normal(size=2) * np.sqrt(2 * b ** -1 * h * np.sqrt(2))
    return x_0 - (face_potential_gradient(x_0)) * h + brownian_motion


def face_non_vectorized(x, y):
    f=(1-x)**2+(y-0.25*x**2)**2+1
    g1=1-np.exp(-0.125*((x-xa)**2+(y-ya)**2))
    g2=1-np.exp(-0.25*(((x-xb)**2+(y-yb)**2)))
    g3=1.2-np.exp(-2*((x+0)**2+(y-2)**2))
    g4=1+np.exp(-2*(x+1.5)**2-(y-3.5)**2-(x+1)*(y-3.5))
    v = f*g1*g2*g3*g4
    return v


def face_non_vectorized_torch(x, y):
    f=(1-x)**2+(y-0.25*x**2)**2+1
    g1=1-torch.exp(-0.125*((x-xa)**2+(y-ya)**2))
    g2=1-torch.exp(-0.25*(((x-xb)**2+(y-yb)**2)))
    g3=1.2-torch.exp(-2*((x+0)**2+(y-2)**2))
    g4=1+torch.exp(-2*(x+1.5)**2-(y-3.5)**2-(x+1)*(y-3.5))
    v = f*g1*g2*g3*g4
    return v

def face_vectorized_torch(xy):
    x = xy[0]
    y = xy[1]
    f=(1-x)**2+(y-0.25*x**2)**2+1
    g1=1-torch.exp(-0.125*((x-xa)**2+(y-ya)**2))
    g2=1-torch.exp(-0.25*(((x-xb)**2+(y-yb)**2)))
    g3=1.2-torch.exp(-2*((x+0)**2+(y-2)**2))
    g4=1+torch.exp(-2*(x+1.5)**2-(y-3.5)**2-(x+1)*(y-3.5))
    v = f*g1*g2*g3*g4
    return v


def face_vectorized(xy):
    x = xy[:,0]
    y = xy[:,1]
    f=(1-x)**2+(y-0.25*x**2)**2+1
    g1=1-np.exp(-0.125*((x-xa)**2+(y-ya)**2))
    g2=1-np.exp(-0.25*(((x-xb)**2+(y-yb)**2)))
    g3=1.2-np.exp(-2*((x+0)**2+(y-2)**2))
    g4=1+np.exp(-2*(x+1.5)**2-(y-3.5)**2-(x+1)*(y-3.5))
    v = f*g1*g2*g3*g4
    return v
#\left(\left(1-x^2\right)+\left(y-0.25x^2\right)\:+\:1\right)

def plot_contours(x_start=-5, x_end=4, y_start=-3, y_end=7, n=100):
    v_func = np.vectorize(face_non_vectorized)  # major key!

    X, Y = np.meshgrid(np.linspace(x_start, x_end, n),
                       np.linspace(y_start, y_end, n))
    Z = v_func(X, Y)
    print(np.max(Z))
    print(np.min(Z))
    tics = np.linspace(0, 80, 30)
    CS = plt.contour(X, Y, Z, tics, colors='grey', linewidth=5)
    plt.contour(X, Y, Z, np.array([5]), colors='blue', linewidth=10)


def face_potential_gradient(xy):
    xy = torch.tensor(xy,dtype=torch.float,requires_grad=True)
    V = face_vectorized_torch(xy)
    grad = torch.autograd.grad(V,xy,allow_unused=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(V), create_graph=True)[0]
    return grad.detach().numpy()
