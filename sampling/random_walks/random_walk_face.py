import time

from euler_maruyama import euler_maruyama_white_noise_face as fp
import autograd.numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
from tqdm import tqdm
import csv

updaters = []
fig, ax = plt.subplots()


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_bounding_area(alpha_shape):
    bound_x, bound_y = alpha_shape.exterior.coords.xy
    return PolyArea(bound_x, bound_y)



def get_updated_offset(x, y, updaters, omega=5, sigma=0.05):
    offset = 0
    for q in range(len(updaters)):
        xn1 = updaters[q][0]
        xn2 = updaters[q][1]
        offset += omega * np.exp(-((x - xn1)**2 + (y - xn2)**2)/sigma)
    return offset


def createGraph(x, h, n, plot_row, plot_col, update_step_size=1000, gaussian=True, sigma=0.05, omega=5, b = 1/30):
    start = time.time()
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in tqdm(range(n)):
        # if (i % update_step_size) == update_step_size-1 and gaussian:
        #     updaters.append(x)
        x = fp.getNextIteration(x, h, updaters=updaters, sigma=sigma, omega=omega, b=b)
        X[i] = x[0]
        Y[i] = x[1]
    ax.scatter(X, Y)

    alpha_shape = alphashape.alphashape(np.array([*zip(X, Y)]), 0)

    # z1, f1, evaluations, h = gd(f, g, z_0, step_function, state)

    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.4))

    end = time.time()
    print(end - start)
    return X,Y

fp.plot_contours()

X, Y = createGraph(np.array([0, 0]), 10 * 10 ** -5, 1000000, 0, 0, omega=5,b=1/3)

header = ['X', 'Y']
data = np.vstack((X,Y)).T
data = np.ndarray.tolist(data)
print(updaters)
print(np.shape(updaters))

with open('/data/samples/face_standard=0.33_n=1000000.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
    writer.writerow("S")
    writer.writerows(updaters)

ax.title.set_text('omega=5,time_step=1000')

plt.show()
