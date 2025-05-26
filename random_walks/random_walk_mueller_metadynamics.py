import time

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
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

def plot_countours():
    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                       np.linspace(-0.5, 2, 100))
    Z = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            print()
            Z[j][i] = get_updated_offset(X[0][i], Y[j][0], updaters)
    tics = np.linspace(-150, 150, 30)
    CS = plt.contour(X, Y, Z, tics)
    plt.clabel(CS, inline=False, fontsize=10)


def createGraph(x, h, n, plot_row, plot_col, update_step_size=1000, gaussian=True, sigma=0.05, omega=5, b = 1/30):
    start = time.time()
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in tqdm(range(n)):
        x = mp.getNextIteration(x, h, updaters=updaters, offset_func="metadynamics", sigma=sigma, omega=omega, b=b)
        X[i] = x[0]
        Y[i] = x[1]
    ax.scatter(X, Y)

    end = time.time()
    print(end - start)
    return X,Y

mp.plot_contours()

file = open('/data/samples/mueller_metadynamics_b=0.033_n=1000000.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)
rows = []
rows2 = []
cutoff = False
for row in csvreader:
    if row[0] == "S":
        cutoff = True
    elif not cutoff:
        rows.append(row)
    else:
        rows2.append(row)
file.close()

arr = np.zeros((len(rows), len(rows[0])))
for i in range(len(rows)):
    for j in range(len(rows[i])):
        arr[i][j] = float(rows[i][j])

if len(rows2) == 0:
    updaters = []
else:
    updaters = np.zeros((len(rows2), len(rows2[0])))
    for i in range(len(rows2)):
        for j in range(len(rows2[i])):
            updaters[i][j] = float(rows2[i][j])

X, Y = createGraph(np.array([0, 0]), 10 ** -5, 200000, 0, 0, omega=5,b=1/10)

header = ['X', 'Y']
data = np.vstack((X,Y)).T
data = np.ndarray.tolist(data)
print(updaters)
print(np.shape(updaters))

with open('/data/samples/mueller_metadynamics_b=0.1_n=200000_precomputed.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
    writer.writerow("S")
    writer.writerows(updaters)

ax.title.set_text('omega=5,time_step=1000')

plt.show()
