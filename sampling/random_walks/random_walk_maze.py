import time

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
from descartes import PolygonPatch
import alphashape
from tqdm import tqdm
import csv

DIRECTORY_PATH = os.path.join(os.getcwd(), "data")
MAZE_PATH = '/mazes/maze_data/maze8by8.csv'
OUTPUT_PATH = '/samples/maze8by8_b=1_n=1000000_factor=40.csv'

factor = 40


def maze_val(x, y):
    row_index = int(np.floor(y * len(vals)))
    column_index = int(np.floor(x * len(vals[0])))
    if row_index < 0 or row_index >= len(vals) - 1 or column_index < 0 or column_index >= len(vals[0]) - 1:
        return 1000
    return vals[row_index][column_index]


def plot_contours(x_start=-2.5, x_end=1.5, y_start=-1, y_end=2.5):
    x = np.linspace(0, 0.9999, len(vals[0]))
    y = np.linspace(0, 0.9999, len(vals))
    x_grid, y_grid = np.meshgrid(y, x)
    z = np.zeros((len(x), len(y)))
    z2 = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j] = maze_val(x_grid[i][j], y_grid[i][j])
            z2[i][j] = np.linalg.norm(maze_gradient(x_grid[i][j], y_grid[i][j]))
    plt.pcolormesh(x_grid, y_grid, z, shading='gourad')
    plt.show()
    plt.pcolormesh(x_grid, y_grid, z2, shading='gourad')


def maze_gradient(x, y):
    row_index = int(np.floor(y * len(vals)))
    column_index = int(np.floor(x * len(vals[0])))
    if row_index < 0:
        return np.array([0, np.abs(np.max(grads))])
    elif row_index >= len(vals) - 1:
        return np.array([0, -np.abs(np.max(grads))])
    elif column_index < 0:
        return np.array([np.abs(np.max(grads)), 0])
    elif column_index >= len(vals[0]) - 1:
        return np.array([-np.abs(np.max(grads)), 0])
    return np.array([grad_x[row_index][column_index], grad_y[row_index][column_index]])


def maze_grad_torch(x, grads):
    row_index = int(np.floor(x[1] * len(vals)))
    column_index = int(np.floor(x[0] * len(vals[0])))
    if row_index < 0:
        return np.array([0, np.abs(np.max(grads))])
    elif row_index >= len(vals) - 1:
        return np.array([0, -np.abs(np.max(grads))])
    elif column_index < 0:
        return np.array([np.abs(np.max(grads)), 0])
    elif column_index >= len(vals[0]) - 1:
        return np.array([-np.abs(np.max(grads)), 0])
    return np.array([grad_x[row_index][column_index], grad_y[row_index][column_index]])

def get_next_iteration(x_0, h, b):
    xtemp = np.random.normal() * np.sqrt(2 * b ** -1 * h * np.sqrt(2))
    ytemp = np.random.normal() * np.sqrt(2 * b ** -1 * h * np.sqrt(2))
    return x_0 + maze_gradient(x_0[0], x_0[1]) * h + np.array([xtemp, ytemp])


def createGraph(x, h, n, b):
    start = time.time()
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in tqdm(range(n)):
        x = get_next_iteration(x, h, b)
        X[i] = x[0]
        Y[i] = x[1]
    plot_contours()
    plt.scatter(X, Y)

    plt.show()

    alpha_shape = alphashape.alphashape(np.array([*zip(X, Y)]), 0)
    # ax.add_patch(PolygonPatch(alpha_shape, alpha=0.4))

    end = time.time()
    print(end - start)
    return X,Y


file = open(DIRECTORY_PATH + MAZE_PATH)
csvreader = csv.reader(file)
rows = []
rows2 = []
cutoff = False
row_len = 0
for row in csvreader:
    if len(row) == row_len or row_len == 0:
        rows.append(row)
        row_len = len(row)
    else:
        rows2.append(row)
file.close()

vals = np.zeros((len(rows), len(rows[0])))
for i in range(len(rows)):
    for j in range(len(rows[i])):
        vals[i][j] = float(rows[i][j])

grads = np.zeros((len(rows2), len(rows2[0])))
for i in range(len(rows2)):
    for j in range(len(rows2[i])):
        grads[i][j] = float(rows2[i][j])

grads = grads * factor
grad_x = grads[:len(grads)//2]
grad_y = grads[len(grads)//2:]


point1 = (0,0)
point2 = (0,0.05)
point3 = (0, 0.1)
point4 = (0.05, 0)
point5 = (0.1, 0)

print(maze_val(point1[0], point1[1]))
print(maze_val(point2[0], point2[1]))
print(maze_val(point3[0], point3[1]))
print(maze_val(point4[0], point4[1]))
print(maze_val(point5[0], point5[1]))

print(maze_gradient(point1[0], point1[1]))
print(maze_gradient(point2[0], point2[1]))
print(maze_gradient(point3[0], point3[1]))
print(maze_gradient(point4[0], point4[1]))
print(maze_gradient(point5[0], point5[1]))



plot_contours(x_start=-2.5, x_end=1.5, y_start=-1, y_end=2.5)
plt.show()

X, Y = createGraph(np.array([0.43, 0.61]), 10 ** -5, 1000000, 1)

header = ['X', 'Y']
data = np.vstack((X,Y)).T
data = np.ndarray.tolist(data)

with open(DIRECTORY_PATH + OUTPUT_PATH, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
    # writer.writerow("S")
    # writer.writerows(updaters)
