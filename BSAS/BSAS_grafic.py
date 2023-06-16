import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def min_distance(clusters, point):
    distances = np.mean(np.abs(clusters - point), axis=1)
    min_dist = np.min(distances)
    closest_cluster = np.argmin(distances)
    return min_dist, closest_cluster

def BSAS(data, t_hold, maxNumClass):
    cluster = []
    cluster.append([data[0]])
    data = data[1:]
    while len(data) != 0:
        min_dist = float('inf')
        idx_cluster = -1
        for i, c in enumerate(cluster):
            dist, _ = min_distance(np.array(c), data[0])
            if dist < min_dist:
                min_dist = dist
                idx_cluster = i
        if min_dist > t_hold and len(cluster) < maxNumClass:
            cluster.append([list(data[0])])
            data = np.delete(data, 0, axis=0)
        else:
            cluster[idx_cluster].append(list(data[0]))
            data = np.delete(data, 0, axis=0)
    return cluster

# Carregar o conjunto de dados Iris
iris = load_iris()
data = iris.data

# Definir os parâmetros do BSAS
t_hold = 0.1  # Limiar de distância
maxNumClass = 6  # Número máximo de classes

# Aplicar o algoritmo BSAS
clusters = BSAS(data, t_hold, maxNumClass)

# Preparar os dados para plotagem
x = data[:, 0]
y = data[:, 1]
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']

# Plotar os pontos originais
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='black', label='Original Points')

# Plotar os clusters resultantes
for i, cluster in enumerate(clusters):
    cluster_points = np.array(cluster)
    cluster_x = cluster_points[:, 0]
    cluster_y = cluster_points[:, 1]
    plt.scatter(cluster_x, cluster_y, c=colors[i], label=f'Cluster {i+1}')

plt.title('Clustering with BSAS Algorithm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
