import numpy as np
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
t_hold = 0.5  # Limiar de distância
maxNumClass = 3  # Número máximo de classes

# Aplicar o algoritmo BSAS
clusters = BSAS(data, t_hold, maxNumClass)

# Imprimir os clusters resultantes
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}:")
    for point in cluster:
        print(point)
    print()
