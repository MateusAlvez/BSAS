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

def on_click(event):
    if event.button == 1:
        x, y = event.xdata, event.ydata
        cluster_id = find_cluster(x, y)
        if cluster_id is not None:
            print(f"Selected Cluster {cluster_id}")
            plot_clusters(cluster_id)

def find_cluster(x, y):
    for i, cluster_points in enumerate(clusters):
        cluster_points = np.array(cluster_points)
        dist = np.sqrt((cluster_points[:, 0] - x) ** 2 + (cluster_points[:, 1] - y) ** 2)
        if np.any(dist < 0.1):  # Definir uma tolerância para a seleção do cluster
            return i
    return None

def plot_clusters(selected_cluster=None):
    plt.cla()  # Limpar o gráfico atual
    plt.scatter(x, y, c=target_colors, label='Original Points')
    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        cluster_x = cluster_points[:, 0]
        cluster_y = cluster_points[:, 1]
        cluster_color = colors[i % len(colors)]
        if selected_cluster is None or selected_cluster == i:
            plt.scatter(cluster_x, cluster_y, c=cluster_color, label=f'Cluster {i+1}')
    plt.title('Clustering with BSAS Algorithm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.draw()

# Carregar o conjunto de dados Iris
iris = load_iris()
data = iris.data
target = iris.target

# Definir os parâmetros do BSAS
t_hold = 0.5  # Limiar de distância
maxNumClass = 3  # Número máximo de classes

# Aplicar o algoritmo BSAS
clusters = BSAS(data, t_hold, maxNumClass)

# Preparar os dados para plotagem
x = data[:, 0]
y = data[:, 1]

# Criar uma lista de cores com base nos rótulos originais
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
target_names = iris.target_names
target_colors = [colors[i] for i in target]

# Plotar os clusters iniciais
plot_clusters()

# Adicionar interatividade
fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
