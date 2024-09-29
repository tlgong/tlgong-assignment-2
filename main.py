import numpy as np
import matplotlib.pyplot as plt
import time
def generate_dataset(num):
    x= np.random.uniform(-10, 10, size=num)
    y = np.random.uniform(-10, 10, size=num)
    coordinates= np.column_stack((x, y))
    # print(coordinates)
    return coordinates

def initialize_centroids_random(K):
    return np.random.uniform(-10, 10, size=(K, 2))


def initialize_centroids_farthest_first(data, K):
    N = data.shape[0]
    centroids = np.zeros((K, data.shape[1]))
    first_index = np.random.choice(N)
    centroids[0] = data[first_index]
    distances = np.linalg.norm(data - centroids[:1], axis=1)
    for i in range(1, K):
        if i>1:
            distances = np.column_stack((distances, np.linalg.norm(data - centroids[i-1:i], axis=1)))
        if i == 1:
            farthest_index = np.argmax(distances)
            centroids[i] = data[farthest_index]
        else:
            temp_distances = np.amin(distances,axis=1)
            farthest_index = np.argmax(temp_distances)
            centroids[i] = data[farthest_index]
    return centroids

def initialize_centroids_kmeans_pp(data, K):
    N, D = data.shape
    centroids = np.empty((K, D))
    first_index = np.random.choice(N)
    centroids[0] = data[first_index]
    distances = np.linalg.norm(data - centroids[0], axis=1) ** 2
    for i in range(1, K):
        prob_distribution = distances / np.sum(distances)
        next_index = np.random.choice(N, p=prob_distribution)
        centroids[i] = data[next_index]
        new_distances = np.linalg.norm(data - centroids[i], axis=1) ** 2
        distances = np.minimum(distances, new_distances)
    return centroids

def K_Mean_byStep(centroid,coordinates,tolerance = 1e-8):
    labels = np.zeros((coordinates.shape[0]))
    for i, point_a in enumerate(coordinates):
        label = 0
        temp_min = 100000000000
        for j, point_b in enumerate(centroid):
            distance = np.linalg.norm(point_a - point_b)
            if distance<temp_min:
                label = j
                temp_min = distance
        labels[i] = label
    new_centroid = np.zeros((centroid.shape[0],2))
    cnt = [0]*coordinates.shape[0]
    for i, point in (enumerate(coordinates)):
        index = int(labels[i])
        new_centroid[index][0] = new_centroid[index][0]*cnt[index]/(cnt[index]+1.0) + point[0]/(cnt[index]+1.0)
        new_centroid[index][1] = new_centroid[index][1] * cnt[index] / (cnt[index] + 1.0) + point[1] / (cnt[index] + 1.0)
        cnt[index] += 1
    centroid_shifts = np.linalg.norm(new_centroid - centroid, axis=1)
    max_shift = np.max(centroid_shifts)
    return new_centroid,np.array(labels),max_shift<tolerance

def K_Mean_toConverge(centroid,coordinates,iteration = 1,tolerance = 1e-8):
    last_centroid = centroid.copy()
    while True:
        new_centroid,labels,converge = K_Mean_byStep(last_centroid,coordinates)
        plot_kmeans_iteration(coordinates,new_centroid,labels,iteration)
        iteration+=1
        if converge:
            break
        last_centroid = new_centroid
    return last_centroid,labels

def plot_kmeans_iteration(data, centroids, labels, iteration):
    plt.clf()
    K = centroids.shape[0]
    colors = plt.cm.get_cmap('viridis', K)
    for k in range(K):
        cluster_points = data[labels == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    s=30, color=colors(k), label=f'Cluster {k + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1],
                s=200, color='red', marker='X', label='Centroids')
    plt.title(f'K-Means Clustering - Iteration {iteration}')
    plt.legend()
    plt.pause(0.5)

x = generate_dataset(500)
plt.figure(figsize=(8, 6))
centroids = initialize_centroids_kmeans_pp(x,10)
centroids, labels,converge = K_Mean_byStep(centroids,x)
plot_kmeans_iteration(x,centroids,labels, 1)
time.sleep(5)
K_Mean_toConverge(centroids,x,2)
plt.show()