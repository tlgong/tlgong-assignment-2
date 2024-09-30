import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class KMeans:
    def __init__(self, K, init_method='random', tolerance=1e-8, max_iterations=300, random_state=None):
        self.K = K
        self.init_method = init_method
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.initial_centroids = None
        self.centroids = None
        self.labels = None
        self.history = []
        self.coordinates = None
        self.manual_centroids = None
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def change_number_of_cluster(self, k):
        self.K = k

    def generate_dataset(self, num=500):
        print(self)
        x = np.random.uniform(-10, 10, size=num)
        y = np.random.uniform(-10, 10, size=num)
        self.coordinates = np.column_stack((x, y))

    def initialize_centroids_random(self):
        centroids = np.random.uniform(-10, 10, size=(self.K, 2))
        self.initial_centroids = centroids.copy()
        self.centroids = centroids.copy()
        return centroids

    def initialize_centroids_farthest_first(self, data):
        N = data.shape[0]
        centroids = np.zeros((self.K, data.shape[1]))
        first_index = np.random.choice(N)
        centroids[0] = data[first_index]
        distances = np.linalg.norm(data - centroids[0], axis=1)
        for i in range(1, self.K):
            farthest_index = np.argmax(distances)
            centroids[i] = data[farthest_index]
            distances = np.minimum(distances, np.linalg.norm(data - centroids[i], axis=1))
        self.initial_centroids = centroids.copy()
        self.centroids = centroids.copy()
        return centroids

    def initialize_centroids_kmeans_pp(self, data):
        N, D = data.shape
        centroids = np.empty((self.K, D))
        first_index = np.random.choice(N)
        centroids[0] = data[first_index]
        distances = np.linalg.norm(data - centroids[0], axis=1) ** 2
        for i in range(1, self.K):
            prob_distribution = distances / np.sum(distances)
            next_index = np.random.choice(N, p=prob_distribution)
            centroids[i] = data[next_index]
            new_distances = np.linalg.norm(data - centroids[i], axis=1) ** 2
            distances = np.minimum(distances, new_distances)
        self.initial_centroids = centroids.copy()
        self.centroids = centroids.copy()
        return centroids

    def initialize_centroids_manual(self, centroids):
        centroids = np.array(centroids)
        if centroids.shape != (self.K, 2):
            raise ValueError(f"Expected {self.K} centroids, but got {centroids.shape[0]}.")
        self.initial_centroids = centroids.copy()
        self.centroids = centroids.copy()
        return centroids

    def initialize_centroids_method(self, data):
        if self.init_method == 'random':
            return self.initialize_centroids_random()
        elif self.init_method == 'farthest_first':
            return self.initialize_centroids_farthest_first(data)
        elif self.init_method == 'kmeans++':
            return self.initialize_centroids_kmeans_pp(data)
        elif self.init_method == 'manual':
            if self.manual_centroids is not None:
                return self.initialize_centroids_manual(self.manual_centroids)
            else:
                raise ValueError("Manual centroids have not been set.")
        else:
            raise ValueError("Invalid init_method. Choose from 'random', 'farthest_first', 'kmeans++', 'manual'.")

    def K_Mean_byStep(self, centroid, coordinates):
        labels = np.zeros(coordinates.shape[0])

        for i, point_a in enumerate(coordinates):
            label = 0
            temp_min = float('inf')
            for j, point_b in enumerate(centroid):
                distance = np.linalg.norm(point_a - point_b)
                if distance < temp_min:
                    label = j
                    temp_min = distance
            labels[i] = label

        new_centroid = np.zeros((centroid.shape[0], 2))
        cnt = [0] * centroid.shape[0]

        for i, point in enumerate(coordinates):
            index = int(labels[i])
            new_centroid[index][0] += point[0]
            new_centroid[index][1] += point[1]
            cnt[index] += 1

        for k in range(self.K):
            if cnt[k] > 0:
                new_centroid[k][0] /= cnt[k]
                new_centroid[k][1] /= cnt[k]
            else:
                new_centroid[k] = centroid[k]

        centroid_shifts = np.linalg.norm(new_centroid - centroid, axis=1)
        max_shift = np.max(centroid_shifts)
        converged = max_shift < self.tolerance
        self.centroids = new_centroid
        self.labels = labels
        return converged

    def get_plot_image(self, iteration):
        plt.figure(figsize=(8, 6))
        if self.centroids is not None:
            print(f"Plotting centroids: {self.centroids}")
            K = self.centroids.shape[0]
            colors = plt.cm.get_cmap('viridis', K)
            for k in range(K):
                cluster_points = self.coordinates[self.labels == k]
                if cluster_points.size > 0:
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                                s=30, color=colors(k), label=f'Cluster {k + 1}')
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                        s=200, color='red', marker='X', label='Centroids')
            plt.title(f'K-Means Clustering - Iteration {iteration}')
            plt.legend()
        else:
            print("Centroids not initialized. Plotting only data points.")
            plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1],
                        s=30, color='blue', label='Data Points')
            plt.title(f'K-Means Clustering - Iteration {iteration}')
            plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return plot_base64

    def fit_one_step(self):
        converged = self.K_Mean_byStep(self.centroids, self.coordinates)
        self.history.append(self.centroids.copy())
        plot_url = self.get_plot_image(len(self.history))
        return plot_url, converged

    def fit_until_converge(self):
        iteration = len(self.history)
        converged = False
        while iteration < self.max_iterations and not converged:
            plot_url, converged = self.fit_one_step()
            iteration += 1
        if converged:
            print(f"Convergence reached at iteration {iteration}.")
        else:
            print("Maximum iterations reached without convergence.")
        return plot_url, converged

    def reset_centroids(self):
        if self.initial_centroids is not None:
            self.centroids = self.initial_centroids.copy()
            distances = np.linalg.norm(self.coordinates[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
            self.labels = np.argmin(distances, axis=1)
            self.history = [self.centroids.copy()]
            plot_url = self.get_plot_image(0)
            print("Centroids have been reset to initial centroids.")
            return plot_url
        else:
            print("Initial centroids not set.")
            return None

    def run_initialization(self):
        if self.init_method != 'manual':
            self.centroids = self.initialize_centroids_method(self.coordinates)
            self.initial_centroids = self.centroids.copy()
            self.history = [self.centroids.copy()]
            distances = np.linalg.norm(self.coordinates[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
            self.labels = np.argmin(distances, axis=1)
            plot_url = self.get_plot_image(0)
            return plot_url
        else:
            if self.centroids is None and self.manual_centroids is not None:
                self.centroids = self.initialize_centroids_method(self.coordinates)
                self.initial_centroids = self.centroids.copy()
                self.history = [self.centroids.copy()]
                distances = np.linalg.norm(self.coordinates[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
                self.labels = np.argmin(distances, axis=1)
                plot_url = self.get_plot_image(len(self.history))
                return plot_url
            else:
                plot_url = self.get_plot_image(len(self.history))
                return plot_url

    def set_manual_centroids(self, centroids):
        print(self)
        self.manual_centroids = centroids
        self.centroids = self.initialize_centroids_manual(centroids)

        distances = np.linalg.norm(self.coordinates[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        self.labels = np.argmin(distances, axis=1)

        self.history = [self.centroids.copy()]
        print(f"Set centroids: {self.centroids}")
