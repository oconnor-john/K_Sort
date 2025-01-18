import random
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

def load_numerical_data(filename: str, column_titles: list) -> dict:
    """Load data from a CSV file and return a dictionary with keys being the
    row number and values as tuples of the data in each row, converted to float.

    Args:
        filename: The name of the CSV file to load.
        column_titles: A list of columns to load.

    Returns:
        A dictionary where each element corresponds to a data point, with keys 
        corresponding to the row number and values as a tuple of floats.

    Example:
        If column_titles = ['Col1', 'Col3'], and the CSV file has the following data:
            Col1, Col2, Col3
             2.4,  5.6,  7.8
            10.0, 42.5, -3.2
            31.4,  0.5, 12.3
        Then the return dictionary will be:
            {0: (2.4, 7.8), 1: (10, -3.2), 2: (31.4, 12.3)}
    """

    dict = {}
    with open(filename, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        count = 0
        for row in reader:
            try:
                data = tuple(float(row[column]) for column in column_titles)
                dict[count] = data
                count += 1
            except:
                ValueError
        return dict

def euclid_dist(point1: tuple, point2: tuple) -> float:
    """Compute the Eucledian distance between two points represented as tuples.
    Listing 7.1 in PPC, with modifications for compliance to PEP8

    Args:
        point1: A tuple representing a point in n-dimensional space.
        point2: A tuple representing a point in n-dimensional space.

    Returns:
        float: The Euclidean distance between the two points.

    Example:
        euclid_dist((1, 2.5), (2.1, 4)) should return 1.86 (approximately).
    """
    
    total = 0
    for n in range(len(point1)):
        distance = ((point1[n] - point2[n]) ** 2)
        total += distance

    total_dist = math.sqrt(total)

    return total_dist


def create_centroids(k: int, data: dict) -> list:
    """Create k centroids by picking random points from the data until 
    you have k unique centroids.

    Args:
        k: The number of centroids to create.
        data: A dictionary where each element corresponds to a data point, with keys
            corresponding to the row number and values as tuples of floats.

    Returns:
        list: a list of centroids, each centroid is a tuple of floats.
    """

    centroids = []
    centroid_count = 0
    centroid_keys = []

    while centroid_count < k:
        key = random.randint(1, len(data))
        if key not in centroid_keys:
            centroids.append(data[key])
            centroid_keys.append(key)
            centroid_count += 1
    return centroids


def create_clusters(k: int, centroids: list, data: dict, repeats=100) -> tuple:
    """Create clusters using the k-means algorithm
    From Listing 7.8, modified to comply with PEP8
    Args:
        k: how many clusters to create
        centroids: the list of centroids, one per cluster
        values: list of tuples
        repeats: how many iterations to run

    Returns:
        list, list: two lists are returned -- one is a list of clusters and the 
            second one is the list of centroids
    """
    
    for n in range(repeats):
        print("****PASS", n + 1, "****")
        clusters = []
        for i in range(k):
            clusters.append([])
        
        for key in data:
            distances = []
            for cluster in centroids:
                dist_to_centroid = euclid_dist(data[key], cluster)
                distances.append(dist_to_centroid)
            
            min_dist = min(distances)
            indexes = distances.index(min_dist)

            clusters[indexes].append(data[key])

        dimensions = 2
        for cluster_idx in range(k):
            sums = [0] * 2
            for point in clusters[cluster_idx]:
                for ind in range(2):
                    sums[ind] += point[ind]

            for ind in range(len(sums)):
                cluster_length = len(clusters[cluster_idx])
                if cluster_length != 0:
                    sums[ind] = sums[ind] / cluster_length

            centroids[cluster_idx] = sums

    
    return clusters, centroids

def visualize_clusters(dataset_name: str, titles: list, clusters: list, centroids: list) -> plt.Figure:
    """
    Visualize the clusters and centroids. Use a different color for each cluster.
    
    Args: 
        dataset_name: The name of the dataset
        titles: list of string column titles
        clusters: list of lists of tuples
        centroids: list of tuples
    
    Returns:
        matplotlib.pyplot.Figure: The figure object
    """
   
   
    # Validate inputs
    if len(titles) < 2:
        raise ValueError("At least two column titles are needed for visualization.")

    # Extract the first two columns for visualization
    x_label, y_label = titles[:2]

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define a colormap
    cmap = get_cmap("tab10")

    # Plot each cluster
    for idx, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            label=f"Cluster {idx + 1}",
            color=cmap(idx),
            alpha=0.6
        )

    # Plot centroids
    centroids = np.array(centroids)
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        color="black", marker="x", s=100, label="Centroids"
    )

    # Set plot details
    ax.set_title(f"Cluster Visualization - {dataset_name}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    # Return the figure object
    plt.show()
    return fig


def main():
    """ Main driver for the program."""

    # Specifies the files and columns to analyze in the keys, and the number
    # of clusters in the values.
    datasets = {('earthquakes', ('latitude', 'longitude')): 5,
                ('earthquakes', ('depth', 'mag')): 5,
                ('cis210_scores', ('Projects', 'Exams')): 5}
    # Feel free to add more datasets or column pairs and experiment with different values of k

    # Compute clusters for all datasets
    for (dataset, titles), k in datasets.items():
        print(f'\nDataset: {dataset} {titles}')
        # Part 8.1
        data = load_numerical_data(dataset + '.csv', column_titles=titles)

        # Part 8.3
        centroids = create_centroids(k, data)
        print("Initialized the centroids.")

        # Parts 8.2 and 8.4 (create_clusters calls euclid_dist)
        clusters, centroids = create_clusters(k, centroids, data)
        print("\nCreated the clusters.")

        visualize_clusters(dataset, titles, clusters, centroids)
        print("Visualized the clusters.")


if __name__ == '__main__':
    main()