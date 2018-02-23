import numpy as np
import pandas as pd


def euclidean(x, y):
    """
    :param x: centroid point
    :param y: all the data points
    :return: euclidean distance between every point and centroid
    """
    return np.sqrt(np.sum((x - y) ** 2, axis=1))

def cosine(x,y):
    x_norm = np.sqrt(np.sum(x**2))
    y_norm = np.sqrt(np.sum(y**2, axis=1))
    return 1 - np.dot(x,y.T)/(x_norm*y_norm)

def nearest_centroid_err(data_np, centroid, metric):
    """
    :param data: numpy array of data
    :param centroid: numpy array of current centroids
    :param metric: which metric to use [1 : euclidean, 2: cosine]
    :return: clusters to which the points belong, sum of distance err
    """
    temp = []
    if metric == 1:
        for ele in centroid:
            temp.append(euclidean(ele, data_np))
    else:
        for ele in centroid:
            temp.append(cosine(ele, data_np))

    dist_mat = np.array(temp).T
    return np.argmin(dist_mat, axis=1), np.sum(np.min(dist_mat,axis=1))

def new_centroids(data_pd, indexes):
    """
    :param data_pd: pandas array of data
    :param indexes: cluster index of every data point
    :return: new centroids
    """
    # clust = []
    # for i in range(k):
    #     ind = [j for j in indexes if j==i]
    #     clust.append(np.mean(data_df.iloc[ind, :].values, axis=0))
    data_pd['clust']=indexes.tolist()
    centroids = data_pd.groupby('clust').mean()
    return np.array(centroids.values)


def k_means(data_np, data_pd, k, metric):
    """
    :param data_np: numpy data frame
    :param data_pd: pandas data frame
    :param k:  number of clusters
    :param metric: which metric to use [1 : euclidean, 2: cosine]
    :return: error, iterations and cluster index
    """
    # idx =
    curr_centroid = data_np[np.random.randint(data_np.shape[0],size=k),:]
    sse = delta = float('inf')
    count = 0
    while delta > 0.01:
        ind, temp_sse = nearest_centroid_err(data_np, curr_centroid, metric)
        curr_centroid = new_centroids(data_pd, ind)
        delta = abs(sse-temp_sse)
        sse = temp_sse
        count += 1
    return sse, count, ind

def accuracy(data_pd, clust_index):
    """
    :param data_pd: pandas actual cluster column
    :param clust_index: cluster index by k-means
    :return: accuracy value
    Needs improvement
    """
    corr = 0
    for i, item in enumerate(clust_index):
        if data_df.iloc[i,0] == item:
            corr += 1
    return corr/len(clust_index)


# file_name = 'C:/Users/onkar/Downloads/kmeans/Data_Cortex_Nuclear.csv'
# data_df = pd.read_csv(file_name)
# data_df = data_df.dropna()
# data_km = data_df.iloc[:, 1:78]


file_name = 'C:/Users/onkar/Downloads/kmeans/ionosphere.data.csv'
data_df = pd.read_csv(file_name)
data_df = data_df.dropna()
data_km = data_df.iloc[:, :33]

data_mat = data_km.values
k = 2

error, iterations, clust_index = k_means(data_mat, data_km, k, 1)
print(accuracy(data_df.iloc[:,34],clust_index))
print(error)





