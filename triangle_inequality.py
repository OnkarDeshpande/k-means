

import pandas as pd
import numpy as np

def euclidean(x, y):
    """
    :param x: centroid point
    :param y: all the data points
    :return: euclidean distance between every point and centroid
    """
    return np.sqrt(np.sum((x - y) ** 2))#, axis=1))


# file_name = 'C:/Users/onkar/Downloads/kmeans/Data_Cortex_Nuclear.csv'
# data_df = pd.read_csv(file_name)
# data_df = data_df.dropna()
# data_km = data_df.iloc[:, 1:78]


# file_name = 'C:/Users/onkar/Downloads/kmeans/ionosphere.data.csv'
# data_df = pd.read_csv(file_name)
# data_df = data_df.dropna()
# data_km = data_df.iloc[:, :33]
# data_mat = data_km.values
# k = 3

data_mat = np.random.uniform(low=0, high=1, size=(10000,1001))
data_km = pd.DataFrame(data_mat, columns=None)
k = 3


low_bound = np.zeros(shape = (data_km.shape[0],k))
high_bound = np.array([float('inf')]*data_km.shape[0])

# print(high_bound)

centroid_pts = data_mat[np.random.randint(data_km.shape[0],size=k),:]

curr_centroid = [0]*data_km.shape[0]

old_centroid = [[0]*data_km.shape[1]]*k
count = 0


##condition that the new centroid is not that different than the old one
while not all(np.allclose(old_clust,new_clust) for old_clust,new_clust in zip(old_centroid,centroid_pts)):
    count += 1
    for point in data_km.index:
        for cluster in range(k):
            ##checking if the distance needs to be calculated (conditon 3)
            if cluster == curr_centroid[point] or \
                high_bound[point] < euclidean(centroid_pts[curr_centroid[point]], centroid_pts[cluster])/2.0 or \
                high_bound[point] <= low_bound[point][cluster] :
                continue

            ##condition 3b
            low_bound[point][cluster] = euclidean(data_mat[point,:], centroid_pts[cluster])


            if high_bound[point] < low_bound[point][cluster]:
                continue

            if high_bound[point] != low_bound[point][curr_centroid[point]]:
                low_bound[point][curr_centroid[point]] = euclidean(data_mat[point,:], centroid_pts[curr_centroid[point]])
                high_bound[point] = low_bound[point][curr_centroid[point]]

            ## condition 3b
            if high_bound[point] > low_bound[point][cluster]:
                curr_centroid[point] = cluster
                high_bound[point] = low_bound[point][cluster]

    ##condition 7
    old_centroid = centroid_pts[:]
    ##for calculating new centroids ... here check if the condition of number of diff_clust < k needs to be checked
    ##condition 4
    data_km['clust'] = curr_centroid
    diff_clust = len(set(curr_centroid))
    # if k != diff_clust:
    #     temp = np.array(data_km.groupby(['clust']).mean().values)
    #     for i,item in enumerate(temp):
    #         centroid_pts[i] = item
    # else:
    centroid_pts = np.array(data_km.groupby(['clust']).mean().values)

    centroid_change = [0]*diff_clust
    ##finding the change in centroid
    for i in range(diff_clust):
        centroid_change[i] = euclidean(old_centroid[i],centroid_pts[i])

    for item in range(data_km.shape[0]):
        ##condition 6
        high_bound[item] += centroid_change[curr_centroid[item]]
        ##condition 5
        for i in range(diff_clust):
            low_bound[item][i] = max(0, low_bound[item][i] - centroid_change[i])

    print(count)
    # break

print(curr_centroid)
err = 0
for i,item in enumerate(curr_centroid):
    err+=euclidean(data_mat[i],centroid_pts[item])
print(err)