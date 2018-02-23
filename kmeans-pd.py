
import pandas as pd

def euclid_dist(x,y):
    return (sum((x-y)**2))**0.5

def nearest_centriod(var_x,centroid_df):
    dist_list = []
    for item in centroid_df.values:
        dist_list.append(euclid_dist(var_x,item))
    return (dist_list.index(min(dist_list)),min(dist_list))


def k_means(df, k):
    pos_ctrd = df.iloc[:k, :]
    sse = delta = float('inf')
    count = 0

    while delta > 0.01:
        clust = []
        temp_sse = 0
        for item in df.values:
            val, err = nearest_centriod(item, pos_ctrd)
            clust.append(val)
            temp_sse += err
        delta = abs(sse - temp_sse)
        sse = temp_sse
        pos_ctrd = pd.concat([df, pd.DataFrame(clust, columns=['clt_id'])], axis=1).groupby('clt_id').mean()
        count += 1

    clust_id_df = pd.DataFrame(clust, columns=['clt_id'])

    return df, clust_id_df, sse, count

file_name = 'C:/Users/onkar/Downloads/kmeans/Data_Cortex_Nuclear.csv'
df=pd.read_csv(file_name)
df_km = df.iloc[:,1:78]
df_km=df_km.dropna()
df_km.reset_index(drop=True,inplace=True)

df_ret,clut_id,error,iterations = k_means(df_km,10)
print(error,iterations)