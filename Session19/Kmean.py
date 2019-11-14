import json

from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot
import pylab

filename = "image_export.json"

with open(filename, 'r') as f:
    datastore = json.load(f)

height_list = []
width_list = []
count_obj = 0
for i in datastore.values():
    #print(len(i[]))
    for j in range(len(i["regions"])):
        count_obj += 1;print("---------------------------------------",count_obj)
        # print(i["regions"][j]["shape_attributes"])#,type(i),len(i))
        c_dict = i["regions"][j]["shape_attributes"]
        # print('x',c_dict["x"])
        # print('y',c_dict["y"])
        # print('width',c_dict["width"])
        # print('height',c_dict["height"])
        # print('norm_width',c_dict["width"]/400)
        # print('norm_height',c_dict["height"]/400)
        width_list.append(c_dict["width"]/400)
        height_list.append(c_dict["height"]/400)
    # for j in i.values():
    #     print(j["shape_attributes"])
    # break
# print(datastore)
# print((height_list),(width_list)) 

X = np.column_stack((height_list, width_list))
# print(X)
# print(X[:,0])
matplotlib.pyplot.scatter(X[:,0],X[:,1])

matplotlib.pyplot.show()   
import matplotlib.pyplot as plt


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.show()
