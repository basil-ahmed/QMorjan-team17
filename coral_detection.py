import numpy as np
import cv2
from osgeo import gdal, gdal_array
import rasterio
from matplotlib import pyplot as plt
from rasterio.plot import show
from matplotlib.colors import ListedColormap

#Clustering packages
from sklearn import cluster
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


mosaic_path = '/4.tiff'
bathymetry_path = '/4.tiff'

# Define a fixed colormap for up to 10 clusters
cluster_colors = ['green', 'navy', 'coral', 'blue']
colormap = ListedColormap(cluster_colors)  
cluster_labels = ["Land", "Deep Water", "Coral Reefs", "Shallow Water"]

###Section 0: Utility functions

def elbow_method_Kmeans(data, cluster_range):
  wcss=[]
  for i_cluster in cluster_range:
      kmeans=cluster.KMeans(n_clusters=i_cluster, init='k-means++',random_state=0)
      kmeans.fit(data)
      wcss.append(kmeans.inertia_)
  return wcss


def bic_score_GMM(data,cluster_range, covariance_type):

  gmm_models=[GMM(n,covariance_type=covariance_type).fit(data) for n in cluster_range]
  bic_score=[i_model.bic(data) for i_model in gmm_models]
  return bic_score

def show_image(image, labels, cmap=None):
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap=cmap)
    
    # Create a legend with the custom labels and colors
    handles = [plt.Rectangle((0, 0), 1, 1, color=colormap.colors[i]) for i in range(len(cluster_labels))]
    plt.legend(handles, cluster_labels, loc='upper right', title="Clusters")
    
    plt.axis('off')

    # Save the image if a path is provided
    plt.savefig("/new_result.tiff", format='tiff', bbox_inches='tight')
    
    plt.show()

def show_plot(x,y,x_label,y_label,title):
  plt.plot(x,y)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.show()

def preprocess_for_clustering(mosaic):
  new_shape = (mosaic.shape[0] * mosaic.shape[1], mosaic.shape[2])
  mosaic_data = mosaic[:,:,:mosaic.shape[2]].reshape(new_shape)
  return mosaic_data

def show_cluster_output(clustering_labels, mosaic, colormap):
    clustering_labels_reshaped = clustering_labels.reshape(mosaic[:, :, 0].shape)
    with open("/content/out.txt", 'w') as file:
        for row in clustering_labels_reshaped:
            file.write(' '.join(map(str, row)) + '\n')
    show_image(clustering_labels_reshaped, cluster_labels, cmap=colormap)

    # Print the size of the picture (resolution)
    print("Picture size (resolution):", clustering_labels_reshaped.shape)

    # Print cluster colors with labels
    unique_labels = np.unique(clustering_labels)
    print("Cluster Labels and Colors:")
    for label in unique_labels:
        color = cluster_colors[label]
        descriptive_label = cluster_labels[label]
        print(f"Cluster {descriptive_label} (color: {color})")

###Section 1: Clustering methods

# 2.1 K-Means Clustering

def kmeans_mosaic(mosaic, no_of_clusters):
  mosaic_data = preprocess_for_clustering(mosaic)
  k_means = cluster.KMeans(n_clusters=no_of_clusters)
  k_means.fit(mosaic_data)
  clustering_labels = k_means.labels_
  return clustering_labels

# 2.2 Gaussian Mixture Model Clustering

def gmm_mosaic(mosaic, no_of_clusters, covariance_type):
    mosaic_data = preprocess_for_clustering(mosaic)
    gmm_model = GMM(n_components=no_of_clusters, covariance_type=covariance_type, random_state=42)
    gmm_model.fit(mosaic_data)
    clustering_labels = gmm_model.predict(mosaic_data)
    return clustering_labels

###Section 2: Loading the remote sensing data

# Loading the reef-mosaic as RGB

reef_mosaic_benthic = cv2.imread(mosaic_path)
reef_mosaic_benthic = cv2.cvtColor(reef_mosaic_benthic, cv2.COLOR_BGR2RGB)
plt.imshow(reef_mosaic_benthic)

# Loading the bathymetry data

bathymetry_raster = rasterio.open(bathymetry_path)
bathymetry_raster_visual= bathymetry_raster.read([1])
show(bathymetry_raster_visual)
bathymetry_raster = bathymetry_raster_visual.reshape(bathymetry_raster_visual.shape[1],bathymetry_raster_visual.shape[2],bathymetry_raster_visual.shape[0])
bathymetry_raster = np.uint16(bathymetry_raster)

# Concatenting bathymetric information with the reef mosaic for geomorphic mapping
bathymetry_raster = cv2.resize(bathymetry_raster, (885, 472))
bathymetry_raster = np.expand_dims(bathymetry_raster, axis=2)  # Add a channel dimension to bathymetry_raster
reef_mosaic_geomorphic = reef_mosaic_geomorphic[:1056, :, :]
reef_mosaic_geomorphic = np.concatenate((reef_mosaic_geomorphic, bathymetry_raster), axis=2)

###Section 3: Benthic maps  
number_of_cluster = 4

#3.1 Kmeans Result

kmeans_labels = kmeans_mosaic(reef_mosaic_benthic,4)
show_cluster_output(kmeans_labels,reef_mosaic_benthic,colormap)

#3.1 Gaussian mixture model clustering result

gmm_labels_benthic = gmm_mosaic(reef_mosaic_benthic,4,'full')
show_cluster_output(gmm_labels_benthic,reef_mosaic_benthic,colormap)