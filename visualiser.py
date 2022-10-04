from tabnanny import verbose
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score,adjusted_rand_score
from umap import UMAP
import os
from category_encoders import PolynomialEncoder
import pandas as pd
dirname = os.path.dirname(__file__)
C = 0
"""
Был обнаружен баг в частях кода, касающийся переменной cluster_labels, а именно , благодаря которой создавался tuple
"""

class Visualise():
    
    def __init__(self):
        pass

    def proto_vis(self,cluster,data,n_clusters=2):
        # encoder = PolynomialEncoder(cols=['Graduation'])
        # new_data = encoder.fit_transoform(data,verbose=1)

        X = StandardScaler().fit_transform(data)

        if cluster == 'KMeans':
            cl = KMeans(n_clusters=n_clusters,random_state=10)
            cluster_labels = cl.fit_predict(X),
            cluster_labels = cluster_labels[0] #поскольку Kmeans.fit_predict() делает тип данных tuple
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        if cluster == 'SpectralClustering':    
            cl = SpectralClustering(n_clusters=n_clusters,n_init=1) # работает чудовищно долго ,random_state=10
            cluster_labels = cl.fit_predict(X),
            cluster_labels = cluster_labels[0]
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        if cluster == 'AgglomerativeClustering':   
            cl = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = cl.fit_predict(X)
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        if cluster =='DBSCAN':
            cl = DBSCAN(eps = 10, min_samples =7)
            cluster_labels = cl.fit_predict(X)
            colors = cm.nipy_spectral(cluster_labels.astype(float) / len(np.unique(cluster_labels.astype(float))))

        fig, (ax1) = plt.subplots(1,1)
        fig.set_size_inches(18, 9)
        # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax1.set_title(f"Визуализация кластеризации для {str(cl).split('(')[0]}")
        silhouette_avg = silhouette_score(X, cluster_labels)
        ax1.set_xlabel(f"""Значения коэфициентся силуэт \n Для алгоритма {str(cluster).split('(')[0]} с количеством кластеров =
            {n_clusters} среднее значение коэффициента силуэт: {silhouette_avg}""")

        ax1.scatter(X[:, 0], X[:, 1], s=30, lw=0, alpha=0.99, c=colors, edgecolor="k")
    

    def reductor_vis(self,reductor,n_clusters,n_components,data,min_dist=0.1,n_neighbors=15,perplexity=30.0):
        X = StandardScaler().fit_transform(data)

        if reductor == 'UMAP':
            r = UMAP(n_components=n_components,min_dist = min_dist,n_neighbors=n_neighbors)
            X_=r.fit_transform(X)
        if reductor == 'TSNE':    
            r = TSNE(n_components=n_components,perplexity=perplexity)
            X_=r.fit_transform(X)
        if reductor == 'PCA':   
            r = PCA(n_components = n_components)
            X_=r.fit_transform(X)


        fig, (ax1) = plt.subplots(1,1)
        fig.set_size_inches(18, 9)
        cl = KMeans(n_clusters=n_clusters,random_state=10)
        cluster_labels = cl.fit_predict(X_),
        cluster_labels = cluster_labels[0]
        silhouette_avg = silhouette_score(X_, cluster_labels)

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax1.set_title(f"Визуализация кластеризации методом KMeans при использовании уменьшителя размерности {reductor} \n при n_components = {n_components}")
        ax1.set_xlabel(f"""Среднее значение коэффициента силуэт: {silhouette_avg}""")

        ax1.scatter(X_[:, 0], X_[:, 1], s=30, lw=0, alpha=0.99, c=colors, edgecolor="k")


    def visualizer(self,cluster,reductor,n_clusters,data,n_components=2,min_dist=0.1,n_neighbors=15,perplexity=30.0):
    
        X = StandardScaler().fit_transform(data)

        if reductor == 'UMAP':
            r = UMAP(n_components=n_components,min_dist = min_dist,n_neighbors=n_neighbors)
            X_=r.fit_transform(X)
        if reductor == 'TSNE':
            r = TSNE(n_components=n_components,perplexity=perplexity)
            X_=r.fit_transform(X)
        if reductor == 'PCA':
            r = PCA(n_components = n_components)
            X_=r.fit_transform(X)

        if cluster == 'KMeans':
            cl = KMeans(n_clusters=n_clusters,random_state=10)
            cluster_labels = cl.fit_predict(X_),
            cluster_labels = cluster_labels[0] #поскольку Kmeans.fit_predict() делает тип данных tuple
        # if cluster == 'AffinityPropagation'    
            # AffinityPropagation(),
        if cluster == 'SpectralClustering':    
            cl = SpectralClustering(n_clusters=n_clusters,random_state=10)
            cluster_labels = cl.fit_predict(X_),
            cluster_labels = cluster_labels[0]
        if cluster == 'AgglomerativeClustering':   
            cl = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = cl.fit_predict(X_)
        
        global C    # Needed to modify global copy of globvar
        C = cluster_labels

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 9)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        

        # clusterer = cl
        # cluster_labels = clusterer.fit_predict(X_)
        
        silhouette_avg = silhouette_score(X_, cluster_labels)
        print(
            f"Для алгоритма {str(cluster).split('(')[0]} с количеством кластеров =",
            n_clusters,
            "среднее значение коэффициента силуэт:",
            silhouette_avg,
        )
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title(f"График силуэт для {str(cluster).split('(')[0]}")
        ax1.set_xlabel(f"""Значения коэфициентся силуэт \n Для алгоритма {str(cluster).split('(')[0]} с количеством кластеров =
            {n_clusters} среднее значение коэффициента силуэт: {silhouette_avg}""")
        ax1.set_ylabel("Лейбл кластера")
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.set_title(f"Визуализация кластеризации для {str(cl).split('(')[0]} при помощи уменьшителя размерности {str(r).split('(')[0]}")
        ax2.scatter(X_[:, 0], X_[:, 1], s=30, lw=0, alpha=0.99, c=colors, edgecolor="k")
        

    def elbow(self,rang,data):
        distortions = []
        K = range(1,rang)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(data)
            distortions.append(kmeanModel.inertia_)

        plt.figure(figsize=(16,8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Зашумление')
        plt.title('Метод локтя, показывающий оптимальное значение классов')
        plt.show()

    def adj(self,y,y_true):

        return adjusted_rand_score(y,y_true)

    def final_vis(self,cluster,reductor,n_clusters,n_components,data,min_dist=0.1,n_neighbors=15,perplexity=30.0):
        X = StandardScaler().fit_transform(data)
        if reductor == 'UMAP':
            r = UMAP(n_components=n_components,min_dist = min_dist,n_neighbors=n_neighbors)
            X_=r.fit_transform(X)
        if reductor == 'TSNE':    
            r = TSNE(n_components=n_components,perplexity=perplexity)
            X_=r.fit_transform(X)
        if reductor == 'PCA':   
            r = PCA(n_components = n_components)
            X_=r.fit_transform(X)
        
        if cluster == 'KMeans':
            cl = KMeans(n_clusters=n_clusters,random_state=10)
            cluster_labels = cl.fit_predict(X_),
            cluster_labels = cluster_labels[0] #поскольку Kmeans.fit_predict() делает тип данных tuple
        # if cluster == 'AffinityPropagation'    
            # AffinityPropagation(),
        if cluster == 'SpectralClustering':    
            cl = SpectralClustering(n_clusters=n_clusters,random_state=10)
            cluster_labels = cl.fit_predict(X_),
            cluster_labels = cluster_labels[0]
        if cluster == 'AgglomerativeClustering':   
            cl = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = cl.fit_predict(X_)


        fig, (ax1) = plt.subplots(1,1)
        fig.set_size_inches(18, 9)
        cl = KMeans(n_clusters=n_clusters,random_state=10)
        cluster_labels = cl.fit_predict(X_),
        cluster_labels = cluster_labels[0]
        global C    # Needed to modify global copy of globvar
        C = cluster_labels
        silhouette_avg = silhouette_score(X_, cluster_labels)
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax1.set_title(f"Визуализация кластеризации методом KMeans при использовании уменьшителя размерности {reductor} \n при n_components = {n_components}")
        ax1.set_xlabel(f"""Среднее значение коэффициента силуэт: {silhouette_avg}""")
        ax1.scatter(X_[:, 0], X_[:, 1], s=30, lw=0, alpha=0.99, c=colors, edgecolor="k")

    
    def n_component(self,n_component,data):
        scaler = MinMaxScaler()
        data_rescaled = scaler.fit_transform(data)
        pca = PCA(n_components = n_component)
        pca.fit(data_rescaled)
        # reduced = pca.transform(data_rescaled)
        plt.rcParams["figure.figsize"] = (12,6)

        fig, ax = plt.subplots()
        xi = np.arange(1, n_component +1, step=1)
        y = np.cumsum(pca.explained_variance_ratio_)

        plt.ylim(0.0,1.1)
        plt.plot(xi, y, marker='o', linestyle='--', color='b')

        plt.xlabel('Количество компонентов')
        plt.xticks(np.arange(0, 26, step=1)) #change from 0-based array index to 1-based human-readable label
        plt.ylabel('Суммарная дисперсия(%)')
        # plt.title('The number of components needed to explain variance')

        plt.axhline(y=0.95, color='r', linestyle='-')
        plt.text(0.5, 0.85, '95% порог отсечения', color = 'red', fontsize=16)

        ax.grid(axis='x')
        plt.show()