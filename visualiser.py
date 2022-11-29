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
import streamlit as st
import matplotlib
# import plotly.plotly as py
import plotly.graph_objects as go
import plotly.express as px
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.subplots import make_subplots

dirname = os.path.dirname(__file__)
C = 0
K = 0
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

        # fig, (ax1) = plt.subplots(1,1)
        # fig.set_size_inches(18, 9)
        # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        # ax1.set_title(f"Визуализация кластеризации для {str(cl).split('(')[0]}")
        # silhouette_avg = silhouette_score(X, cluster_labels)
        # ax1.set_xlabel(f"""Значения коэфициентся силуэт \n Для алгоритма {str(cluster).split('(')[0]} с количеством кластеров =
        #     {n_clusters} среднее значение коэффициента силуэт: {round(silhouette_avg,2)}""")

        # ax1.scatter(X[:, 0], X[:, 1], s=30, lw=0, alpha=0.99, c=colors, edgecolor="k")
        silhouette_avg = silhouette_score(X, cluster_labels)
        plot = go.Figure(data=[go.Scatter(
                         x=X[:,0],
                         y=X[:,1],
                         fill = 'none',
                         mode ='markers',
                         marker=dict(
                            size=6,
                            color=cluster_labels, #set color equal to a variable
                            # colorscale='blues',
                            line_width=1, # one of plotly colorscales
                            showscale=True
    ))])
        plot.update_xaxes(
            tickangle = 90,
            title_text = f"""Значения коэффициента силуэт для алгоритма {str(cluster).split('(')[0]} с количеством кластеров = {n_clusters}
             среднее значение коэффициента силуэт: {round(silhouette_avg,3)}""",
            title_font = {"size":11},
            # title_standoff = 25

        )

        st.plotly_chart(plot)
        # plotly_fig = tls.mpl_to_plotly(fig) ## convert
        # iplot(plotly_fig)
        # plt = px.scatter(X[:, 0], X[:, 1],color=colors)
        # plt.show()


        
    

    def reductor_vis(self,reductor,n_clusters,n_components,data,min_dist=0.1,n_neighbors=15,perplexity=30.0):
        X = StandardScaler().fit_transform(data)

        if reductor == 'UMAP':
            r = UMAP(n_components=n_components,n_neighbors=n_neighbors)
            X_=r.fit_transform(X)
        if reductor == 'TSNE':    
            r = TSNE(n_components=n_components,perplexity=perplexity)
            X_=r.fit_transform(X)
        if reductor == 'PCA':   
            r = PCA(n_components = n_components)
            X_=r.fit_transform(X)


        # fig, (ax1) = plt.subplots(1,1)
        # fig.set_size_inches(18, 9)
        cl = KMeans(n_clusters=n_clusters,random_state=10)
        cluster_labels = cl.fit_predict(X_),
        cluster_labels = cluster_labels[0]
        silhouette_avg = silhouette_score(X_, cluster_labels)

        # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        # ax1.set_title(f"Визуализация кластеризации методом KMeans при использовании уменьшителя размерности {reductor} \n при n_components = {n_components}")
        # ax1.set_xlabel(f"""Среднее значение коэффициента силуэт: {round(silhouette_avg, 2)}""")

        # ax1.scatter(X_[:, 0], X_[:, 1], s=30, lw=0, alpha=0.99, c=colors, edgecolor="k")
        plot = go.Figure(data=[go.Scatter(
                         x=X_[:,0],
                         y=X_[:,1],
                        #  text = (f"Визуализация кластеризации методом KMeans при использовании уменьшителя размерности {reductor} \n при n_components = {n_components}"),
                         fill = 'none',
                         mode ='markers',
                         marker=dict(
                            size=6,
                            color=cluster_labels, #set color equal to a variable
                            # colorscale='blues',
                            line_width=1, # one of plotly colorscales
                            showscale=True
    ))])
        plot.update_xaxes(
            tickangle = 90,
            title_text = f"""Визуализация кластеризации методом KMeans при использовании уменьшителя размерности {reductor} <br> при n_components = {n_components}.Среднее значение коэффициента силуэт: {silhouette_avg:.4f}""",
            title_font = {"size":11},
            # title_standoff = 25

        )
        st.plotly_chart(plot)



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
        
        # global C    # Needed to modify global copy of globvar
        # C = cluster_labels

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.set_size_inches(18, 9)
        # ax1.set_xlim([-0.1, 1])
        # ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        

        # clusterer = cl
        # cluster_labels = clusterer.fit_predict(X_)
        
        # silhouette_avg = silhouette_score(X_, cluster_labels)
        # print(
        #     f"Для алгоритма {str(cluster).split('(')[0]} с количеством кластеров =",
        #     n_clusters,
        #     "среднее значение коэффициента силуэт:",
        #     round(silhouette_avg, 2),
        # )
        # sample_silhouette_values = silhouette_samples(X, cluster_labels)
        # y_lower = 10

        # for i in range(n_clusters):
        #     # Aggregate the silhouette scores for samples belonging to
        #     # cluster i, and sort them
        #     ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        #     ith_cluster_silhouette_values.sort()
        #     size_cluster_i = ith_cluster_silhouette_values.shape[0]
        #     y_upper = y_lower + size_cluster_i
        #     color = cm.nipy_spectral(float(i) / n_clusters)
        #     ax1.fill_betweenx(
        #         np.arange(y_lower, y_upper),
        #         0,
        #         ith_cluster_silhouette_values,
        #         facecolor=color,
        #         edgecolor=color,
        #         alpha=0.7,
        #     )
        #     # Label the silhouette plots with their cluster numbers at the middle
        #     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        #     # Compute the new y_lower for next plot
        #     y_lower = y_upper + 10  # 10 for the 0 samples
        # # Label the silhouette plots with their cluster numbers at the middle
        # ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # # Compute the new y_lower for next plot
        # y_lower = y_upper + 10  # 10 for the 0 samples
        # ax1.set_title(f"График силуэт для {str(cluster).split('(')[0]}")
        # ax1.set_xlabel(f"""Значения коэфициентся силуэт \n Для алгоритма {str(cluster).split('(')[0]} с количеством кластеров =
        #     {n_clusters} среднее значение коэффициента силуэт: {silhouette_avg}""")
        # ax1.set_ylabel("Лейбл кластера")
        # # The vertical line for average silhouette score of all the values
        # ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        # ax1.set_yticks([])  # Clear the yaxis labels / ticks
        # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        # # 2nd Plot showing the actual clusters formed
        # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        # ax2.set_title(f"Визуализация кластеризации для {str(cl).split('(')[0]} при помощи уменьшителя размерности {str(r).split('(')[0]}")
        # ax2.scatter(X_[:, 0], X_[:, 1], s=30, lw=0, alpha=0.99, c=colors, edgecolor="k")

        #-----------------------plotly-------------------------------
       
        silhouette_avg = silhouette_score(X_, cluster_labels)
        fig = make_subplots(rows=1, cols=2,subplot_titles=(f'График "Силуэт" для {n_clusters} кластеров.',"Визуализация кластеризированных значений"))
        fig['layout']['xaxis1'].update(title=f'Значения коэффициента "силуэт"={silhouette_avg:.3f}',range=[-0.1,1])
        fig['layout']['yaxis1'].update(title='Метка кластера')

        # silhouette_avg = silhouette_score(X_, cluster_labels)
        # print(
        #     f"Для алгоритма {str(cluster).split('(')[0]} с количеством кластеров =",
        #     n_clusters,
        #     "среднее значение коэффициента силуэт:",
        #     round(silhouette_avg, 2),
        # )
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            # colors = cm.nipy_spectral(float(i) / n_clusters)
            # colors = cm.Spectral(cluster_labels.astype(float)/n_clusters)
            colors = matplotlib.colors.colorConverter.to_rgb(cm.Spectral(float(i) / n_clusters))
            colors = 'rgb'+str(colors)
            
            # st.write(cluster_labels[cluster_labels==i])
            # st.write(sample_silhouette_values)
            # st.write(ith_cluster_silhouette_values) 
            # colors_list.append(colors)
            filled_area = go.Scatter(y=np.arange(y_lower,y_upper),
                                     x=ith_cluster_silhouette_values,
                                     mode='lines',
                                     showlegend=False,
                                     marker=dict(
                                              color = colors),
                                     fill='tozerox'
                                     )
            fig.append_trace(filled_area,row = 1,col = 1)
                
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        axis_line = go.Scatter(x=[silhouette_avg],
                               y=[0,10],
                               showlegend=False,
                               mode='lines',
                               line=dict(color='red',dash='dash',width=1))

        fig.append_trace(axis_line,row = 1,col = 1)
        
        for i in range(n_clusters):
            colors = matplotlib.colors.colorConverter.to_rgb(cm.Spectral(float(i) / n_clusters))
            colors = 'rgb'+str(colors)
            X__=np.asarray([X_[j] for j in np.where(cluster_labels == i)])
            
            # st.write(X__[0][:, 0]) #дополнительный срез [0] необходим, поскольку переменная Х__ собралась неправильно, не знаю как сделать её подобной Х_
            # st.write(X__[0][:, 1])
            clusters = go.Scatter(x = X__[0][:, 0],
                                  y = X__[0][:, 1],
                                  showlegend=False,
                                  legendgroup="fixed_features",
                                  mode='markers',
                                  marker=dict(color = colors,size=4,line_width=1, # one of plotly colorscales
                                  showscale=False)
                                  )
            fig.append_trace(clusters,1,2)
        fig.update_layout()
        st.plotly_chart(fig)
        # гипотетически есть ещё один способ синхронизировать цвета, на этот раз при помощи cluster_labels[clusterlabels==i] в верхнем цикле и cluster_labels в визуализации второго графика
        
    

    def elbow(self,rang,data):
        distortions = []
        # X = StandardScaler().fit_transform(data) #added new
        K = range(1,rang)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(data) #changed from 'data' to 'X'
            distortions.append(kmeanModel.inertia_)

        plt.figure(figsize=(16,8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Зашумление')
        plt.title('Метод локтя, показывающий оптимальное значение классов')
        plt.show()

    def adj(self,y,y_true):

        return adjusted_rand_score(y,y_true)

    def final_vis(self,cluster,reductor,n_clusters,n_components,data,n_neighbors=15,perplexity=30.0):
        X = StandardScaler().fit_transform(data)
        if reductor == 'UMAP':
            r = UMAP(n_components=n_components,n_neighbors=n_neighbors)
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
        global K 
        K = n_clusters

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

        plot = go.Figure(data=[go.Scatter(
                         x=X_[:,0],
                         y=X_[:,1],
                         fill = 'none',
                         mode ='markers',
                         marker=dict(
                            size=6,
                            color=cluster_labels, #set color equal to a variable
                            # colorscale='blues',
                            line_width=1, # one of plotly colorscales
                            showscale=True
    ))])
        plot.update_layout(
            title=f"""Визуализация кластеризации методом KMeans при использовании уменьшителя размерности {reductor} \n при n_components = {n_components}""",
            xaxis_title = """Среднее значение коэффициента силуэт: {silhouette_avg:.3f}""",
            font = dict(size = 11)
            )
        # plot.update_xaxes(
        #     tickangle = 90,
        #     title_text = f"""Визуализация кластеризации методом KMeans при использовании уменьшителя размерности {reductor} \n при n_components = {n_components}
        #      Среднее значение коэффициента силуэт: {silhouette_avg:.3f}""",
        #     title_font = {"size":11},
        #     # title_standoff = 25

        

        st.plotly_chart(plot)
    
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