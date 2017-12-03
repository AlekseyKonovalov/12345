import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

data = pd.read_csv("Wholesale_customers_data.csv", header=0, sep=",")
#print (data)
#print (data.columns)

#nominal (категориальные)
#print (data['Channel'].value_counts())
#print (data['Region'].value_counts())

#Continuous (вещественные количественые)
#print (data['Fresh'].value_counts())
#print (data['Milk'].value_counts())
#print (data['Grocery'].value_counts())
#print (data['Frozen'].value_counts())
#print (data['Detergents_Paper'].value_counts())
#print (data['Delicassen'].value_counts())


#статистические характеристики
#print(data[['Fresh', 'Milk', 'Grocery', 'Frozen','Detergents_Paper', 'Delicassen']].describe())

for feature in ['Channel', 'Region']:
    data[feature] = data[feature].astype('category')

#print(data.describe(include=['category']))

# обработка пропущенных значений
#print(data.describe(include='all').loc['count'])#пропущенных значений нет

#Fresh
#ind = np.arange(data['Fresh'].count())
#plt.bar(ind, data['Fresh'])
#plt.show()
#print (data[data['Fresh'] > 60000])
#pаменим большие значения средним по классу Region = 3
mean_fare_for_1_class = data[(data['Fresh'] < 60000) & (data['Region'] == 3)]['Fresh'].mean()
#print (mean_fare_for_1_class)
data.loc[data['Fresh'] > 60000, 'Fresh'] = mean_fare_for_1_class
#ind = np.arange(data['Fresh'].count())
#plt.bar(ind, data['Fresh'])
#plt.show()

#milk
#ind = np.arange(data['Milk'].count())
#plt.bar(ind, data['Milk'])
#plt.show()
#print (data[data['Milk'] > 50000])
mean_fare_for_1_class = data[(data['Milk'] < 50000) & (data['Region'] == 3)]['Milk'].mean()
#print (mean_fare_for_1_class)
data.loc[data['Milk'] > 50000, 'Milk'] = mean_fare_for_1_class
#ind = np.arange(data['Milk'].count())
#plt.bar(ind, data['Milk'])
#plt.show()

#Delicassen
ind = np.arange(data['Delicassen'].count())
plt.bar(ind, data['Delicassen'])
#plt.show()
#print (data[data['Delicassen'] > 10000])
mean_fare_for_1_class = data[(data['Delicassen'] < 10000) & (data['Region'] == 3)]['Delicassen'].mean()
#print (mean_fare_for_1_class)
data.loc[data['Delicassen'] > 10000, 'Delicassen'] = mean_fare_for_1_class
ind = np.arange(data['Delicassen'].count())
plt.bar(ind, data['Delicassen'])
#plt.show()

#Grocery
#ind = np.arange(data['Grocery'].count())
#plt.bar(ind, data['Grocery'])
#plt.show()
#print (data[data['Grocery'] > 60000])
mean_fare_for_1_class = data[(data['Grocery'] < 60000)& (data['Channel'] == 2)]['Grocery'].mean()
#print (mean_fare_for_1_class)
data.loc[data['Grocery'] > 60000, 'Grocery'] = mean_fare_for_1_class
#ind = np.arange(data['Grocery'].count())
#plt.bar(ind, data['Grocery'])
#plt.show()

#Frozen
#ind = np.arange(data['Frozen'].count())
#plt.bar(ind, data['Frozen'])
#plt.show()
#print (data[data['Frozen'] > 20000])
mean_fare_for_1_class = data[(data['Frozen'] < 20000) & (data['Channel'] == 1)]['Frozen'].mean()
#print (mean_fare_for_1_class)
data.loc[data['Frozen'] > 20000, 'Frozen'] = mean_fare_for_1_class
#ind = np.arange(data['Frozen'].count())
#plt.bar(ind, data['Frozen'])
#plt.show()

#Detergents_Paper
#ind = np.arange(data['Detergents_Paper'].count())
#plt.bar(ind, data['Detergents_Paper'])
#plt.show()
#print (data[data['Detergents_Paper'] > 30000])
mean_fare_for_1_class = data[(data['Detergents_Paper'] < 30000) & (data['Channel'] == 2)]['Detergents_Paper'].mean()
#print (mean_fare_for_1_class)
data.loc[data['Detergents_Paper'] > 30000, 'Detergents_Paper'] = mean_fare_for_1_class
#ind = np.arange(data['Detergents_Paper'].count())
#plt.bar(ind, data['Detergents_Paper'])
#plt.show()

# Привести Dataset к нужному виду для подачи в алгоритм
quantitative = data[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
X = quantitative.values
#print (X)
stand_scaler = StandardScaler().fit(X)
X_normalized = stand_scaler.transform(X)
#print (X_normalized)

# Осуществить сокращение размерности данных методом главных компонент
pca = PCA(n_components=2).fit(X_normalized)
X_reduced = pca.transform(X_normalized)
#print (X_reduced)

#кластеризация

#Агломеративная кластеризация
#дендрограмма

Z = linkage(X_reduced, method='ward', metric='euclidean')
plt.figure()
dn = dendrogram(Z)
plt.show()

###
labels = AgglomerativeClustering(n_clusters=4, affinity='manhattan', linkage='complete').fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels)
print ("AgglomerativeClustering(n_clusters=4, affinity='manhattan', linkage='complete')")
#plt.show()
print (silhouette_score(X_reduced, labels, metric='manhattan'))

labels = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward').fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels)
print ("AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')")
#plt.show()
print (silhouette_score(X_reduced, labels, metric='euclidean'))

labels = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='complete').fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels)
print ("AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='complete')")
#plt.show()
print (silhouette_score(X_reduced, labels, metric='cosine'))


#k-means
labels = KMeans(n_clusters=4, init='k-means++', max_iter=500).fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels)
print ("KMeans(n_clusters=4, init='k-means++', max_iter=500)")
print (silhouette_score(X_reduced, labels))
#plt.show()

labels = KMeans(n_clusters=4, max_iter=200).fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels)
print ("KMeans(n_clusters=4, max_iter=200)")
print (silhouette_score(X_reduced, labels))
#plt.show()

labels = KMeans(n_clusters=4, n_init=20 , max_iter=400).fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels)
print ("KMeans(n_clusters=4, n_init=20 , max_iter=400)")
print (silhouette_score(X_reduced, labels))
#plt.show()

#DBSCAN
labels =  DBSCAN().fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels)
print ("DBSCAN()")
print (silhouette_score(X_reduced, labels))
#plt.show()

labels =  DBSCAN( eps=1 ).fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels)
print ("DBSCAN( eps=1 )")
print (silhouette_score(X_reduced, labels))
#plt.show()

labels =  DBSCAN(min_samples=10 ).fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels)
print ("DBSCAN(min_samples=10 )")
print (silhouette_score(X_reduced, labels))
#plt.show()

labels =  DBSCAN(eps=0.25, min_samples=10  ).fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels)
print ("DBSCAN(eps=0.25, min_samples=10  )")
print (silhouette_score(X_reduced, labels))
#plt.show()