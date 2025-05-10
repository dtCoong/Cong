import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

os.makedirs("sourcecode/image/bai3", exist_ok=True)

df = pd.read_csv("sourcecode/results.csv")
teams = sorted(df['team'].unique())
columns = [i for i in df.columns][5:]


df = df.replace('N/a', np.nan)
df.fillna(0,inplace = True)
df[columns] = df[columns].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[columns])


wcss = []
for k in range(1,11):
    kmean = KMeans(n_clusters=k, random_state=0)
    kmean.fit(X_scaled)
    wcss.append(kmean.inertia_)
plt.figure(figsize=(10,6))
plt.plot(range(1,11),wcss, marker='o')
plt.title('Elbow')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('sourcecode/image/bai3/ElbowMethod.png',dpi=300)
plt.show()
plt.close()



k = 4 
kmean = KMeans(n_clusters=k, random_state=0)
kmean.fit(X_scaled)
pca = PCA(n_components = 2)
pca.fit(X_scaled)
data_pca = pca.transform(X_scaled)
data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2'])
data_pca['Cluster'] = kmean.labels_
sns.scatterplot(data=data_pca, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=60)
plt.title(f'KMeans Clustering (k={k}) with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend(title='Cluster')
plt.savefig('sourcecode/image/bai3/PCA.png',dpi=300)
plt.show()
plt.close()