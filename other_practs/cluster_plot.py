import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans

data = pd.read_csv("a.csv")
x = data.drop("Outcome",axis=1)

df = pd.DataFrame(x)
features =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Apply k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)

cluster_labels_1=kmeans.labels_
print(cluster_labels_1)

cluster_centers = kmeans.cluster_centers_
print(cluster_centers)

plt.figure(figsize=(9,6))
sb.scatterplot(x='Pregnancies',y='Glucose',hue=cluster_labels_1,palette = "Dark2", legend = False,data=x)
sb.scatterplot(x = cluster_centers[:, 0], y = cluster_centers[:, 1], color = "black", marker = "x", s = 200)
plt.ylabel('Glucose')
plt.xlabel('Pregnancies')
plt.title('Cluster Plot')
plt.show()
