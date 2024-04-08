import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Read the dataset
df = pd.read_csv('drone_data.csv', delimiter=';') #Read the CSV File
# Exclude non-numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])
# Calculate the correlation matrix
corr_matrix = numeric_df.corr()
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()