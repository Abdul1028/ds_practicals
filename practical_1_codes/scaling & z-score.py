import pandas as pd

df = pd.read_csv('drone_data.csv', delimiter=';')


def min_max_scaling(column, new_min, new_max):
    return (column - column.min()) / (column.max() - column.min()) * (new_max - new_min) + new_min

def z_score_normalization(column):
    return (column - column.mean()) / column.std()



# Apply min-max scaling to the 4th column and create a new column 'NSB_scaled'
df['NSB_scaled'] = min_max_scaling(df['NSB (mpsas)'], 1, 10)
print("After scaling: ")
print(df.head())

# Apply z-score normalization to the 'NSB (mpsas)' column and create a new column 'NSB_zscore'
df['NSB_zscore'] = z_score_normalization(df['NSB (mpsas)'])
print("After normalization: ")
print(df)
