import pandas as pd

df = pd.read_csv("drone_data.csv")


# Removes whole row and columns not recommended
# # Drop rows with any missing values
# df.dropna(inplace=True)
#
# # Drop columns with any missing values
# df.dropna(axis=1, inplace=True)

# Fill missing values with a specific value
df['Exposure time (sec)'].fillna(df['Exposure time (sec)'].mean(), inplace=True)
print("filled")

# # Interpolate missing values using linear interpolation
# df.interpolate(method='linear', inplace=True)
#
# # Remove duplicate rows
# df.drop_duplicates(inplace=True)
#
# # Remove duplicate rows based on specific columns
# df.drop_duplicates(subset=['column1', 'column2'], inplace=True)
