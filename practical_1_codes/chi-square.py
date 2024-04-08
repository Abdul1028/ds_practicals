import pandas as pd
from scipy.stats import chi2_contingency
# Read the dataset
df = pd.read_csv('drone_data.csv', sep=';')

# Create a contingency table of counts between 'Altitude (m)' and 'Exposure time (sec)'
contingency_table = pd.crosstab(df['Altitude (m)'], df['Exposure time (sec)'])

# Perform chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print the results of the chi-square test
print(f"Chi-square statistic: {chi2}")  # Print chi-square statistic
print(f"P-value: {p}")  # Print p-value
print(f"Degrees of freedom: {dof}")  # Print degrees of freedom
print("Expected frequencies:")  # Print expected frequencies
print(expected)

