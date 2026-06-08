import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data.csv')

# Set up the plotting environment
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 10))

# 1. Correlation Heatmap
sns.heatmap(df.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.savefig('spearman_correlation_matrix.png')
plt.close()

# 2. Pairplot colored by the target label to see distributions and separations
# (Sampling 500 points for speed if the dataset is large)
sample_df = df.sample(n=min(1000, len(df)), random_state=42)
sns.pairplot(sample_df, hue='label', palette='husl', diag_kind='kde')
plt.savefig('features_pairplot.png')
plt.close()

print("Visualizations saved as 'spearman_correlation_matrix.png' and 'features_pairplot.png'!")