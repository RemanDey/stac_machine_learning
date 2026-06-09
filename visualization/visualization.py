import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lunadem
# Load the dataset
data = pd.read_csv('/home/remandey/my-programs/stac/stac_machine_learning/visualization/data.csv')
a=lunadem.extract_feature_alpha(data)
data['feature_alpha'] = a
b=lunadem.extract_feature_beta(data)
data['feature_beta'] = b
c=lunadem.extract_feature_gamma(data)
data['feature_gamma'] = c
d=lunadem.extract_feature_delta(data)
data['feature_delta'] = d

# Set up the plotting environment
sns.set_theme(style="whitegrid")

# 1. Correlation Heatmap
methods=['pearson', 'spearman', 'kendall']
for method in methods:
    plt.figure(figsize=(14, 12))
    corr = data.corr(method=method)
    # Create a mask for the upper triangle to reduce redundancy and clutter
    
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap='coolwarm', 
        square=True, linewidths=.5, cbar_kws={"shrink": .8}
    )
    plt.title(f'Feature Correlation Matrix ({method.capitalize()})', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'{method}_correlation_matrix.png')
    plt.close()
    print(f"Visualization saved as '{method}_correlation_matrix.png'!")

# 2. Pairplot colored by the target label (Moved outside the loop to avoid redundant compute)
sample_df = data.sample(n=min(1000, len(data)), random_state=42)
sns.pairplot(sample_df, hue='label', palette='husl', diag_kind='kde')
plt.savefig('features_pairplot.png')
plt.close()
print("Visualization saved as 'features_pairplot.png'!")