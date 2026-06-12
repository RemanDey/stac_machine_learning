import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import lunadem
import numpy as np
# Load the dataset
data = pd.read_csv('/home/remandey/my-programs/stac/stac_machine_learning/visualization/data_engineered.csv')
a=lunadem.extract_feature_alpha(data)
data['feature_alpha'] = a
b=lunadem.extract_feature_beta(data)
data['feature_beta'] = b
c=lunadem.extract_feature_gamma(data)
data['feature_gamma'] = c
d=lunadem.extract_feature_delta(data)
data['feature_delta'] = d

# Get all column names except 'label'
variables = [col for col in data.columns if col != 'label']

# Create scatter plots for each variable against 'label'
for var in variables:
    # A high MI score confirms a relationship exists despite zero correlation
    x = data[var]
    y = data['label']
    mi_score = mutual_info_regression(x.values.reshape(-1, 1), y)
    print(f'Mutual Information between {var} and label: {mi_score[0]:.4f}')
    plt.figure() # Create a new figure for each plot
    plt.scatter(np.sin(x), y)
    plt.xlabel(var.replace('_', ' ').title()) # Format variable name for readability
    plt.ylabel('Label')
    plt.title(f'{var.replace("_", " ").title()} vs Label')
    plt.grid(True)

    # plt.savefig(f"{var}vsLabel.png", dpi=1200)

# 3. Display the plot

    plt.show()