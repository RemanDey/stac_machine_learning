import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data = pd.read_csv('/home/remandey/my-programs/stac/stac_machine_learning/visualization/data.csv')
# Set up the plotting environment
sns.set_theme(style="whitegrid")
x=data['sensor_noise_beta']
x=np.sin(x)  # Apply sine transformation to the crater_density feature
print(x)
y=data['label']
sns.scatterplot(x=x, y=y)
plt.title('Crater Density vs Label', fontsize=16)
plt.xlabel('Crater Density', fontsize=12)
plt.ylabel('Label', fontsize=12)
plt.show()