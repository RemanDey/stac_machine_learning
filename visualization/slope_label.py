import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')
x=data['slope']
y=data['label']
plt.scatter(x, y)
plt.xlabel('Slope')
plt.ylabel('Label')
plt.title('Slope vs Label')
plt.grid(True)
plt.show()