import lunadem
import matplotlib.pyplot as plt
import pandas as pd
data = lunadem.get_previously_available_data()
pd.DataFrame(data).to_csv('data.csv', index=False)
a=lunadem.extract_feature_alpha(data)
b=lunadem.extract_feature_beta(data)
c=lunadem.extract_feature_gamma(data)
d=lunadem.extract_feature_delta(data)
print("Feature Alpha:", a)
print("Feature Beta:", b)
print("Feature Gamma:", c)
print("Feature Delta:", d)
# plt.hist(data['sensor_noise_alpha'], bins=20, alpha=0.7, label='Sensor Noise Alpha')
plt.hist(data['label'], bins=20, alpha=0.7, label='Sensor Noise Beta')
plt.show()