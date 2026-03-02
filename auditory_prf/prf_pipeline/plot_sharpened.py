import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/ekim/auditory-pRF-subcortical/sharpened.csv')

plt.figure(figsize=(12, 4))
plt.plot(df['column_name'])
plt.title('Sharpened Array')
plt.xlabel('Index')
plt.ylabel('Value')
plt.tight_layout()
plt.show()