import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv("temp.csv")

plt.plot(data)

plt.tight_layout()
plt.show()