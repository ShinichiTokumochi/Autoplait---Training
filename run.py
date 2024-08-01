import pandas as pd
import matplotlib.pyplot as plt

from autoplait import AutoPlait

data = pd.read_csv("./console.csv")

fig, ax = plt.subplots()

#print(data)
ax.plot(data)
ax.axvspan(20, 50, color="gray", alpha=0.3)

plt.show()

AutoPlait(data)