import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

from autoplait import AutoPlait

data = pd.read_csv("./console.csv")

fig, ax = plt.subplots()

colors = list(matplotlib.colors.CSS4_COLORS.values())

#print(data)
ax.plot(data)
result = AutoPlait(data.to_numpy())
for i, S in enumerate(result):
    for l, r in S:
        ax.axvspan(l, r, color=colors[i], alpha=0.8)

plt.show()

AutoPlait(data)