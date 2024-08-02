import pandas as pd
import matplotlib.pyplot as plt

from autoplait import AutoPlait, RegimeSplit

data = pd.read_csv("./console.csv")

fig, ax = plt.subplots()

#print(data)
ax.plot(data)
S1, S2 = RegimeSplit(data.to_numpy())
for l, r in S1:
    ax.axvspan(l, r - 1, color="gray", alpha=0.3)

plt.show()

AutoPlait(data)