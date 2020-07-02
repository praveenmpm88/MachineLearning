import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb
fig = mp.figure()
fig, axes = mp.subplots(1, 2, figsize=(10, 5))
data = {'species': ['a', 'a', 'a', 'b', 'b'],
        'age': [9, 5, 1, 10, 20],

        }
df = pd.DataFrame(data)
print(df)
sb.pointplot(x='species', y='age', data=df, ax=axes[0])
sb.barplot(x='species', y='age', data=df, ax=axes[1])
axes[0].set_title("Pointsplot")
axes[1].set_title("barplot")
mp.tight_layout()
mp.show()