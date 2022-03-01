import pandas as pd
import numpy as np

a = np.zeros((2,3))
b = pd.DataFrame(a)
c = pd.Series([1,2,3])

print(isinstance(a,pd.DataFrame))
print(isinstance(b,pd.DataFrame))
print(isinstance(c,pd.Series))