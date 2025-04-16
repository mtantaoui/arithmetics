import numpy as np

n = 1_000_000_000

a = np.arange(start=1, stop=n+1).astype(np.float32)
b = np.arange(start=1, stop=n+1).astype(np.float32)



c = np.add(a,b)