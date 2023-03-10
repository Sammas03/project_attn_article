
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(123)

x1 = np.random.normal(0, 1, size=1000)
x2 = np.random.normal(-2, 3, size=1000)
x3 = np.random.normal(3, 2.5, size=1000)
x4 = np.random.normal(4, 2.5, size=1000)
x5 = np.random.normal(5, 3, size=1000)
x6 = np.random.normal(4, 2, size=1000)

# 当在同一幅图表中创建多个直方图，最好使用'stepfilled'，并调整透明度
kwargs = {
    "bins": 40,
    "histtype": "stepfilled",
    "alpha": 0.5,
    "density":True
}

fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(x1, label="MLP", **kwargs)
ax.hist(x2, label="AEMLP", **kwargs)
ax.hist(x3, label="GRU", **kwargs)
ax.hist(x4, label="LSTM", **kwargs)
ax.hist(x5, label="CNNLSTM", **kwargs)
ax.hist(x6, label="Proposal Model", **kwargs)
ax.set_title("Histogram for multiple variables")
ax.legend()

plt.show()
