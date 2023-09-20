from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'


def parser(x):
    return datetime.strptime("199" + x, "%Y-%m")


series = pd.read_csv(
    "./data/sales-of-shampoo.csv",
    header=0,
    parse_dates=[0],
    index_col=0,
    # squeeze=True,
    date_parser=parser,
)
x = series.values
x = np.nan_to_num(x)
size = int(len(x) * 0.66)
train, test = x[:size], x[size:]
history = [e for e in train]
predictions = list()

for t in range(len(test)):
    model = sm.tsa.arima.ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output.values[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print("predicted=%f, expected=%f" % (yhat, obs))
error = mean_squared_error(test, predictions)
print("Test MSE: %.3f" % error)
# plt.plot([e[0] for e in test])
plt.plot(test)
plt.plot([e.item() for e in predictions], color="red")
plt.plot(range(350, 800, 50), color="green")
plt.show()


print(test)
print([e.item() for e in predictions])
