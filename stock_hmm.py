import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import hmmlearn.hmm  as hmm
import warnings

warnings.filterwarnings('ignore')
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2017, 12, 25)

df = web.DataReader('AMZN', 'yahoo', start, end)#.reset_index()
print(df.head())
df.plot()
plt.show()

closing_values = df.Close.values.astype('float32')

diff_percentage = 100.0 * np.diff(closing_values) / closing_values[:-1]
print(diff_percentage.shape)
print(diff_percentage.dtype)
#dates = df.Date.values[1:]
#print(dates)
volume_of_shares = df.Volume.values[1:].astype('float32')
print(volume_of_shares.shape)
print(volume_of_shares.dtype)

# Stack the percentage diff and volumn valudes column-wise fro training
X = np.column_stack([diff_percentage, volume_of_shares])

print("\n Training HMM ..")
print('X shape', X.shape)
model = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=2000).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)
print('hidden states shape', hidden_states.shape)
print('model means shape', model.means_.shape)
print('model covars shape', model.covars_.shape)

num_samples = 500
samples, states = model.sample(num_samples)

print(states.shape)
print(set(states))
print(samples.shape)
print(samples[:10])
plt.plot(np.arange(num_samples), samples[:, 0], c='black')
plt.show()
