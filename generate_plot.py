import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

value_list = np.zeros((1000,5))

for i in range(5):
    fields = ['Step', 'Value']
    df = pd.read_csv('./checkpoints/run-train-tag-batch_loss_' + str(i+1) + '.csv', usecols=fields)

    steps = df.Step.to_numpy()
    value = df.Value.to_numpy()

    value_list[:,i] = value

value_mean = np.mean(value_list, axis=1)
value_std = np.std(value_list, axis=1)

plt.plot(steps, value_mean)
plt.xlabel('Train Step')
plt.ylabel('Tversky Loss')
plt.fill_between(steps, value_mean - value_std, value_mean + value_std, alpha=.5)
plt.grid()
plt.show()