#%%
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_data(val_acc_history, loss_acc_history):
    plt.plot(loss_acc_history, label = 'Validation')
    plt.title('Loss per epoch')
    plt.legend()
    plt.show()
    plt.plot(val_acc_history, label = 'Validation')
    plt.title('Accuracy per epoch')
    plt.legend()
    plt.show()
#%%
acc = np.load('GoogLeNet_val_acc_history.npy', allow_pickle=True)
loss = np.load('GoogLeNet_loss_history.npy', allow_pickle=True)


plot_data(acc, loss)

# %%
