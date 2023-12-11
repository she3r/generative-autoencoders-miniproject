import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_dataset(view_data, iteration_num, decoded_data):
    n_rows = 2
    n_cols = len(view_data)
    fig = plt.figure(constrained_layout=True, figsize = (2*n_cols, 2*n_rows))  
    subfigs = fig.subfigures(nrows=n_rows, ncols=1)

    axes = [None, None]
    subfigs[0].suptitle('Dane z MNIST')
    axes[0] = subfigs[0].subplots(nrows=1, ncols=n_cols)

    subfigs[1].suptitle(f'Rekonstrukcje po {iteration_num} iteracjach')
    axes[1] = subfigs[1].subplots(nrows=1, ncols=n_cols)

    for i in range(n_cols):
        axes[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
        axes[0][i].set_xticks(())
        axes[0][i].set_yticks(())
    for i in range(n_cols):
        axes[1][i].clear()
        axes[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
        axes[1][i].set_xticks(())
        axes[1][i].set_yticks(())


    plt.show()


def plot_tsne(x_test_compressed, labels, iteration_num, num_samples=10000):
    tsne = TSNE(init="pca", learning_rate="auto", random_state=42)
    x_test_2D = tsne.fit_transform(x_test_compressed.view(num_samples, -1).cpu().detach().numpy())

    plt.suptitle(f"Przestrzeń ukryta po TSNE dla {iteration_num} epoki")
    plt.scatter(x_test_2D[:, 0], x_test_2D[:, 1], c=labels, s=1, cmap="tab10")
    plt.colorbar()

    plt.show()


def dkl(mean, logvar):
    var = logvar.exp()
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - var)
