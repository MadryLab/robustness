import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
import seaborn as sns

def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax

def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, filename=None):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)                
            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0: 
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def show_image_column(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, filename=None):
    W, H = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)
            ax.imshow(xlist[w][h].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and h == 0:
                ax.set_title(ylist[w], fontsize=fontsize)
            if tlist: 
                ax.set_title(tlist[w][h], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def filter_data(metadata, criteria, value):
    crit = [True] * len(metadata) 
    for c, v in zip(criteria, value):
        v = [v] if not isinstance(v, list) else v
        crit &= metadata[c].isin(v)
    metadata_int = metadata[crit]
    exp_ids = metadata_int['exp_id'].tolist()
    return exp_ids

def plot_axis(ax, x, y, xlabel, ylabel, **kwargs):
    ax.plot(x, y, **kwargs)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)


def plot_tsne(x, y, npca=50, markersize=10):
    Xlow = PCA(n_components=npca).fit_transform(x)
    Y = manifold.TSNE(n_components=2).fit_transform(Xlow)
    palette = sns.color_palette("Paired", len(np.unique(y)))
    color_dict = {l: c for l, c in zip(range(len(np.unique(y))), palette)}
    colors = [color_dict[l] for l in y]
    plt.scatter(Y[:, 0], Y[:, 1], markersize, colors, 'o')
    plt.show()
