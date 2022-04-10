import numpy as np
import matplotlib.pyplot as plt
from pathlib2 import Path
from avgn.utils.paths import ensure_dir
from avgn.visualization.projections import scatter_spec
import hdbscan
import umap
import scipy
import pandas as pd
import numpy as np
import json
from pathlib import Path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import hdbscan
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
import scipy
import sys
from avgn.signalprocessing.create_spectrogram_dataset import flatten_spectrograms
from visualize import (
    cluster_composition_num,
    cluster_composition_percent,
    distribution_of_variables_into_clusters,
    umap_plot,
)
distinct_colors_22 = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                      '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                      '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                      '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                      '#ffffff', '#000000']

# def cluster_syllables_to_HDBSCAN(syllable_df, dist, neighbor, fig_out):
#     # print(indv, np.sum(syllable_df))
#     specs = np.array(
#         [
#             i / np.max(i)
#             for i in syllable_df.spectrogram.values
#         ]
#     )
#     # flatten the spectrograms into 1D
#     specs_flattened = flatten_spectrograms(specs)
#     # project (you can choose different UMAP parameters here)
#     fit = umap.UMAP(min_dist=dist, n_neighbors=neighbor, verbose=True)
#     embedding = fit.fit_transform(specs_flattened)
#     z = list(embedding)
#
#     # cluster
#     clusterer = hdbscan.HDBSCAN(
#         # min_cluster_size should be the smallest size we would expect a cluster to be
#         min_cluster_size=int(len(z) * 0.01),
#     ).fit(z)
#
#     # create a scatterplot of the projections
#     scatter_spec(
#         np.vstack(z),
#         specs,
#         column_size=8,
#         pal_color="tab20",
#         color_points=False,
#         enlarge_points=20,
#         figsize=(10, 10),
#         scatter_kwargs={
#             'labels': list(clusterer.labels_),
#             'alpha': 0.25,
#             's': 0.25,
#             'show_legend': False
#         },
#         matshow_kwargs={
#             'cmap': plt.cm.Greys
#         },
#         line_kwargs={
#             'lw': 3,
#             'ls': "dashed",
#             'alpha': 0.25,
#         },
#         draw_lines=True,
#         n_subset=1000,
#         border_line_width=3,
#     )
#     fig_out = fig_out/ "cluster_to_HDBSCAN_with_/min_dist_"+str(dist)+".png"
#     ensure_dir(fig_out)
#     plt.savefig(fig_out)
#     # plt.show()
#     # syllable_df["specs_flattened_"+str(dist)] = specs_flattened
#     return embedding


def project_syllables(syllable_df, dist, neighbors, fig_out,N_COMP=None):
    print("project_syllables".center(40,"*"))
    import math
    specs = np.array(
        [
            i / np.max(i)
            for i in syllable_df.spectrogram.values
        ]
    )
    specs_flattened = flatten_spectrograms(specs) # flatten the spectrograms into 1D
    # fit = umap.UMAP(min_dist=dist,n_neighbors= neighbors,verbose=True)     # project (you can choose different UMAP parameters here)
    fit = umap.UMAP(n_components=N_COMP,min_dist=dist,n_neighbors= neighbors,verbose=True)     # project (you can choose different UMAP parameters here)
    embedding = fit.fit_transform(specs_flattened)
    z = list(embedding)

    if Path(fig_out).stem == "cluster_to_HDBSCAN":
        # cluster
        clusterer = hdbscan.HDBSCAN(
            # min_cluster_size should be the smallest size we would expect a cluster to be
            min_cluster_size=math.ceil(len(z) * 0.01),
        ).fit(z)
        label = list(clusterer.labels_)
    else:
        label = list(syllable_df['labels'].values)
    syllable_df['HDBSCAN'] = label
    create_projections_scatterplot(z, specs, label, fig_out)
    return embedding

def create_projections_scatterplot(z, specs, label, fig_out):
    scatter_spec(
        z=np.vstack(z),
        specs=specs,
        column_size=8,
        pal_color="tab20",
        color_points=False,
        enlarge_points=20,
        figsize=(10, 10),
        scatter_kwargs={
            'labels': label,
            'alpha': 0.25,
            's': 0.25,
            'show_legend': False
        },
        matshow_kwargs={
            'cmap': plt.cm.Greys
        },
        line_kwargs={
            'lw': 3,
            'ls': "dashed",
            'alpha': 0.25,
        },
        draw_lines=True,
        n_subset=1000,
        border_line_width=3,
    )
    ensure_dir(fig_out)
    plt.savefig(fig_out)
    # plt.show()

# Import functions for calculating unadjusted Rand score
# Resource: https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation

def rand_index_score(clusters, classes):
    tp_plus_fp = scipy.special.comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = scipy.special.comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(scipy.special.comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = scipy.special.comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def calc_rand(pred, true):
    classnames, pred = np.unique(pred, return_inverse=True)
    classnames, true = np.unique(true, return_inverse=True)
    return (rand_index_score(pred, true))



def evaluate_clustering(syllable_df, embedding, dist, neighbor, results_loc, labels):
    import math
    HDBSCAN = hdbscan.HDBSCAN(min_cluster_size=math.ceil(0.01 * embedding.shape[0]),
                              cluster_selection_method='leaf').fit(embedding)
    hdb_labels = HDBSCAN.labels_  # these are the predicted clusters labels
    hdb_labels_no_noise = hdb_labels.copy()
    assigned = np.full((len(hdb_labels_no_noise),), False)
    assigned[np.where(hdb_labels_no_noise != -1)[0]] = True
    hdb_labels_no_noise = hdb_labels_no_noise[assigned]  # the predicted cluster labels without noise datapoints
    df_no_noise = syllable_df.loc[assigned]  # the dataframe without noise datapoints
    embedding_no_noise = embedding[assigned, :]  # the UMAP coordinates without noise datapoints
    results_txt = "{}/results.txt".format(results_loc)
    ensure_dir(results_txt)
    f = open(results_txt, "w")
    HDBSCAN_no_filters= calc_stats(neighbor, dist, hdb_labels, list(syllable_df['labels'].values), embedding, labels)
    HDBSCAN_no_noise= calc_stats(neighbor, dist, hdb_labels_no_noise, list(df_no_noise['labels'].values),
                                 embedding_no_noise, labels)
    f.write("HDBSCAN_no_filters: ")
    f.write(str(HDBSCAN_no_filters))
    f.write("HDBSCAN_no_noise: ")
    f.write(str(HDBSCAN_no_noise))
    print("HDBSCAN_no_filters".center(40,"*"), "\n", HDBSCAN_no_filters)
    print("HDBSCAN_no_noise".center(40, "*"), "\n", HDBSCAN_no_noise)
    HDBSCAN_no_filters = pd.DataFrame(data=[HDBSCAN_no_filters], columns=labels).set_index(labels[0:1])
    HDBSCAN_no_noise = pd.DataFrame(data=[HDBSCAN_no_noise], columns=labels).set_index(labels[0:1])
    return syllable_df,HDBSCAN_no_filters, HDBSCAN_no_noise


def calc_stats(neighbor, dist, cluster_labels, true_labels, embedding_data, labels):
    dict= {
        "n_neighbors": neighbor, "min_dist": dist,
           "RI": calc_rand(cluster_labels, true_labels),
           "ARI": adjusted_rand_score(cluster_labels, true_labels),
           "SIL": silhouette_score(embedding_data, cluster_labels),
           "N_clust": len(list(set(cluster_labels)))
    }
    # dict={labels[i]:dict[i] for i in range(len(dict))}
    return dict

def create_leveled_df():
    stats = ["RI", "ARI", "SIL", "N_clust"]
    types = ["index", "HDBSCAN", "HDBSCAN_no_noise"]
    n_neighbors = [5, 10, 15, 30, 50, 100, 150, 200]
    min_dist = [0.0, 0.001, 0.01, 0.1, 1.0]
    dist = np.repeat(min_dist, len(n_neighbors))
    n = np.tile(n_neighbors, len(min_dist))
    data = np.ones((len(n_neighbors) * len(min_dist)))
    d2 = {'data': data,
          'min_dist': dist,
          'n_neighbors': n}
    df2 = pd.DataFrame(data=d2)
    df2 = df2.set_index(['min_dist', 'n_neighbors'])
    t = np.repeat(types, len(stats))
    s = np.tile(stats, len(types))
    data = np.ones((len(types) * len(stats)))
    # print(len(t),len(s),len(data))
    d = {'data': data,
         'stats': s,
         'types': t}
    df = pd.DataFrame(data=d)

def a(arr1,arr2):
    for a1 in arr1:
        for a2 in arr2:
            yield (a1,a2)

def find_max(loc):
    labels=["RI", "ARI", "SIL", "N_clust"]
    hdbscan_no_noise = pd.read_csv(loc+"/HDBSCAN_no_noise.csv")
    hdbscan = pd.read_csv(loc+"/HDBSCAN.csv")
    hdbscan_maxV = hdbscan.idxmax()
    hdbscan_minV = hdbscan.idxmin()
    hdbscan_no_noise_maxV = hdbscan_no_noise.idxmax()
    hdbscan_no_noise_minV = hdbscan_no_noise.idxmin()
    data=[]
    for label in labels:
        hdbscan_max = hdbscan.iloc[hdbscan_maxV.loc[label]]
        hdbscan_min = hdbscan.iloc[hdbscan_minV.loc[label]]
        hdbscan_no_noise_max=hdbscan.iloc[hdbscan_no_noise_maxV.loc[label]]
        hdbscan_no_noise_min=hdbscan.iloc[hdbscan_no_noise_minV.loc[label]]
        data.append([hdbscan_max[label], hdbscan_max.n_neighbors, hdbscan_max.min_dist,
                     hdbscan_min[label], hdbscan_min.n_neighbors, hdbscan_min.min_dist,
                     hdbscan_no_noise_max[label], hdbscan_no_noise_max.n_neighbors, hdbscan_no_noise_max.min_dist,
                     hdbscan_no_noise_min[label], hdbscan_no_noise_min.n_neighbors, hdbscan_no_noise_min.min_dist
                     ])
    df = pd.DataFrame(data=data, index=labels,
                           columns=pd.MultiIndex.from_product([["HDBSCAN","HDBSCAN_no_noise"],["max", "min"], ["value", "n_neighbors", "min_dist"]]))


def cluster_content_analysis(df, embedding, dist, neighbor, results_loc, labels):
    import math
    # evaluate_clustering(df, embedding, dist, neighbor, results_loc, labels)
    LABEL_COL = 'labels'
    analyze_by = LABEL_COL  # Column name of variable by which to analyze cluster content.
    # Select any variable of interest in DF
    HDBSCAN = hdbscan.HDBSCAN(min_cluster_size=math.ceil(0.01 * embedding.shape[0]),
                              cluster_selection_method='leaf').fit(embedding)
    hdb_labels = HDBSCAN.labels_  # these are the predicted clusters labels
    # df['HDBSCAN'] = hdb_labels  # add predicted cluster labels to dataframe
    by_types = sorted(list(set(df[analyze_by])))
    cluster_labeltypes = sorted(list(set(hdb_labels)))
    umap_plot(embedding, hdb_labels, ['#d9d9d9'] + distinct_colors_22, results_loc/"hdb_labels")
    umap_plot(embedding, df[LABEL_COL], distinct_colors_22, results_loc/"original_labels")
    stats_tab = cluster_composition_num(df, analyze_by, hdb_labels, cluster_labeltypes, by_types, results_loc, dist)
    cluster_composition_percent(stats_tab, analyze_by, cluster_labeltypes, by_types, results_loc, dist)
    results=distribution_of_variables_into_clusters(stats_tab, cluster_labeltypes, by_types, analyze_by, results_loc, neighbor, dist)
    return results

