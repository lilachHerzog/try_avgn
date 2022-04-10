#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib2 import Path
from tqdm.autonotebook import tqdm
from avgn.utils.paths import ensure_dir

def to_min_dist(x):
    return 10**(-x)
n_neighbors = [5, 10, 15, 30, 50, 100, 150, 200]
min_dist = [0.0, 0.001, 0.01, 0.1, 0.25, 1.0]
stars="********"

def generate_json(indv_vars):
    import os
    json_loc=Path(variables.DATA_DIR,"raw", variables.DATASET_ID, 'JSON')
    if indv_vars.indv in os.listdir(json_loc):
        print("json available")
        return
    print("\ngenerate_json\n".center(40, "*"))
    from avgn.custom_parsing.general_parsing import generate_json_wav_noise

    for wav_loc in tqdm(indv_vars.wav_list):
        json_txt = generate_json_wav_noise(wav_loc, variables.raw_loc)
    # print an example JSON corresponding to the dataset we just made
    print(json_txt)

def try_different_params(syllable_df, results_loc,save_loc, indv):
    from run_all.clustering import evaluate_clustering
    print("try_different_params".center(60,"#"))
    labels=['n_neighbors', 'min_dist', "RI", "ARI", "SIL", "N_clust"]
    df_HDBSCAN = pd.DataFrame(columns=labels).set_index([labels[0], labels[1]])
    df_HDBSCAN_no_noise = pd.DataFrame(columns=labels).set_index([labels[0], labels[1]])
    print(("projecting and clustering for {}").format(indv).center(60, "*"), "\n")
    results_df = pd.DataFrame()
    embedding_df = pd.DataFrame()
    for neighbor in n_neighbors:
        for dist in min_dist:
            neighbor_loc = Path(results_loc, str(neighbor) + "_neighbors/min_dist"+str(dist))
            print(dist, neighbor)
            embedding, results=project_and_cluster(syllable_df, neighbor_loc,save_loc, indv, neighbor=neighbor, dist=dist)
            # if 'UMAP' not in syllable_df.columns:
            #         project_syllables(syllable_df, dist, neighbor, Path(neighbor_loc, "project_syllables/"))
            #         embedding = project_syllables(syllable_df, dist, neighbor, Path(neighbor_loc, "cluster_to_HDBSCAN/"))
            # else:
            #     print("already clustered")
            #     embedding = [syllable_df[x] for x in syllable_df.columns if 'UMAP' in x]
            # labels = ['n_neighbors', 'min_dist', "RI", "ARI", "SIL", "N_clust"]
            # cluster_content_analysis(syllable_df, embedding, dist, neighbor, neighbor_loc, labels)
            evaluate_clustering(syllable_df, embedding, dist, neighbor, neighbor_loc, labels)
            syllable_df, HDBSCAN_no_filters, HDBSCAN_no_noise = evaluate_clustering(
                syllable_df, embedding, neighbor, dist, neighbor_loc, labels
            )
            # insert_loc = df_HDBSCAN.shape[1]
            df_HDBSCAN=df_HDBSCAN.append(HDBSCAN_no_filters)
            df_HDBSCAN_no_noise=df_HDBSCAN_no_noise.append(HDBSCAN_no_noise)
            results_df=results_df.append(results).set_index(['n_neighbor', 'min_dist'])
            embedding_df=results_df.append(embedding).set_index(['n_neighbor', 'min_dist'])
    pd.DataFrame(df_HDBSCAN, columns=labels).to_csv(Path(results_loc, 'HDBSCAN_no_filters.csv'))
    pd.DataFrame(df_HDBSCAN_no_noise, columns=labels).to_csv(Path(results_loc, 'HDBSCAN_no_noise.csv'))
    pd.DataFrame(results_df).to_csv(Path(results_loc, 'results_df.csv'))
    to_pickle(embedding_df, Path(results_loc, 'embedding_df.xlsx'))

def project_and_cluster(syllable_df, results_loc,save_loc, indv, neighbor=50, dist=0.25):
    cols = [x for x in syllable_df.columns if "UMAP" in x]
    # if len(cols) > 0:
    from run_all.clustering import project_syllables
    print((""
           " syllables for " + indv).center(60, "*"))
    show_syllables(syllable_df, results_loc)
    embedding = project_syllables(syllable_df, dist, neighbor, Path(results_loc, "project_syllables.png"), N_COMP=3)
    for i in range(embedding.shape[1]):
        syllable_df['UMAP' + str(i + 1)] = embedding[:, i]
    embedding = project_syllables(syllable_df, dist, neighbor, Path(results_loc, "cluster_to_HDBSCAN.png"), N_COMP=3)
    pkl_name=save_loc /str(neighbor)/ str(dist)/ "after_UMAP.pkl"
    ensure_dir(pkl_name)
    to_pickle(syllable_df.drop("HDBSCAN",1), pkl_name)
    # else:
    #     print("already clustered")
    #     embedding = [syllable_df[x] for x in syllable_df.columns if 'UMAP' in x]
    labels = ['n_neighbors', 'min_dist', "RI", "ARI", "SIL", "N_clust"]
    from run_all.clustering import cluster_content_analysis
    results=cluster_content_analysis(syllable_df, embedding, dist, neighbor, results_loc, labels)
    return pd.DataFrame(data=embedding, index=['n_neighbor', 'min_dist']), results


def show_syllables(syllable_df, save_loc):
    import matplotlib.pyplot as plt
    from avgn.visualization.spectrogram import draw_spec_set
    specs = np.array([i / np.max(i) for i in syllable_df.spectrogram.values])
    specs[specs < 0] = 0
    draw_spec_set(specs, zoom=1,
                  maxrows=8,
                  colsize=40, fig_loc=save_loc/"show_syllables.png")
    # plt.show()
    fig_out = Path(save_loc, "show_syllables.png")
    ensure_dir(fig_out)
    plt.savefig(fig_out)

def normalize_rescale_pad_spectrograms(syllables_spec, n_jobs, verbosity, indv, figures_loc):
    from run_all.spectrograms import (
        normalize_spectrograms,
        rescale_spectrograms,
        pad_spectrograms,
    )
    print(("rescale spectrograms for " + indv).center(60, "*"))
    syllables_spec = rescale_spectrograms(syllables_spec, n_jobs, verbosity, log_scaling_factor=10,
                                          figures_loc=figures_loc)

    print(("normalize spectrograms for " + indv).center(60, "*"))
    normalize_spectrograms(syllables_spec)

    print(("pad spectrograms for " + indv).center(60, "*"))
    syllables_spec = pad_spectrograms(syllables_spec, n_jobs, verbosity, figures_loc)

    print("the dimensionality of the dataset is ", np.shape(syllables_spec))
    return syllables_spec


def to_pickle(df, save_loc):
    print(" to_pickle ".center(40, "*"))
    ensure_dir(save_loc)
    df.to_pickle(save_loc)

def create_datasets_and_pandas_dataframe(n_jobs, verbosity, indv_vars, figures_loc):
    print("  creating datasets  for {}  ".format(indv_vars.indv).center(60, "*"))
    from spectrograms import create_spectrograms
    from run_all.datasets_and_dataframes import make_dataset, create_dataframe
    ensure_dir(indv_vars)
    generate_json(indv_vars)
    pkl_loc = Path(save_loc, "after_UMAP.pkl")
    if pkl_loc in list((save_loc).iterdir()):
        print("\ninitial syllable_df including UMAP available\n")
        return pd.read_pickle(pkl_loc)
    pkl_loc = Path(save_loc, "the_whole_dataset.pkl")
    if pkl_loc in list((save_loc).iterdir()):
        print("\ninitial syllable_df available\n")
        return pd.read_pickle(pkl_loc)
    print("  creating dataframes  for {}  ".format(indv_vars.indv).center(60, "*"))
    dataset = make_dataset(indv_vars.this_dataset, variables.DATASET_ID, variables.hparams)
    syllable_df = create_dataframe(dataset, n_jobs, verbosity, figures_loc)


    print("  creating spectrograms  for {}  ".format(indv_vars.indv).center(60, "*"))
    syllables_spec, syllable_df = create_spectrograms(dataset, syllable_df, n_jobs, verbosity, figures_loc)
    syllable_df['spectrogram'] = normalize_rescale_pad_spectrograms(syllables_spec, n_jobs, verbosity, indv_vars.indv, figures_loc)
    syllable_df = remove_long_calls(syllable_df, this_dataset)
    print(syllable_df[:3])
    to_pickle(syllable_df,pkl_loc)
    to_pickle(syllable_df.drop('audio', 1), save_loc /'dataset_no_audio.pkl')
    return syllable_df

if __name__ == '__main__':
    from run_all.datasets_and_dataframes import General_Variables, Indv_Variables, remove_long_calls

    variables = General_Variables() #dt_id="2022-04-08_01-07-30"
    n_jobs = -1
    verbosity = 10
    for indv in variables.hyrax_list:
        indv_vars = Indv_Variables(indv.stem, variables.dataset_loc, variables.raw_loc)
        print(indv_vars.indv.center(60, "*"))
        this_dataset=indv_vars.this_dataset
        results_loc = Path(this_dataset, "results")
        save_loc = Path(this_dataset, "different_dfs")
        ensure_dir(save_loc)
        syllable_df = create_datasets_and_pandas_dataframe(n_jobs, verbosity, indv_vars, results_loc)
        try_different_params(syllable_df, results_loc,save_loc, indv_vars.indv)
        # project_and_cluster(syllable_df, results_loc,save_loc, indv_vars.indv)

##
#save df without UMAP, and save UMAP of each dist/neighbor differently
# columns=UMAP1,UMAP2, rows=neigbor+dist

# i have the max of each שילוב in different_dfs.glob("**/**/distribution_of_variables_into_clusters.xlxs")
# find where we have maximum