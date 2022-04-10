import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.legend import Legend
import matplotlib
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from avgn.utils.paths import ensure_dir
from pathlib2 import Path


def umap_plot(axes, scat_labels, mycolors, fig_out):
    labeltypes = sorted(list(set(scat_labels)))
    pal = sns.color_palette(mycolors, n_colors=len(labeltypes))
    color_dict = dict(zip(labeltypes, pal))
    c = [color_dict[val] for val in scat_labels]
    scatters = []
    for label in labeltypes:
        scatters.append(matplotlib.lines.Line2D([0], [0], linestyle="none", c=color_dict[label], marker='o'))
    if axes.shape[1] > 1:
        x = axes[:, 0]
        y = axes[:, 1]
        figsize = (6, 6)
        fig = plt.figure(figsize=figsize, frameon=False)
        ax = umap_2D_get_axes(x, y, c)
        outname = str(fig_out) + '_umap_2Dplot.png'
        show_fig(ax, scatters, labeltypes, outname)
        if axes.shape[1] > 2:
            figsize = (10, 10)
            fig = plt.figure(figsize=figsize, frameon=False)
            z = axes[:, 2]
            ax = umap_3D_get_axes(x, y, z, fig, c)
            outname = str(fig_out) + '_umap_3Dplot.png'
            show_fig(ax, scatters, labeltypes, outname)

def show_fig(ax,scatters, labeltypes,outname):
    ax.legend(scatters, labeltypes, numpoints=1)
    ensure_dir(outname)
    plt.savefig(outname, facecolor="white")

def umap_2D_get_axes(x, y, c):
    """
    """

    plt.scatter(x, y, alpha=1,
                s=10, c=c)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    return plt

def umap_3D_get_axes(x, y, z, fig, c):
    """
    """

    ax = fig.add_subplot(111, projection='3d')

    Axes3D.scatter(ax,
                   xs=x,
                   ys=y,
                   zs=z,
                   zdir='z',
                   s=20,
                   label=c,
                   c=c,
                   depthshade=False)

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    return ax


def plotly_viz(x, y, z, scat_labels, mycolors):
    """
    Function that creates interactive 3D plot with plotly from
    an input dataset, color-colored by the provided labels.

    Parameters
    ----------
    x : 1D numpy array (numeric) or list
        x coordinates of datapoints

    y: 1D numpy array (numeric) or list
       y coordinates of datapoints

    z: 1D numpy array (numeric) or list
       z coordinates of datapoints

    scat_labels: List-of-Strings
                 Datapoint labels

    mycolors: String or List-of-Strings
              Seaborn color palette name (e.g. "Set2") or list of
              colors (Hex value strings) used for coloring datapoints
              (e.g. ["#FFEBCD","#0000FF",...])

    Returns
    -------

    -

    """
    labeltypes = sorted(list(set(scat_labels)))
    pal = sns.color_palette(mycolors, n_colors=len(labeltypes))
    color_dict = dict(zip(labeltypes, pal))
    c = [color_dict[val] for val in scat_labels]

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                       hovertext=scat_labels,
                                       marker=dict(
                                           size=4,
                                           color=c,  # set color to an array/list of desired values
                                           opacity=0.8
                                       ))])

    fig.update_layout(scene=dict(
        xaxis_title='UMAP1',
        yaxis_title='UMAP2',
        zaxis_title='UMAP3'),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10))

    return fig


def cluster_composition_num(df, analyze_by, cluster_labels, cluster_labeltypes, by_types, results_loc, dist):

    stats_tab = np.zeros((len(cluster_labeltypes), len(by_types)))
    for i, clusterlabel in enumerate(cluster_labeltypes):
        label_df = df.loc[cluster_labels == clusterlabel]
        for j, by_type in enumerate(by_types):
            stats_tab[i, j] = sum(label_df[analyze_by] == by_type)

    stats_tab_df = pd.DataFrame(stats_tab, index=cluster_labeltypes, columns=by_types)
    show_cluster_evaluation_matrix(cluster_labeltypes, by_types, stats_tab_df, analyze_by, str(results_loc) + '/cluster_composition_absolute_numbers.png', dist, fmt='.0f')
    return stats_tab

def cluster_composition_percent(stats_tab, analyze_by, cluster_labeltypes, by_types, results_loc, dist):
    stats_tab_norm = np.zeros((stats_tab.shape))
    rowsums = np.sum(stats_tab, axis=1)
    for i in range(stats_tab.shape[0]):
        stats_tab_norm[i, :] = stats_tab[i, :] / rowsums[i]
    stats_tab_norm_df = pd.DataFrame(stats_tab_norm, index=cluster_labeltypes, columns=by_types) * 100
    show_cluster_evaluation_matrix(cluster_labeltypes, by_types, stats_tab_norm_df, analyze_by, str(results_loc) + '/cluster_composition_percent.png', dist, fmt='.1f')



def show_cluster_evaluation_matrix(cluster_labeltypes, by_types, stats_tab, analyze_by, save_loc, dist, fmt):
    plt.figure(figsize=(int(len(cluster_labeltypes)), int(len(by_types)) / 2))
    ax = sns.heatmap(stats_tab, annot=True, cmap='viridis', fmt=fmt, cbar=False)
    ax.set_xlabel(analyze_by, fontsize=16)
    ax.set_ylabel("Cluster label", fontsize=16)
    ax.tick_params(labelsize=16)
    ensure_dir(save_loc)
    fig = ax.get_figure()
    fig.savefig(save_loc)
    save_loc=Path(Path(save_loc).parent.parent,(Path(save_loc).stem+".xlsx"))
    ensure_dir(save_loc)
    stats_tab.to_excel(save_loc,sheet_name=str(dist), index=cluster_labeltypes, columns=by_types)


def distribution_of_variables_into_clusters(stats_tab, cluster_labeltypes, by_types, analyze_by, results_loc, neighbor, dist):
    stats_tab_norm = np.zeros((stats_tab.shape))
    colsums = np.sum(stats_tab, axis=0)
    for i in range(stats_tab.shape[1]):
        stats_tab_norm[:, i] = stats_tab[:, i] / colsums[i]
    new_df = pd.DataFrame.from_dict(data={"n_neighbor": [neighbor], "min_dist": [dist]})
    stats_tab_norm_df = pd.DataFrame(stats_tab_norm, index=cluster_labeltypes, columns=by_types) * 100
    show_cluster_evaluation_matrix(cluster_labeltypes=cluster_labeltypes,by_types=by_types, stats_tab=stats_tab_norm_df, analyze_by=analyze_by, save_loc=str(results_loc) + '/distribution_of_variables_into_clusters.png', dist=dist, fmt='.1f')
    for label in stats_tab_norm_df.columns:
        places = stats_tab_norm_df[label].where(stats_tab_norm_df[label] > 0).dropna()
        data = {}
        for place in places.keys():
            data[place] = [stats_tab_norm_df[label][place]]
        df = pd.DataFrame.from_dict(data=data)
        new_df.insert(loc=new_df.shape[1], column=label, value=[df])
    return new_df

def more_evaluation(df,rusults_loc, n_neighbor,min_dist):
    labels=df.columns
    new_df = pd.DataFrame.from_dict(data={"n_neighbor": [n_neighbor], "min_dist": [min_dist]})
    for label in labels:
        places = df[label].where(df[label] > 0).dropna()
        data = {}
        for place in places.keys():
            data[place] = [df[label][place]]
        new_df.insert(loc=new_df.shape[1], column=label, value=[data])

    # distribution_df = pd.read_excel(rusults_loc / "distribution_of_variables_into_clusters.xlsx")
    # cluster_composition_percent = pd.read_excel(rusults_loc / "cluster_composition_percent.xlsx")
    # cluster_composition_absolute_numbers=pd.read_excel(rusults_loc/"cluster_composition_absolute_numbers.xlsx")

distinct_colors_20 = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                       '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                       '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                       '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                       '#ffffff', '#000000']

def generate_spectrogram_images(df, DATA, LABEL_COL):
    import os
    import io
    import pickle
    import librosa
    SPEC_COL = 'spectrograms'  # column name that contains the spectrograms
    ID_COL = 'callID'  # column name that contains call identifier (must be unique)

    OVERWRITE = False  # If there already exists an image_data.pkl, should it be overwritten? Default no

    # Spectrogramming parameters (needed for generating the images)

    FFT_WIN = 0.03
    FFT_HOP = 0.00375

    # Make sure the spectrogramming parameters are correct!
    # They are used to set the correct time and frequency axis labels for the spectrogram images.

    # If you are using bandpass-filtered spectrograms...
    if 'filtered' in SPEC_COL:
        # ...FMIN is set to LOWCUT, FMAX to HIGHCUT and N_MELS to N_MELS_FILTERED

        # FMIN = LOWCUT
        # FMAX = HIGHCUT
        # N_MELS = N_MELS_FILTERED
        FMIN = 0
        FMAX = 1000
        N_MELS = 40
    if OVERWRITE == False and os.path.isfile(os.path.join(os.path.sep, DATA, 'image_data.pkl')):
        print("File already exists. Overwrite is set to FALSE, so no new image_data will be generated.")

        # Double-ceck if image_data contains all the required calls
        with open(os.path.join(os.path.sep, DATA, 'image_data.pkl'), 'rb') as handle:
            image_data = pickle.load(handle)
        image_keys = list(image_data.keys())
        expected_keys = list(df[ID_COL])
        missing = list(set(expected_keys) - set(image_keys))

        if len(missing) > 0:
            print("BUT: The current image_data.pkl file doesn't seem to contain all calls that are in your dataframe!")

    else:
        image_data = {}
        for i, dat in enumerate(df.spectrograms):
            print('\rProcessing i:', i, '/', df.shape[0], end='')
            dat = np.asarray(df.iloc[i][SPEC_COL])
            sr = df.iloc[i]['samplerate_hz']
            plt.figure()
            librosa.display.specshow(dat, sr=sr, hop_length=int(FFT_HOP * sr), fmin=FMIN, fmax=FMAX, y_axis='mel',
                                     x_axis='s', cmap='inferno')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            byte_im = buf.getvalue()
            image_data[df.iloc[i][ID_COL]] = byte_im
            plt.close()

        # Store data (serialize)
        with open(os.path.join(os.path.sep, DATA, 'image_data.pkl'), 'wb') as handle:
            pickle.dump(image_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Load image data (deserialize)
    with open(os.path.join(os.path.sep, DATA, 'image_data.pkl'), 'rb') as handle:
        image_data = pickle.load(handle)
    if ID_COL not in df.columns:
        print("Missing identifier column: ", ID_COL)
        raise
    labeltypes = sorted(list(set(df[LABEL_COL])))

    if len(labeltypes) <= len(distinct_colors_20):
        color_dict = dict(zip(labeltypes, distinct_colors_20[0:len(labeltypes)]))
    else:
        # if > 20 different labels, some will have the same color
        distinct_colors = distinct_colors_20 * len(labeltypes)
        color_dict = dict(zip(labeltypes, distinct_colors[0:len(labeltypes)]))



def build_dictionary(df, labeltypes, AUDIO_COL):
    audio_dict = {} # dictionary that contains audio data for each labeltype
    sr_dict = {} # dictionary that contains samplerate data for each labeltype
    sub_df_dict = {} # dictionary that contains the dataframe for each labeltype
    for i,labeltype in enumerate(labeltypes):
        sub_df = df.loc[df.label==labeltype,:]
        sub_df_dict[i] = sub_df
        audio_dict[i] = sub_df[AUDIO_COL]
        sr_dict[i] = sub_df['samplerate_hz']

def build_traces(labeltypes, color_dict, LABEL_COL,sub_df_dict):
    traces = []
    for i,labeltype in enumerate(labeltypes):
        sub_df = sub_df_dict[i]
        trace = go.Scatter3d(x=sub_df.UMAP1,
                             y=sub_df.UMAP2,
                             z=sub_df.UMAP3,
                             mode='markers',
                             marker=dict(size=4,
                                         color=color_dict[labeltype],
                                         opacity=0.8),
                             name=labeltype,
                             hovertemplate = [x for x in sub_df[LABEL_COL]])
        traces.append(trace)

    layout = go.Layout(
        scene=go.layout.Scene(
            xaxis = go.layout.scene.XAxis(title='UMAP1'),
            yaxis = go.layout.scene.YAxis(title='UMAP2'),
            zaxis = go.layout.scene.ZAxis(title='UMAP3')),
            height = 1000,
            width = 1000)

    figure = go.Figure(data=traces, layout=layout)