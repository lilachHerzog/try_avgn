import pandas as pd
import numpy as np
import librosa
from tqdm.auto import tqdm
from pathlib2 import Path
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from avgn.utils.paths import ensure_dir


stars="********"

labels = {
    'B' : 1,
    'C' : 2,
    'CHUCKS' : 3,
    'COO' : 4,
    'CS' : 5,
    'GR' : 6,
    'GW' : 7,
    'GWB' : 8,
    'GWH' : 9,
    'H' : 10,
    'H3' : 11,
    'HC' : 12,
    'HTC' : 13,
    'HW' : 14,
    'SQ' : 15,
    'TRILL' : 16,
    'WB' : 17,
    'WHINE' : 18,
    'WHO' : 19,
    }

def make_dataset(dataset_loc, DATASET_ID, hparams):
    import json
    from avgn.dataset import DataSet
    print("make_dataset".center(40,"*"))
    subdir_list = list((dataset_loc).iterdir())
    for elem in subdir_list:
        print(elem.stem)
    dataset = DataSet(DATASET_ID, hparams = hparams, dataset_loc=dataset_loc, indv=dataset_loc.stem)
    # # segment an example
    # key='A4MERGED_file_4_153_(2013_05_27-15_51_24)_BK1MLX126127_GW_006'
    # segment_spec_custom(key, dataset.data_files[key], plot=True)
    # # segment part
    # segment_dataset(dataset.data_files[], 1, 3)
    # #segment all
    # segment_dataset(dataset.data_files[], -1, -1)
    print(json.dumps(dataset.sample_json, indent=4, default=str)[:400] + '...')
    print(len(dataset.data_files))
    return dataset

def create_dataframe(dataset, n_jobs, verbosity, FIGs_LOC):
    from avgn.signalprocessing.create_spectrogram_dataset import create_label_df
    print("create_dataframe".center(40,"*"))
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllable_dfs = parallel(
            delayed(create_label_df)(
                dataset.data_files[key].data,
                hparams=dataset.hparams,
                labels_to_retain=['labels', "sequence_num", 'length_s'],
                unit="notes",
                dict_features_to_retain = [],
                key = key,
            )
            for key in tqdm(dataset.data_files.keys())
        )
    syllable_df = pd.concat(syllable_dfs)
    len(syllable_df)
    syllable_df=grab_audio(dataset, syllable_df, n_jobs, verbosity)
    syllable_df['audio'] = [librosa.util.normalize(i) for i in syllable_df.audio.values]
    # print(syllable_df['audio'].values.size)
    # plot_example_audios(syllable_df, FIGs_LOC)
    return syllable_df

def remove_long_calls(syllable_df, out_loc, MIN_DUR = 0, MAX_DUR = 1):
    if 'length_s' not in syllable_df.columns:
        syllable_df['length_s'] = syllable_df.end_times.sub(syllable_df.start_times)
    df_sure_labels = syllable_df[syllable_df.labels.isin(labels)]
    out_loc=out_loc / "long_calls.txt"
    ensure_dir(out_loc)
    df_other_labels = syllable_df[syllable_df.labels.isin(labels)==False]
    f = open(out_loc, "w")
    f.write("recordings with the correct label but longer than 1 are: ")
    df_sure_labels[df_sure_labels["length_s"] > MAX_DUR].to_string(f, index=False, columns=["sequence_num", "labels", "length_s"])
    f.write("/n/nrecordings with the wrong label that are nontheless shorter than 1 are: ")
    df_other_labels[df_other_labels["length_s"] <= MAX_DUR].to_string(f, index=False, columns=["sequence_num", "labels", "length_s"])
    f.close()
    print("Dropped ", df_other_labels.shape[0], "rows not with the right label ")
    syllable_df = df_sure_labels
    print("Dropped ", syllable_df.loc[syllable_df['length_s'] < MIN_DUR, :].shape[0], "rows below ", MIN_DUR, "s (min_dur)")
    syllable_df = syllable_df.loc[syllable_df['length_s'] >= MIN_DUR, :]
    print("Dropped ", syllable_df.loc[syllable_df['length_s'] > MAX_DUR, :].shape[0], "rows above ", MAX_DUR, "s (max_dur)")
    syllable_df = syllable_df.loc[syllable_df['length_s'] <= MAX_DUR, :]
    return  syllable_df


def load_dataset(file_name, DATA_DIR, DATASET_ID, DT_ID):
    print(" load_dataset".center(40, "*"))
    syllable_df = pd.read_pickle(Path(DATA_DIR, DATASET_ID, DT_ID, file_name))
    print(syllable_df[:3])
    return syllable_df


class Segmentation_Parameters(object):
    """
    """

    def __init__(self, **kwargs):
        self.set_defaults()
        self.__dict__.update(kwargs)

    def set_defaults(self):
        self.n_fft = 1024
        self.hop_length_ms = 2
        self.win_length_ms = 4
        self.ref_level_db = 20
        self.pre = 0.97
        self.min_level_db = -60
        self.min_level_db_floor = -20
        self.db_delta = 5
        self.silence_threshold = 0.05
        # self.min_silence_for_spec=0.5
        self.min_silence_for_spec = 0
        self.max_vocal_for_spec = 0.5,
        self.min_syllable_length_s = 0.01
        self.butter_min = 500
        self.butter_max = 10000
        self.spectral_range = [500, 10000]

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


class General_Variables(object):
    """
        """
    from datetime import datetime

    def __init__(self, dt_id= datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), **kwargs):
        self.set_defaults(dt_id)
        self.__dict__.update(kwargs)
        self.hyrax_list.sort()
        self.print_vars()
        # ensure_dir(self.dataset_loc)

    def set_defaults(self, dt_id):
        from avgn.utils.hparams import HParams
        self.DATA_DIR = "/usr/local/lib/python3.9/site-packages/avgn_paper/data"
        self.DATASET_ID = 'hyrax_by_indvs'
        self.raw_loc= Path(self.DATA_DIR, 'raw', self.DATASET_ID)
        raw_labelled_wav=Path(self.raw_loc, "labelled")
        self.hyrax_list = list((raw_labelled_wav).glob("**"))
        self.hyrax_list.remove(raw_labelled_wav)
        # self.wav_list = list((raw_labelled_wav).glob("**/*.wav"))
        self.DT_ID = dt_id
        self.dataset_loc = Path(self.DATA_DIR, "processed", self.DATASET_ID, self.DT_ID)
        self.labels = ['B', 'C', 'CHUCKS', 'COO', 'CS', 'GR', 'GW', 'GWB', 'GWH', 'H', 'H3', 'HC', 'HTC', 'HW', 'SQ', 'TRILL', 'WB', 'WHINE', 'WHO']
        self.hparams = HParams(
            num_mel_bins=64,
            mel_lower_edge_hertz=500,
            mel_upper_edge_hertz=9999,
            butter_lowcut=500,
            butter_highcut=9999,
            ref_level_db=20,
            min_level_db=-30,
            mask_spec=True,
            win_length_ms=10,
            hop_length_ms=2,
            nex=-1,
            n_jobs=-1,
            verbosity=1,
        )
        self.seg_params = Segmentation_Parameters()

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
    def print_vars(self):
        print("GENERAL: ", self.DATA_DIR,
              "\nDATA_DIR: ", self.DATA_DIR,
              "\nDATASET_ID: ", self.DATASET_ID,
              "\nhyrax_list: ", self.hyrax_list,
              "\nDT_ID: ", self.DT_ID)


class Indv_Variables(object):
    """
        """

    def __init__(self, indv, dataset_loc, raw_loc, **kwargs):
        self.set_defaults(indv, dataset_loc, raw_loc)
        self.__dict__.update(kwargs)
        # ensure_dir(self.this_dataset)

    def set_defaults(self, indv,dataset_loc, raw_loc):
        self.indv = indv
        self.this_dataset = Path(dataset_loc, self.indv)
        self.results_loc=self.this_dataset / "results"
        self.save_loc=self.this_dataset / "different_dfs"
        self.wav_list = list((raw_loc).glob(indv+"/*.wav"))

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


def segment_spec_custom(key, df, RESULTS_LOC=None, save=False, plot=False, variables=General_Variables()):
    if RESULTS_LOC is None:
        RESULTS_LOC = variables.dataset_loc/"results"
    seg_params=variables.seg_params
    from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation
    from avgn.signalprocessing.filtering import butter_bandpass_filter
    from avgn.utils.audio import load_wav
    from avgn.utils.json import NoIndent, NoIndentEncoder
    import json
    from vocalseg.dynamic_thresholding import plot_segmentations
    print(stars+"\nsegment_spec_custom\n"+stars)
    # load wav
    rate, data = load_wav(df.data["wav_loc"])
    # filter data
    data = butter_bandpass_filter(data, seg_params.butter_min, seg_params.butter_max, rate)

    # segment
    results = dynamic_threshold_segmentation(
        data,
        rate,
        n_fft=seg_params.n_fft,
        hop_length_ms=seg_params.hop_length_ms,
        win_length_ms=seg_params.win_length_ms,
        min_level_db_floor=seg_params.min_level_db_floor,
        db_delta=seg_params.db_delta,
        ref_level_db=seg_params.ref_level_db,
        pre=seg_params.pre,
        min_silence_for_spec=seg_params.min_silence_for_spec,
        max_vocal_for_spec=seg_params.max_vocal_for_spec,
        min_level_db=seg_params.min_level_db,
        silence_threshold=seg_params.silence_threshold,
        verbose=True,
        min_syllable_length_s=seg_params.min_syllable_length_s,
        spectral_range=seg_params.spectral_range,
    )
    if results is None:
        return
    if plot:
        plot_segmentations(
            results["spec"],
            results["vocal_envelope"],
            results["onsets"],
            results["offsets"],
            seg_params.hop_length_ms,
            rate,
            figsize=(100, 5),
        )
        plt.show()
        ensure_dir(RESULTS_LOC)
        fig_out = Path(RESULTS_LOC, "results.png")
        ensure_dir(fig_out)
        plt.savefig(fig_out)

    # save the results
    json_out = Path(
        variables.DATA_DIR
        , "processed"
        , (variables.DATASET_ID + "_segmented")
        , variables.DT_ID
        , "JSON"
        , (key + ".JSON")
    )

    json_dict = df.data.copy()

    json_dict["indvs"][list(df.data["indvs"].keys())[0]]["syllables"] = {
        "start_times": NoIndent(list(results["onsets"])),
        "end_times": NoIndent(list(results["offsets"])),
    }

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)
    # save json
    if save:
        ensure_dir(json_out.as_posix())
        print(json_txt, file=open(json_out.as_posix(), "w"))

    print(json_txt)

    return results

def segment_dataset(dataset, nex, n_jobs=-1, verbose=11):
    import joblib
    print(stars+"\nsegment_dataset\n"+stars)
    indvs = np.array(['_'.join(list(i)) for i in dataset.json_indv])
    np.unique(indvs), np.unique(indvs)
    for indv in tqdm(np.unique(indvs), desc="individuals"):
        print(indv)
        indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv][:nex]

        joblib.Parallel(n_jobs, verbose)(
                joblib.delayed(segment_spec_custom)(key, dataset.data_files[key], plot=True)
                     for key in tqdm(indv_keys, desc="files", leave=False)
            )


def grab_audio(dataset, syllable_df, n_jobs, verbosity):
    from avgn.signalprocessing.create_spectrogram_dataset import get_row_audio
    print("grab_audio".center(40, "*"))
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllable_dfs = parallel(
            delayed(get_row_audio)(
                syllable_df[syllable_df.key == key],
                dataset.data_files[key].data['wav_loc'],
                dataset.hparams
            )
            for key in tqdm(syllable_df.key.unique())
        )
    syllable_df = pd.concat(syllable_dfs)
    len(syllable_df)
        # this is just getting rid of syllables that are zero seconds long, which be the product of errors in segmentation
    df_mask  = np.array([len(i) > 0 for i in tqdm(syllable_df.audio.values)])
    syllable_df = syllable_df[np.array(df_mask)]
    # syllable_df['audio'] = [librosa.util.normalize(i) for i in syllable_df.audio.values]
    print(syllable_df[:3])
    return syllable_df


# plot some example audio
def plot_example_audios(syllable_df, FIGs_LOC):
    print("********\nplot_example_audios\n********")
    nrows = 5
    ncols = 10
    zoom = 2
    fig, axs = plt.subplots(ncols=ncols, nrows = nrows,figsize = (ncols*zoom, nrows+zoom/1.5))
    for i, syll in tqdm(enumerate(syllable_df['audio'].values), total = nrows*ncols):
        ax = axs.flatten()[i]
        ax.plot(syll)
        fig_out = Path(FIGs_LOC, "example_audios.png")
        ensure_dir(fig_out)
        plt.savefig(fig_out)

        if i == nrows*ncols -1:
            break

def excel_to_json(dataset_loc, DT_ID, excel_loc="/usr/local/lib/python3.9/site-packages/avgn_paper/data/raw/hyrax_females/excel_data"):
    import re
    excel_list = list(Path(excel_loc).glob("*.xlsx"))
    dataset_loc=Path("/usr/local/lib/python3.9/site-packages/avgn_paper/data/processed/hyrax_by_indvs_new")

    num_of_files = {}
    years=[]
    weird_names=[]
    for excel in excel_list[0:-1]:
        excel_Data = pd.read_excel(excel, header=0)
        df = pd.DataFrame(excel_Data,columns=excel_Data.columns)
        indv = Path(excel).stem
        df = df.drop(columns=[x for x in df.columns if re.search('Unnamed: ', x)], axis=1)
        df = df.dropna(subset=['end time'])
        df['file no'] = [Path(str(x).replace("\\", "/")).stem + ".wav" for x in df['file no']]
        df['WaveFileName']= [Path(x).stem + ".wav" for x in df['file no']]
        weird_name, year= change_names(df['WaveFileName'])
        weird_names.append(weird_name)
        years.append(year)# text = [re.split(spliter[1], re.split(spliter[0], Path(x).stem)[-1])[0] for x in df['WaveFileName']]
        print(indv, year[0], weird_name[0])
        name=df.loc[0,'WaveFileName']
        for i in df.index:
            if str(df.loc[i,'WaveFileName']) != "nan.wav":
                name = df.loc[i,'WaveFileName']
                this_name = name
            else:
                df.loc[i,'WaveFileName'] = "nan"
            df.loc[i,'WaveFileName'] = name
            if weird_name.size == 1:
                df.loc[i, 'WaveFileName']=str(df.loc[i, 'WaveFileName']).replace("_"+weird_name[0],"")
            if year.size == 1:
                df.loc[i, 'WaveFileName']=str(df.loc[i, 'WaveFileName']).replace(year[0]+"-","")
            else:
                print("the years of ",indv,"are\n",year)
            num_of_files[indv] = df["WaveFileName"].unique()
        # json_txt = convert_to_json(df, indv, excel_loc.parent, dataset_loc, DT_ID="2022-03-10_14-01-53")
        # convert_to_json(df, indv, Path(excel_loc).parent, dataset_loc, DT_ID)
    write_names_excel(num_of_files)

def change_names(names):
    import re
    spliter = ["((file_.*(.*_)\())", "((-.*\)_))"]
    for i in range(3):
        spliter[1] = "((.*_){"+str(i)+"}(-.*\)_))"
        year = np.unique([re.split(spliter[1], re.split(spliter[0], Path(x).stem)[-1])[0] for x in names])
        year = find_names(year)
        if year.size == 1:
            break
    weird_name = np.unique(
        [re.split(spliter[1], re.split(spliter[0], Path(x).stem)[-1])[-1] for x in names])
    weird_name = find_names(weird_name)
    return weird_name, year

def find_names(names):
    to_delete=[]
    for j in range(len(names)-2):
        for i in range(j+1, len(names)):
            name=names[i]
            if names[j] in name or name in ["nan", "file no", "file no.", "filee no", "break",
                                                 "peak ampl(meanentire)"]:
                print("deleting= ", names[i])
                to_delete.append(i)
        for i,x in enumerate(to_delete):
            names=np.delete(names, x-i)
        to_delete=[]
    return names

def write_names_excel(num_of_files):
    import xlsxwriter
    xlx_loc=Path('/usr/local/lib/python3.9/site-packages/avgn_paper/data/processed/hyrax_by_indvs_new/file_names.xlsx')
    workbook = xlsxwriter.Workbook(xlx_loc)
    worksheet = workbook.add_worksheet()
    row = 0
    column = 0
    for indv in num_of_files.keys():
        # write operation perform
        worksheet.write(row, column, indv)
        row += 1
        worksheet.write(row, column, str(len(num_of_files[indv])))
        for file in num_of_files[indv]:
            row += 1
            worksheet.write(row, column, str(file))
        # incrementing the value of row by one
        # with each iterations.
        column += 1
        row = 0
    workbook.close()


def convert_to_json(df, indv, RAW_DATASET_LOC, dataset_loc, DT_ID):
    DATASET_ID = dataset_loc.stem
    DATA_DIR = dataset_loc.parent
    from avgn.utils.audio import get_samplerate
    import librosa
    from avgn.utils.json import NoIndent, NoIndentEncoder
    import json
    wfn_df_prev=df["WaveFileName"][0]
    for wfn in tqdm(df["WaveFileName"].unique(), leave=False):
        wfn_df = df[df["WaveFileName"] == wfn]
        wav_loc = RAW_DATASET_LOC / "Wave" / indv  / wfn
        sr = get_samplerate(wav_loc.as_posix())
        wav_duration = librosa.get_duration(filename=wav_loc)
        json_dict = {}
        json_dict["species"] = " Procavia capensis"
        json_dict["common_name"] = "Rock hyrax"
        json_dict["wav_loc"] = wav_loc.as_posix()
        json_dict["samplerate_hz"] = sr
        json_dict["length_s"] = wav_duration
        seq_df = pd.DataFrame(
            (
                [
                    [
                        list(np.repeat(sequence_num, len(row.NotePositions))),
                        list(row.NoteLabels),
                        np.array(
                            (np.array(row.NotePositions).astype("int") + int(row.Position))
                            / sr
                        ).astype("float64"),
                        np.array(
                            (
                                    np.array(row.start_times).astype("int")
                                    + np.array(row.end_times-row.start_times).astype("int")
                                    + int(row.Position)
                            )
                            / sr
                        ).astype("float64"),
                    ]
                    for sequence_num, (idx, row) in enumerate(wfn_df.iterrows())
                ]
            ),
            columns=["sequence_num", "labels", "start_times", "end_times"],
        )

        # add syllable information
        json_dict["indvs"] = {
            indv: {
                "notes": {
                    "start_times": NoIndent(
                        list(np.concatenate(seq_df.start_times.values))
                    ),
                    "end_times": NoIndent(list(np.concatenate(seq_df.end_times.values))),
                    "labels": NoIndent(list(np.concatenate(seq_df.labels.values))),
                    "sequence_num": NoIndent(
                        [int(i) for i in np.concatenate(seq_df.sequence_num.values)]
                    ),
                }
            }
        }

        # dump dict into json format
        json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

        wav_stem = indv + "_" + wfn.split(".")[0]
        json_out = (
                DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
        )

        # save json
        ensure_dir(json_out.as_posix())
        print(json_txt, file=open(json_out.as_posix(), "w"))
        return json_txt

        # print("leave: ",name)
song_df = pd.DataFrame(
        columns=[
            "start_times",
            "end_times",
            "labels",
            "sequence_num",
            "indvs",
            "indvi",
            "key",
            "NoteLabels",
            'duration',
            'interval',
            'break',
            'peak freq(meanentire)',
            'peak ampl(meanentire)',
        ]
    )

# song_df
# get_info(name, df)
# for name in names:
#     start_times = np.fromfunction()
#     end_times = []
#     for i, df_name in enumerate(df["file no"]):
#         if df_name == name:
#             start_times
#
#     [x for x in df["file no" == name]["start time"]]
#     [x for x in df["file no" == name]["end time"]]
# #
# # loop through XML annotation files
# for bird_loc in tqdm(annotation_files):
#     # grab the
#     bird_xml = xml.etree.ElementTree.parse(bird_loc).getroot()
#     bird = bird_loc.parent.stem
#     # loop through each "sequence" in the datset (corresponding to a bout)
#     for element in tqdm(list(bird_xml), leave=False):
#         if element.tag == "Sequence":
#             notePositions = []
#             noteLengths = []
#             noteLabels = []
#             # get the metadata for that sequence
#             for seq_element in list(element):
#                 if seq_element.tag == "Position":
#                     position = seq_element.text
#                 elif seq_element.tag == "Length":
#                     length = seq_element.text
#                 elif seq_element.tag == "WaveFileName":
#                     WaveFileName = seq_element.text
#                 elif seq_element.tag == "NumNote":
#                     NumNote = seq_element.text
#                 # get the metadata for the note
#                 elif seq_element.tag == "Note":
#                     for note_element in list(seq_element):
#                         if note_element.tag == "Label":
#                             noteLabels.append(note_element.text)
#                         elif note_element.tag == "Position":
#                             notePositions.append(note_element.text)
#                         elif note_element.tag == "Length":
#                             noteLengths.append(note_element.text)
#             # add to the pandas dataframe
#             song_df.loc[len(song_df)] = [
#                 bird,
#                 WaveFileName,
#                 position,
#                 length,
#                 NumNote,
#                 notePositions,
#                 noteLengths,
#                 noteLabels,
#             ]
#
