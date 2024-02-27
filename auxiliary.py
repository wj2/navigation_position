
import os
import scipy.io as sio
import skimage.io as skio
import re
import numpy as np
import pandas as pd
import pickle

import general.utility as u
import general.data_io as gio


BASEFOLDER = "../data/navigation_position/"

session_template = "(?P<animal>[a-zA-Z]+)_(?P<date>[0-9]+)"

def load_session_files(
        folder,
        spikes="spike_times_df.pkl",
        bhv="behavior_df.pkl",
        good_neurs="good_neurons.pkl",
):
    out_dict = {}
    out_dict["spikes"] = pickle.load(open(os.path.join(folder, spikes), "rb"))
    out_dict["bhv"] = pickle.load(open(os.path.join(folder, bhv), "rb"))
    out_dict["good_neurs"] = pickle.load(open(os.path.join(folder, good_neurs), "rb"))
    return out_dict


def organize_spikes(spikes, neur_info):
    neur_regions = tuple(neur_info["region"])
    n_trls = len(spikes)
    neur_regions_all = (neur_regions,) * n_trls
    spike_times = []
    for _, row in spikes.iterrows():
        spk_times_i = np.zeros(len(row), dtype=object)
        spk_times_i = row.to_numpy()
        spike_times.append(spk_times_i)
    return neur_regions_all, spike_times


def load_gulli_hashim_data_folder(
        folder,
        session_template=session_template,
        max_files=np.inf,
):
    dates = []
    monkeys = []
    n_neurs = []
    datas = []
    files_loaded = 0
    folder_gen = u.load_folder_regex_generator(
        folder, session_template, load_func=load_session_files, open_file=False,
    )
    for fl, fl_info, data_fl in folder_gen:
        dates.append(fl_info["date"])
        monkeys.append(fl_info["animal"])
        n_neurs.append(len(data_fl["good_neurs"]))
        neur_regions, spikes = organize_spikes(
            data_fl["spikes"], data_fl["good_neurs"],
        )
        data_all = data_fl["bhv"]
        data_all["spikeTimes"] = spikes
        data_all["neur_regions"] = neur_regions
        datas.append(data_all)

        files_loaded += 1
        if files_loaded > max_files:
            break
    super_dict = dict(
        date=dates,
        animal=monkeys,
        data=datas,
        n_neurs=n_neurs,
    )
    return super_dict
