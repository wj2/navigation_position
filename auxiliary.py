import os
import scipy.ndimage as snd
import numpy as np
import pandas as pd

import general.utility as u
import general.data_io as gio


BASEFOLDER = "../data/navigation_position/"
FIGFOLDER = "navigation_position/figs/"
session_template = "(?P<animal>[a-zA-Z]+)_(?P<date>[0-9]+)"


def load_sessions(folder=BASEFOLDER, correct_only=False, uninstructed_only=True):
    data = gio.Dataset.from_readfunc(
        load_gulli_hashim_data_folder,
        folder,
    )
    data_use = mask_completed_trials(data, correct_only=correct_only)
    if uninstructed_only:
        data_use = mask_uninstructed_trials(data_use)
    return data_use


def load_session_files(
    folder,
    spikes="spike_times.pkl",
    bhv="[0-9]+_[a-z]+_VR_behave\\.pkl",
    good_neurs="good_neurons.pkl",
):
    out_dict = {}
    out_dict["spikes"] = pd.read_pickle(open(os.path.join(folder, spikes), "rb"))
    bhv_fl = u.get_matching_files(folder, bhv)[0]
    out_dict["bhv"] = pd.read_pickle(open(bhv_fl, "rb"))

    out_dict["good_neurs"] = pd.read_pickle(
        open(os.path.join(folder, good_neurs), "rb")
    )
    return out_dict


def organize_spikes(spikes, neur_info):
    neur_regions = tuple(neur_info["region"])
    n_trls = len(spikes)
    neur_regions_all = (neur_regions,) * n_trls
    spike_times = []
    for i, (_, row) in enumerate(spikes.iterrows()):
        spk_times_i = np.zeros(len(row), dtype=object)
        for j, r_ij in enumerate(row.to_numpy()):
            spk_times_i[j] = np.array(r_ij)
        spike_times.append(spk_times_i)
    return neur_regions_all, spike_times


timing_rename_dict = {
    "BehavioralCodes.TrialEpochTimes.AutomaticRotation_1.0": "pre_rotation_start",
    "BehavioralCodes.TrialEpochTimes.AutomaticRotation_1.1": "pre_rotation_end",
    "BehavioralCodes.TrialEpochTimes.CuedNavigation.0": "nav_start",
    "BehavioralCodes.TrialEpochTimes.CuedNavigation.1": "nav_end",
    "BehavioralCodes.TrialEpochTimes.AutomaticRotation_2.0": "post_rotation_start",
    "BehavioralCodes.TrialEpochTimes.AutomaticRotation_2.1": "post_rotation_end",
    "BehavioralCodes.TrialEpochTimes.ChoiceLocationApproach.0": "choice_approach_start",
    "BehavioralCodes.TrialEpochTimes.ChoiceLocationApproach.1": "choice_approach_end",
    "BehavioralCodes.TrialEpochTimes.PreChoiceDelay.0": "pre_choice_start",
    "BehavioralCodes.TrialEpochTimes.PreChoiceDelay.1": "pre_choice_end",
    "BehavioralCodes.TrialEpochTimes.Choice.0": "choice_start",
    "BehavioralCodes.TrialEpochTimes.Choice.1": "choice_end",
    (
        "BehavioralCodes.TrialEpochTimes.ObjectApproach.0",
        "BehavioralCodes.TrialEpochTimes.Response.0"
    ): "approach_start",
    (
        "BehavioralCodes.TrialEpochTimes.ObjectApproach.1",
        "BehavioralCodes.TrialEpochTimes.Response.1"
    ): "approach_end",
}
info_rename_dict = {
    "Float0_IsEast": "IsEast",
    "Float1_IsInstructed": "IsInstructed",
    "Float2_isTargetRightSide": "target_right",
    "Float5_IsTestCondition": "generalization_trial",
    "Float8_IsNorth": "IsNorth",
    "UserVars.ChoseWhite": "chose_white",
    "UserVars.ChoseBlue": "chose_blue",
    "UserVars.ChosePink": "chose_pink",
    "UserVars.ChoseRight": "chose_right",
    "UserVars.RestructuredVRData.Rotation": "rotation_tc",
    "UserVars.RestructuredVRData.Position_X": "pos_x",
    "UserVars.RestructuredVRData.Position_Y": "pos_z",
    "UserVars.RestructuredVRData.Position_Z": "pos_y",
    "RestructuredAnalog.Eye.0": "eye_x",
    "RestructuredAnalog.Eye.1": "eye_y",
    "UserVars.VR_Trial.Fix_Position_World.x": "eye_world_x",
    "UserVars.VR_Trial.Fix_Position_World.y": "eye_world_z",
    "UserVars.VR_Trial.Fix_Position_World.z": "eye_world_y",
}


def find_crossings(trl_pos, border=500, thresh=2):
    cross_times = np.zeros(len(trl_pos), dtype=object)
    cross_dir = np.zeros(len(trl_pos), dtype=object)
    for i, trl in enumerate(trl_pos.to_numpy()):
        ms_times = np.arange(len(trl))
        relative_border = trl - border
        near_border = np.abs(relative_border) < thresh
        labels, num_objs = snd.label(near_border)
        slices = snd.find_objects(labels, num_objs)
        cross_times_i = []
        cross_dir_i = []
        for sli in slices:
            event_pos = relative_border[sli]
            s_pre = np.sign(event_pos[0])
            s_post = np.sign(event_pos[-1])
            cross = s_pre * s_post
            direction = s_pre < s_post
            cross_ind = np.argmin(near_border[sli])
            time_i = ms_times[sli][cross_ind]
            if cross:
                cross_times_i.append(time_i)
                cross_dir_i.append(direction)
        cross_times[i] = cross_times_i
        cross_dir[i] = cross_dir_i
    return cross_times, cross_dir


def get_relevant_crossing(crossings, crossing_dirs, decision_times):
    rel_time = np.zeros(len(crossings))
    rel_time[:] = np.nan
    rel_dir = np.zeros(len(crossings))
    rel_dir[:] = np.nan
    for i, crosses_i in enumerate(crossings.to_numpy()):
        crosses_i = np.array(crosses_i)
        mask_i = crosses_i < decision_times.iloc[i]
        crosses_i = crosses_i[mask_i]
        crossing_dir_i = np.array(crossing_dirs[i])[mask_i]
        if len(crosses_i) > 0:
            rel_time[i] = crosses_i[-1]
            rel_dir[i] = crossing_dir_i[-1]
    return rel_time, rel_dir


def rename_fields(df, *dicts):
    full_dict = {}
    list(full_dict.update(d) for d in dicts)
    for old_name, new_name in full_dict.items():
        if u.check_list(old_name):
            present = list(on in df.columns for on in old_name)
            inds = np.where(present)[0]
            if len(inds) == 0:
                raise IOError("column {} is missing".format(old_name))
            old_name = old_name[inds[0]]
        if old_name in df.columns:
            rename = df[old_name]
        else:
            rename = np.ones(len(df)) * np.nan
            print("no {} in the columns".format(old_name))
        df[new_name] = rename
    return df


def mask_completed_trials(
    data,
    correct_only=False,
    completed_field="completed_trial",
    correct_field="correct_trial",
):
    mask = data[completed_field]
    if correct_only:
        mask = mask.rs_and(data[correct_field])
    return data.mask(mask)


def mask_uninstructed_trials(
    data,
    instructed_field="IsInstructed",
):
    mask = data[instructed_field] == 0
    return data.mask(mask)


def extract_time_field(data, t_field, extract_field):
    times = data[t_field].to_numpy()
    ef_data = data[extract_field].to_numpy()
    fvs = np.zeros(len(data))
    for i, row in enumerate(ef_data):
        if np.isnan(times[i]):
            val = np.nan
        else:
            val = row[int(times[i])]
        fvs[i] = val
    return fvs


def discretize_rotation(rots, cents=(0, 90, 180, 270)):
    rots = np.expand_dims(rots.to_numpy(), 1)
    cents = np.expand_dims(cents, 0)
    dists = u.normalize_periodic_range(rots - cents, radians=False)
    bins = np.argmin(np.abs(dists), axis=1)
    return bins


def get_last_choices(choices, mask=None, n_back=1):
    if mask is None:
        mask = np.ones(len(choices), dtype=bool)
    last_choice = np.zeros(len(choices))
    last_choice[:] = np.nan
    inds = np.where(mask)[0]
    for i, ind1 in enumerate(inds[n_back:]):
        # ind0 is ith correct decision
        # ind1 is i + nth correct decision
        # fill from i + n - 1 to i + n
        # with ith decision
        ind0 = inds[i]
        ind_last = inds[n_back + i - 1]
        last_choice[ind_last + 1:ind1 + 1] = choices[ind0]
    return last_choice


date_task_dict = {
    "20231223": "IsEast",
    "20240112": "IsNorth",
    "20240115": "IsNorth",
}


def load_gulli_hashim_data_folder(
    folder,
    session_template=session_template,
    max_files=np.inf,
    exclude_last_n_trls=None,
    rename_dicts=None,
    load_only_nth_files=None,
    date_task_dict=date_task_dict,
):
    if rename_dicts is None:
        rename_dicts = (timing_rename_dict, info_rename_dict)
    dates = []
    monkeys = []
    n_neurs = []
    datas = []
    files_loaded = 0
    folder_gen = u.load_folder_regex_generator(
        folder,
        session_template,
        load_func=load_session_files,
        open_file=False,
        load_only_nth_files=load_only_nth_files,
    )
    for fl, fl_info, data_fl in folder_gen:
        dates.append(fl_info["date"])
        monkeys.append(fl_info["animal"])
        n_neurs.append(len(data_fl["good_neurs"]))
        neur_regions, spikes = organize_spikes(
            data_fl["spikes"],
            data_fl["good_neurs"],
        )
        data_all = data_fl["bhv"]["data_frame"]
        if len(data_all) > len(spikes):
            diff = len(data_all) - len(spikes)
            print(
                "difference in length between data ({}) and spikes ({})"
                "in file {}".format(len(data_all), len(spikes), fl)
            )
            data_all = data_all[:-diff].copy()
        data_all["spikeTimes"] = spikes
        data_all["neur_regions"] = neur_regions
        data_all["completed_trial"] = np.isin(data_all["TrialError"], (0, 6))
        data_all["correct_trial"] = data_all["TrialError"] == 0
        data_all = rename_fields(data_all, *rename_dicts)

        task_key = date_task_dict.get(fl_info["date"])
        data_all["relevant_position"] = data_all[task_key]
        data_all["white_right"] = np.logical_or(
            np.logical_and(
                data_all[task_key] == 1,
                data_all["target_right"] == 1,
            ),
            np.logical_and(
                data_all[task_key] == 0,
                data_all["target_right"] == 0,
            ),
        )
        data_all["pink_right"] = np.logical_or(
            np.logical_and(
                data_all[task_key] == 1,
                data_all["target_right"] == 1,
            ),
            np.logical_and(
                data_all[task_key] == 0,
                data_all["target_right"] == 0,
            ),
        )
        data_all["pre_choice_rotation"] = extract_time_field(
            data_all,
            "post_rotation_end",
            "rotation_tc",
        )
        data_all["choice_rotation"] = discretize_rotation(
            data_all["pre_choice_rotation"],
        )
        data_all["last_choice_white"] = get_last_choices(data_all["chose_white"])
        data_all["last_correct_choice_white"] = get_last_choices(
            data_all["chose_white"],
            mask=data_all["correct_trial"] == 1,
        )
        data_all["last_completed_choice_white"] = get_last_choices(
            data_all["chose_white"],
            mask=data_all["completed_trial"] == 1,
        )                                    
        
        out = find_crossings(data_all["pos_x"])
        data_all["border_crossing_x"], data_all["border_crossing_x_dir"] = out
        
        out = find_crossings(data_all["pos_y"])
        data_all["border_crossing_y"], data_all["border_crossing_y_dir"] = out
        
        out = get_relevant_crossing(
            data_all["border_crossing_x"],
            data_all["border_crossing_x_dir"],
            data_all["approach_start"],
        )
        data_all["relevant_crossing_x"], data_all["relevant_crossing_x_dir"] = out

        out = get_relevant_crossing(
            data_all["border_crossing_y"],
            data_all["border_crossing_y_dir"],
            data_all["approach_start"],
        )
        data_all["relevant_crossing_y"], data_all["relevant_crossing_y_dir"] = out
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
