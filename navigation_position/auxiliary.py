import os
import scipy.ndimage as snd
import numpy as np
import pandas as pd
import skimage as skimg

import general.utility as u
import general.data_io as gio


BASEFOLDER = "../data/navigation_position/"
FIGFOLDER = "navigation_position/figs/"
session_template = "(?P<animal>[a-zA-Z]+)_(?P<date>[0-9]+)"
dated_session_template = "(?P<animal>[a-zA-Z]+)_(?P<date>{date})"


REGIONLIST = (None, ("DLPFCv", "DLPFCd"), ("HPC",), ("PMd",))


def load_sessions(
    folder=BASEFOLDER, correct_only=False, uninstructed_only=True, **kwargs
):
    data = gio.Dataset.from_readfunc(
        load_gulli_hashim_data_folder,
        folder,
        **kwargs,
    )
    data_use = mask_completed_trials(data, correct_only=correct_only)
    if uninstructed_only:
        data_use = mask_uninstructed_trials(data_use)
    return data_use


def get_date_list(folder=BASEFOLDER, template=session_template):
    gen = u.folder_regex_generator(folder, template)
    dates = []
    for _, f_info in gen:
        dates.append(f_info["date"])
    return dates


def load_dated_session(
    date, dated_session_template=dated_session_template, **kwargs,
):
    use_template = dated_session_template.format(date=date)
    data = load_sessions(session_template=use_template, **kwargs)
    return data


def load_session_files(
    folder,
    spikes="spike_times.pkl",
    bhv="[0-9]+_[a-z]+_(VR|airpuff)_behave\\.pkl",
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
        "BehavioralCodes.TrialEpochTimes.Response.0",
    ): "approach_start",
    (
        "BehavioralCodes.TrialEpochTimes.ObjectApproach.1",
        "BehavioralCodes.TrialEpochTimes.Response.1",
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
    "UserVars.VR_Trial.Target_Positions.x": "targ_x",
    "UserVars.VR_Trial.Target_Positions.z": "targ_y",
    "UserVars.VR_Trial.Distractor_Positions.x": "dist_x",
    "UserVars.VR_Trial.Distractor_Positions.z": "dist_y",
    "RestructuredAnalog.Eye.0": "eye_x",
    "RestructuredAnalog.Eye.1": "eye_y",
    "UserVars.VR_Trial.Fix_Position_World.x": "eye_world_x",
    "UserVars.VR_Trial.Fix_Position_World.y": "eye_world_z",
    "UserVars.VR_Trial.Fix_Position_World.z": "eye_world_y",
    "UserVars.VR_Trial.Additional_Positions.x": "waypoints_x",
    "UserVars.VR_Trial.Additional_Positions.y": "waypoints_z",
    "UserVars.VR_Trial.Additional_Positions.z": "waypoints_y",
}


saccade_start_time = "SaccadeStruct.StartTime"
saccade_end_time = "SaccadeStruct.EndTime"
saccade_starts = ("SaccadeStruct.StartPointX", "SaccadeStruct.StartPointY")
saccade_ends = ("SaccadeStruct.EndPointX", "SaccadeStruct.EndPointY")


def get_saccade_info(
    data,
    tzf=None,
    tbeg=0,
    tend=None,
    saccade_start_time=saccade_start_time,
    saccade_end_time=saccade_end_time,
    saccade_starts=saccade_starts,
    saccade_ends=saccade_ends,
    sub_tzf=True,
):
    out_start = []
    out_end = []
    out_start_times = []
    out_end_times = []
    if tzf is not None:
        tzs = data[tzf]
    saccade_start_times = data[saccade_start_time]
    saccade_end_times = data[saccade_end_time]
    starts = data[list(saccade_starts)]
    ends = data[list(saccade_ends)]
    if tend is None:
        tend = np.inf
    for i, st_i in enumerate(saccade_start_times):
        ends_i = []
        starts_i = []
        start_times_i = []
        end_times_i = []
        for j, st_ij in enumerate(st_i):
            st_ij = np.array(st_ij)
            et_ij = np.array(saccade_end_times[i].iloc[j])
            if tzf is not None:
                st_ij_mask = st_ij - tzs[i].iloc[j]
                et_ij_mask = et_ij - tzs[i].iloc[j]
            if sub_tzf:
                st_ij_use = st_ij_mask
                et_ij_use = et_ij_mask
            else:
                st_ij_use = st_ij
                et_ij_use = et_ij
            t_mask = np.logical_and(st_ij_mask >= tbeg, st_ij_mask < tend)
            start_times_i.append(st_ij_use[t_mask])
            end_times_i.append(et_ij_use[t_mask])

            s = np.stack(starts[i].iloc[j].to_numpy(), axis=1)
            starts_i.append(s[t_mask])
            e = np.stack(ends[i].iloc[j].to_numpy(), axis=1)
            ends_i.append(e[t_mask])
        out_start.append(starts_i)
        out_end.append(ends_i)
        out_start_times.append(start_times_i)
        out_end_times.append(end_times_i)
    return out_start_times, out_end_times, out_start, out_end
            

def combine_temporal_keys(data, keys, filt_func=None, tzf=None, tbeg=0, tend=None):
    tks = data[list(keys)]
    out = []
    if tzf is not None:
        tzs = data[tzf]
    for i, tk in enumerate(tks):
        tk_arr = tk.to_numpy()
        out_sess = []
        for j, tk_ij in enumerate(tk_arr):
            if tzf is not None:
                tz_ij = tzs[i].iloc[j]
                if tend is None:
                    end = None
                else:
                    end = int(tz_ij + tend)
                sl = slice(int(tz_ij + tbeg), end)
            else:
                sl = slice(0, None)
            if u.check_list(tk_ij[0]):
                x = np.stack(tk_ij, axis=1)
            else:
                x = tk_ij[None]
            x = x[sl]
            if filt_func is not None:
                x = filt_func(x)
            ts = np.arange(len(x)) + tbeg
            out_sess.append((x, ts))
        out.append(out_sess)
    return out


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
                print("column {} is missing".format(old_name))
            else:
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
    targ=0,
):
    mask = data[instructed_field] == targ
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


def make_unique_conds(
    data,
    keys=("choice_pos_x", "choice_pos_y", "pre_choice_rotation"),
    rounds=(10, 10, 15),
    periodic=(False, False, True),
    rot_only=False,
):
    if rot_only:
        keys = ("pre_choice_rotation",)
        rounds = (15,)
        periodic = (True,)
    out = []
    entries = data[list(keys)]
    for ent in entries:
        arr = ent.to_numpy()
        comb = []
        for j in range(len(keys)):
            rounded = round_conds(arr[:, j], factor=rounds[j])
            if periodic[j]:
                rounded = (
                    np.degrees(u.normalize_periodic_range(np.radians(rounded))) + 180
                )
            comb.append(rounded)
        out.append(np.stack(comb, axis=1))
    return gio.ResultSequence(out)


def round_conds(arr, factor=10):
    pts_all = np.round(arr / factor) * factor
    return pts_all


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
        last_choice[ind_last + 1 : ind1 + 1] = choices[ind0]
    return last_choice


def _round_fields(df, fields, round_to=5, periodic=False, radians=False):
    for field in fields:
        val = df[field]
        quant = np.round(val / round_to, decimals=0) * round_to
        if periodic:
            quant = u.normalize_periodic_range(quant, radians=radians)
        df[field + "_rounded"] = quant

    return df


def _add_unique_conditions(df, new_field, cond_fields):
    _, conds = np.unique(df[list(cond_fields)].to_numpy(), axis=0, return_inverse=True)
    df[new_field] = conds
    return df


default_cond_fields = ("xPosition_rounded", "yPosition_rounded", "rotation_rounded")
training_spec_file = (
    "(?P<date>[0-9]+)_(?P<monkey>[a-zA-Z]+)_.*_Training_CHOICE_.*\\.txt"
)
default_spec_file = "(?P<date>[0-9]+)_(?P<monkey>[a-zA-Z]+)_.*_ALL_.*\\.txt"


def load_views_session(
    folder,
    spec_file_template=default_spec_file,
    img_template=".*_(?P<num>[0-9]+)\\.png",
    position_fields=("xPosition", "yPosition"),
    rotation_fields=("rotation",),
    condition_field="condition_number",
    position_round=5,
    rotation_round=22.5,
    cond_fields=default_cond_fields,
    remove_transparency=True,
):
    _, _, info = u.get_first_matching_file(
        folder, spec_file_template, load_func=pd.read_csv
    )
    gen = u.load_folder_regex_generator(folder, img_template, load_func=skimg.io.imread)
    img_list = []
    trl_nums = []
    for fp, img_info, img in gen:
        img_num = int(img_info["num"])
        trl_nums.append(img_num)
        img_list.append(img)
    img_list = np.stack(img_list, axis=0)
    trl_nums = np.array(trl_nums)
    inds = np.argsort(trl_nums)
    imgs = img_list[inds]
    trl_nums = trl_nums[inds]
    info["trl_num"] = trl_nums
    if remove_transparency:
        imgs = imgs[..., :-1]
    info = _round_fields(info, position_fields, round_to=position_round)
    info = _round_fields(info, rotation_fields, round_to=rotation_round, periodic=True)

    info = _add_unique_conditions(info, condition_field, cond_fields)
    return imgs, info


default_column_names = ("xPosition", "yPosition", "x_move", "y_move", "E-W", "rotation")


def load_views(
    folder,
    spec_file="view_coordinates.txt",
    img_template="(?P<num>[0-9]+).JPG",
    column_names=default_column_names,
    test_inds=np.arange(32, 80),
    position_fields=("xPosition", "yPosition"),
    condition_field="condition_number",
    test_column="isTestCondition",
    cond_fields=default_cond_fields,
):
    info = pd.read_csv(os.path.join(folder, spec_file), sep="\t", header=None)
    gen = u.load_folder_regex_generator(folder, img_template, load_func=skimg.io.imread)
    img_list = []
    info_list = []
    for fp, img_info, img in gen:
        img_num = int(img_info["num"])
        info_list.append(info.iloc[img_num])
        img_list.append(img)
    img_list = np.stack(img_list, axis=0)
    info_list = np.stack(info_list, axis=0)
    info_list[:, 0] = info_list[:, 0] + info_list[:, 2]
    info_list[:, 1] = info_list[:, 1] + info_list[:, 3]
    angle = np.expand_dims(np.arctan2(*u.make_unit_vector(info_list[:, 2:4]).T), 1)
    info_list = np.concatenate((info_list, angle), axis=1)
    info_dict = {k: info_list[:, i] for i, k in enumerate(column_names)}
    info_df = pd.DataFrame.from_dict(info_dict)
    info_df[test_column] = np.isin(np.arange(len(info_df)), test_inds)
    info_df = _round_fields(info_df, position_fields)
    info_df = _add_unique_conditions(info_df, condition_field, cond_fields)
    return img_list, info_df


date_task_dict = {
    "20231223": ("IsEast", "IsNorth"),
    "20240112": ("IsNorth", "IsEast"),
    "20240115": ("IsNorth", "IsEast"),
}


def extract_tc_position(xs, ys, ts):
    x_pos = np.zeros(len(xs))
    y_pos = np.zeros(len(xs))
    for i, x_i in enumerate(xs):
        y_i = ys[i]
        if np.isnan(ts[i]):
            x_pos[i] = np.nan
            y_pos[i] = np.nan
        else:
            ind = int(ts[i])
            x_pos[i] = x_i[ind]
            y_pos[i] = y_i[ind]
    return x_pos, y_pos


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
        ns_mask = data_all["Float9_RuleEW0NS1"] == 1
        ew_mask = data_all["Float9_RuleEW0NS1"] == 0
        if task_key is None:
            is_east = data_all["IsEast"]
            is_north = data_all["IsNorth"]
            rel_pos = np.zeros(len(is_east))
            rel_pos[ns_mask] = is_north[ns_mask]
            rel_pos[ew_mask] = is_east[ew_mask]
            irrel_pos = np.zeros(len(is_east))
            irrel_pos[ns_mask] = is_east[ns_mask]
            irrel_pos[ew_mask] = is_north[ew_mask]
            data_all["relevant_position"] = rel_pos
            data_all["irrelevant_position"] = irrel_pos
        else:
            data_all["relevant_position"] = data_all[task_key[0]]
            data_all["irrelevant_position"] = data_all[task_key[1]]
        data_all["white_right"] = np.logical_or(
            np.logical_and(
                data_all["relevant_position"] == 1,
                data_all["target_right"] == 1,
            ),
            np.logical_and(
                data_all["relevant_position"] == 0,
                data_all["target_right"] == 0,
            ),
        )
        data_all["pink_right"] = np.logical_or(
            np.logical_and(
                data_all["relevant_position"] == 1,
                data_all["target_right"] == 1,
            ),
            np.logical_and(
                data_all["relevant_position"] == 0,
                data_all["target_right"] == 0,
            ),
        )
        data_all.loc[ew_mask, "pink_right"] = pd.NA
        data_all.loc[ns_mask, "white_right"] = pd.NA
        data_all["pre_choice_rotation"] = extract_time_field(
            data_all,
            "post_rotation_end",
            "rotation_tc",
        )
        data_all["choice_pos_x"], data_all["choice_pos_y"] = extract_tc_position(
            data_all["pos_x"],
            data_all["pos_y"],
            data_all["choice_start"],
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
