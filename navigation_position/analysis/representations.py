import numpy as np
import sklearn.neighbors as sknn
import imblearn.under_sampling as imb_us

import rsatoolbox as rsa
import general.neural_analysis as na
import general.data_io as gio


def equals_one(x):
    return x == 1


def equal_0(x):
    return x == 0


def _less_than_y(x, y):
    return x < y


def less_than_180(x):
    return _less_than_y(x, 180)


def less_than_2(x):
    return _less_than_y(x, 2)


default_funcs = {
    "choice orientation": less_than_2,
    "rewarded": equal_0,
}


reduced_time_dict = {
    "pre_rotation_end": (-1000, 0),
    "nav_end": (-1000, 0),
    "relevant_crossing_x": (-500, 500),
    "relevant_crossing_y": (-500, 500),
    "post_rotation_end": (-1000, 0),
    "choice_approach_end": (-1000, 0),
    "choice_start": (-500, 500),
    "approach_start": (-500, 500),
    "approach_end": (-1000, 0),
}


default_dec_variables = {
    "relevant position": "relevant_position",
    "east-west position": "IsEast",
    "north-south position": "IsNorth",
    "white side": "white_right",
    "pink side": "pink_right",
    "correct side": "target_right",
    "choice side": "chose_right",
    "choice color white": "chose_white",
    "choice color pink": "chose_pink",
    "choice orientation": "choice_rotation",
    "rule": "Float9_RuleEW0NS1",
    "rewarded": "TrialError",
}


def border_crossing_masks(
    data,
    crossing_fields=("relevant_crossing_x", "relevant_crossing_y"),
    time_range=("nav_start", "nav_end"),
):
    cross_trials_sep = list(data[cf].rs_isnan().rs_not() for cf in crossing_fields)
    cross_trials_sep_times = list(data[cf] for cf in crossing_fields)
    cross_trials = cross_trials_sep[0]
    cross_times = list(np.zeros(len(cf)) for cf in cross_trials_sep[0])
    non_cross_trials = cross_trials.rs_not()
    non_cross_times = list(np.zeros(len(cf)) for cf in cross_trials_sep[0])
    time_sample = data[list(time_range)]
    rng = np.random.default_rng()
    for i, ct in enumerate(cross_times):
        ct[:] = np.nan
        non_cross_times[i][:] = np.nan
    for i, cts in enumerate(cross_trials_sep):
        cross_trials = cross_trials.rs_or(cts)
        non_cross_trials = non_cross_trials.rs_and(cts.rs_not())
        for j, cts_j in enumerate(cts):
            mask = cts_j.to_numpy()
            cross_times[j][mask] = cross_trials_sep_times[i][j][mask]
            times_j = time_sample[j].to_numpy()
            samp = rng.uniform(times_j[:, 0], times_j[:, 1])
            non_cross_times[j][~mask] = samp[~mask]
    cross_times = gio.ResultSequence(cross_times)
    non_cross_times = gio.ResultSequence(non_cross_times)
    return (cross_trials, cross_times), (non_cross_trials, non_cross_times)


def make_variable_masks(
    data,
    dec_variables=default_dec_variables,
    func_dict=None,
    and_mask=None,
):
    """
    The variables of interest are:
    position (binary and continuous), correct side, view direction, choice
    """
    if func_dict is None:
        func_dict = default_funcs
    masks = {}
    for k, v in dec_variables.items():
        func = func_dict.get(k, equals_one)
        m1 = func(data[v])
        m2 = func(data[v]).rs_not()
        if and_mask is not None:
            m1 = m1.rs_and(and_mask)
            m2 = m2.rs_and(and_mask)
        masks[k] = (m1, m2)
    return masks


default_intersection_variables = (
    "relevant position",
    "white side",
    "choice side",
)


def make_mask_intersection(
    data,
    intersection_variables=default_intersection_variables,
    dec_variables=default_dec_variables,
    and_mask=None,
):
    new_dict = {k: dec_variables[k] for k in intersection_variables}
    if and_mask is not None:
        data = data.mask(and_mask)
    masks = make_variable_masks(data, dec_variables=new_dict)
    first_mask = list(ms[0] for ms in masks.values())
    intersection_inds = []
    comb_masks = []
    for i in range(len(first_mask[0])):
        comb_mask = np.stack(list(fm[i] for fm in first_mask), axis=1)
        combs, inds = np.unique(comb_mask, axis=0, return_inverse=True)
        comb_masks.append(combs)
        intersection_inds.append(inds)
    return data, comb_masks, intersection_inds


def condition_distances(
    data,
    tbeg,
    tend,
    tzf,
    intersection_variables=default_intersection_variables,
    and_mask=None,
    resamples=100,
    x_targ=-250,
    regions=None,
    **kwargs,
):
    data, combs, inds = make_mask_intersection(
        data, intersection_variables=intersection_variables, and_mask=and_mask
    )
    pops, xs = data.get_populations(
        tend - tbeg, tbeg, tend, time_zero_field=tzf, regions=regions,
    )
    t_ind = np.argmin(np.abs(xs - x_targ))
    rdm_list = []
    for i, pop in enumerate(pops):
        ui = np.unique(inds[i])
        inds_i = inds[i]
        pipe = na.make_model_pipeline(**kwargs)
        pop = pipe.fit_transform(pop[..., t_ind])

        rdm_arr = np.zeros((resamples, len(ui), len(ui)))
        sampler = imb_us.RandomUnderSampler()
        for j in range(resamples):
            pop_j, inds_ij = sampler.fit_resample(pop, inds_i)

            data = rsa.data.Dataset(pop_j, obs_descriptors={"stimulus": inds_ij})

            rdm = rsa.rdm.calc_rdm(
                data, descriptor="stimulus", noise=None, method="crossnobis"
            )

            rdm_arr[j] = rdm.get_matrices()[0]
        rdm_list.append(rdm_arr)
    return rdm_list, combs


def condition_averages(
    data,
    tbeg,
    tend,
    tzf,
    intersection_variables=default_intersection_variables,
    and_mask=None,
    **kwargs,
):
    data, combs, inds = make_mask_intersection(
        data, intersection_variables=intersection_variables, and_mask=and_mask
    )
    pops, xs = data.get_populations(tend - tbeg, tbeg, tend, time_zero_field=tzf)
    avgs = []
    for i, pop in enumerate(pops):
        ui = np.unique(inds[i])
        inds_i = inds[i]
        pipe = na.make_model_pipeline(**kwargs, tc=True)
        pop = pipe.fit_transform(pop)
        avg_resp = np.zeros((len(ui),) + pop.shape[1:])
        for j, ui_j in enumerate(ui):
            mask = inds_i == ui_j
            avg_resp[j] = np.mean(pop[mask], axis=0)
        avgs.append(avg_resp)
    return avgs, combs, xs


default_contrast_variables = {
    "east-west position": ("IsNorth",),
    "north-south position": ("IsEast",),
}


def make_variable_generalization_masks(
    data,
    contrast_variables=default_contrast_variables,
    func_dict=None,
):
    masks = make_variable_masks(data)
    masks_gen = {}
    for k, (m1, m2) in masks.items():
        for cv in contrast_variables[k]:
            func = func_dict.get(cv, equals_one)
            m1_p = m1.rs_and(func(data[cv]))
            m2_p = m2.rs_and(func(data[cv]))
            m1_n = m1.rs_and(~func(data[cv]))
            m2_n = m2.rs_and(~func(data[cv]))
            fcv = "{} x {}".format(k, cv)
            masks_gen[fcv] = ((m1_p, m2_p), (m1_n, m2_n))
    return masks_gen


def decode_masks_reverse(
    data,
    mask1,
    mask2,
    *args,
    gen_mask1=None,
    gen_mask2=None,
    **kwargs,
):
    out1 = decode_masks(
        data, mask1, mask2, *args, gen_mask1=gen_mask1, gen_mask2=gen_mask2, **kwargs,
    )
    out2 = decode_masks(
        data, gen_mask1, gen_mask2, *args, gen_mask1=mask1, gen_mask2=mask2, **kwargs,
    )
    dec = np.stack((out1[0], out2[0]), axis=1)
    gen = np.stack((out1[-1], out2[-1]), axis=1)
    xs = out1[1]
    return dec, xs, gen


def decode_masks(
    data,
    mask1,
    mask2,
    tbeg,
    tend,
    tzf,
    winsize=500,
    stepsize=50,
    gen_mask1=None,
    gen_mask2=None,
    use_nearest_neighbors=False,
    n_neighbors=5,
    **kwargs,
):
    if use_nearest_neighbors:
        kwargs["model"] = sknn.KNeighborsClassifier
        kwargs["params"] = {"n_neighbors": n_neighbors}
    out = data.decode_masks(
        mask1,
        mask2,
        winsize,
        tbeg,
        tend,
        stepsize,
        decode_m1=gen_mask1,
        decode_m2=gen_mask2,
        time_zero_field=tzf,
        **kwargs,
    )
    return out


def decode_times(data, time_dict=None, dec_vars=None, **kwargs):
    if time_dict is None:
        time_dict = reduced_time_dict
    if dec_vars is None:
        dec_vars = default_dec_variables
    out_dict = {}

    masks = make_variable_masks(data)
    for time_k, ts in time_dict.items():
        out_dict[time_k] = {}
        for var_k, k_masks in masks.items():
            out = decode_masks(data, *k_masks, *ts, time_k, **kwargs, ret_pops=True)
            out_dict[time_k][var_k] = out
    return out_dict


default_region_dict = {
    "dlPFC": ("DLPFCv", "DLPFCd"),
    "HPC": ("HPC",),
    "PMd": ("PMd",),
    "all": None,
}


def decode_regions(func, *args, region_list=default_region_dict, **kwargs):
    out_dict = {}
    for r_name, r_list in region_list.items():
        out_dict[r_name] = func(*args, regions=r_list, **kwargs)
    return out_dict


