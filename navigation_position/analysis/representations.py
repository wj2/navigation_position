import numpy as np
import scipy.signal as sig
import sklearn.neighbors as sknn
import sklearn.metrics.pairwise as skmp
import sklearn.svm as skm
import imblearn.under_sampling as imb_us
import itertools as it
import pandas as pd

import rsatoolbox as rsa
import general.neural_analysis as na
import general.utility as u
import general.data_io as gio
import navigation_position.auxiliary as npa
from . import view as npav


def equals_one_zero(x):
    return _equals(x, 1, 0)


def _equals(x, t1, t2):
    return x == t1, x == t2


def _less_than_greater_than_y(x, y):
    return x < y, x >= y


def less_than_greater_than_180(x):
    return _less_than_greater_than_y(x, 180)


def less_than_greater_than_2(x):
    return _less_than_greater_than_y(x, 2)


def _format_spikes(spk):
    pairs = []
    for ind, spk_i in enumerate(spk):
        arr_ind = np.stack((spk_i, np.ones_like(spk_i) * ind), axis=1)
        pairs.append(arr_ind)

    return np.concatenate(pairs, axis=0)


def _filter_neg(x):
    return x[np.all(x > 0, axis=1)]


default_waypoint_keys = (
    "UserVars.VR_Trial.Additional_Positions.x",
    "UserVars.VR_Trial.Additional_Positions.z",
)


def get_waypoint_locs(data, waypoint_keys=default_waypoint_keys):
    return npa.combine_temporal_keys(data, waypoint_keys, filt_func=_filter_neg)


def decode_prior_field(
    data,
    field,
    tbeg,
    tend,
    binsize=500,
    binstep=50,
    time_zero_field="choice_start",
    regions=None,
    n_trials=1,
    n_folds=50,
    **kwargs,
):
    pops, xs = data.get_populations(
        binsize,
        tbeg,
        tend,
        binstep,
        time_zero_field=time_zero_field,
        regions=regions,
    )
    targs = data[field]
    outs = []
    for i, pop in enumerate(pops):
        targ = targs[i]
        targ_shift = targ.to_numpy().astype(bool)[:-n_trials]
        pop_shift = pop[n_trials:]
        out = na.fold_skl_shape(pop_shift, targ_shift, n_folds, mean=False, **kwargs)
        outs.append(out)
    return outs, xs


def decode_pairs_pop(
    pop,
    cond,
    n_folds=50,
    test_prop=0.1,
    ret_dicts=False,
    shuffle=False,
    info=None,
    **kwargs,
):
    u_conds = np.unique(cond, axis=0)
    combs = it.combinations(range(len(u_conds)), 2)
    out_dec = np.zeros((len(u_conds), len(u_conds), n_folds, pop.shape[-1]))
    if info is not None:
        info = info.to_numpy()
    if ret_dicts:
        out_dicts = {}
    combs = ((0, 7),)
    for i, j in combs:
        m1 = np.all(u_conds[i] == cond, axis=1)
        m2 = np.all(u_conds[j] == cond, axis=1)
        pop_ij = np.concatenate((pop[m1], pop[m2]), axis=0)
        label_ij = np.concatenate((np.ones(sum(m1)), np.zeros(sum(m2))))
        if info is not None:
            info_ij = np.concatenate((info[m1], info[m2]), axis=0)
            info_ij = np.concatenate((info_ij, label_ij[:, None]), axis=1)

        out = na.fold_skl_shape(
            pop_ij,
            label_ij,
            n_folds,
            test_prop=test_prop,
            mean=False,
            balance_rel_fields=True,
            rel_flat=label_ij,
            shuffle=shuffle,
            **kwargs,
        )
        out_dec[i, j] = out["score"]
        if ret_dicts:
            out["pop_ij"] = pop_ij
            out["label_ij"] = label_ij
            if info is not None:
                out["info_ij"] = info_ij
            out_dicts[(i, j)] = out
    if ret_dicts:
        out_full = (out_dec, out_dicts)
    else:
        out_full = out_dec
    return out_full


def decode_masks_fix_seq_gen(
    data, tk, t1, t2, gk="generalization_trial", g1=0, g2=1, **kwargs
):
    m1 = data[tk] == t1
    m2 = data[tk] == t2
    g1 = data[gk] == g1
    g2 = data[gk] == g2
    m1_tr = m1.rs_and(g1)
    m2_tr = m2.rs_and(g1)
    m1_te = m1.rs_and(g2)
    m2_te = m2.rs_and(g2)

    return decode_masks_fix_seq(
        data, m1_tr, m2_tr, decode_m1=m1_te, decode_m2=m2_te, **kwargs
    )


def decode_masks_fix_seq(
    data,
    m1,
    m2,
    tbeg=0,
    tend=0,
    winsize=200,
    tzf="choice_start",
    n_saccs=2,
    min_trials=4,
    test_prop=0.2,
    n_folds=100,
    decode_m1=None,
    decode_m2=None,
    **kwargs,
):
    out_start = data.decode_masks(
        m1,
        m2,
        winsize,
        tbeg,
        tend,
        winsize,
        n_folds=n_folds,
        time_zero_field=tzf,
        min_trials_pseudo=min_trials,
        test_prop=test_prop,
        decode_m1=decode_m1,
        decode_m2=decode_m2,
        **kwargs,
    )
    dec_start, xs = out_start[:2]
    decs = [dec_start]
    if decode_m1 is not None:
        gens = [out_start[-1]]
    for i in range(n_saccs):
        ts, _ = npav.get_nth_fixation(data, i, tzf=tzf)
        if decode_m1 is not None:
            use_ts = (ts,) * 4
        else:
            use_ts = (ts, ts)

        out_i = data.decode_masks(
            m1,
            m2,
            winsize,
            tbeg,
            tend,
            winsize,
            n_folds=n_folds,
            time_zeros=use_ts,
            min_trials_pseudo=min_trials,
            test_prop=test_prop,
            decode_m1=decode_m1,
            decode_m2=decode_m2,
            **kwargs,
        )
        dec_i = out_i[0]
        decs.append(dec_i)
        if decode_m1 is not None:
            gens.append(out_i[-1])
    decs_out = np.stack(decs, axis=1)
    if decode_m1 is not None:
        gens_out = np.stack(gens, axis=1)
        out = (decs_out, xs, gens_out)
    else:
        out = (decs_out, xs)
    return out


def decode_view_info(
    data, ns=(-1, 0, 1, 2), info=("chose_right", "white_right", "pink_right"), **kwargs
):
    pass


def decode_strict_fixation_seq(
    data, labels, n=2, initial_winsize=200, tzf="choice_start", **kwargs
):
    first_sacc, _ = npav.get_nth_fixation(data, 0, start_or_end="start", tzf=tzf)
    initial = decode_strict_fixation(
        data,
        labels,
        None,
        fix_starts=data[tzf],
        fix_ends=first_sacc,
        tzf=tzf,
        **kwargs,
    )
    out_dicts = list([x] for x in initial)

    for i in range(n):
        out_i = decode_strict_fixation(data, labels, i, tzf=tzf, **kwargs)
        list(out_dicts[j].append(x) for j, x in enumerate(out_i) if x is not None)
    out = []
    for od in out_dicts:
        if od[0] is None:
            out.append(None)
        else:
            out.append(u.aggregate_dictionary(od))
    return out


def generalize_strict_fixation_pops(decs, targs):
    outs = []
    for i, dec in enumerate(decs):
        if dec is None:
            out = None
        else:
            targ_i = targs[i].to_numpy().astype(float)
            out = generalize_strict_fixation(dec, targ_i)
        outs.append(out)
    return outs


def generalize_strict_fixation(dec_dict, targs):
    pops = dec_dict["X"]
    ests = dec_dict["estimators"]
    n_pops = len(pops)
    out = np.zeros((n_pops, n_pops, ests.shape[1], ests.shape[-1]))
    for i, j in it.product(range(n_pops), repeat=2):
        if i == j:
            out[i, j] = dec_dict["score"][i]
        else:
            pops_j, targs_j = u.filter_nan(pops[j], targs)
            out[i, j] = na.apply_estimators(ests[i], pops_j, targs_j, transpose=False)
    return out


def decode_strict_fixation(
    data,
    labels,
    n,
    tzf="choice_start",
    n_folds=100,
    test_prop=0.2,
    fix_starts=None,
    fix_ends=None,
    regions=None,
    **kwargs,
):
    if fix_starts is None:
        fix_starts, _ = npav.get_nth_fixation(data, n, tzf=tzf)
    if fix_ends is None:
        fix_ends, _ = npav.get_nth_fixation(data, n + 1, start_or_end="start", tzf=tzf)

    pops = data.get_bounded_firing_rates(fix_starts, fix_ends, regions=regions)
    out_dicts = []
    for i, pop in enumerate(pops):
        labels_i = np.array(labels[i]).astype(float)
        pop_i, labels_i_filt = u.filter_nan(pop, labels_i)
        if np.prod(pop_i.shape) > 0:
            out = na.fold_skl_shape(
                pop_i,
                labels_i_filt,
                n_folds,
                test_prop=test_prop,
                mean=False,
                **kwargs,
            )
            out["X"] = pop
            out["y"] = labels_i
        else:
            out = None
        out_dicts.append(out)
    return out_dicts


def get_fixation_pops(
    data, ns, keys, combine_func=np.concatenate, tzf="choice_start", **kwargs
):
    pops_ns, starts, ends, ns_track = [], [], [], []
    for n in ns:
        fix_starts, xy_starts = npav.get_nth_fixation(data, n, tzf=tzf)
        fix_ends, xy_ends = npav.get_nth_fixation(
            data, n + 1, start_or_end="start", tzf=tzf
        )
        if n == -1:
            fix_starts = data[tzf]
            xy_starts = xy_ends

        pops = data.get_bounded_firing_rates(fix_starts, fix_ends, **kwargs)
        pops_ns.append(pops)
        starts.append(xy_starts)
        ends.append(xy_ends)
        ns_track.append(list((n,) * len(x) for x in xy_starts))
    info = data[list(keys)]
    outs = []
    for i, info_i in enumerate(info):
        outs_i = []
        for j, n in enumerate(ns):
            out_ij = {
                "pop": pops_ns[j][i],
                "start_xy": starts[j][i],
                "end_xy": ends[j][i],
                "ns": ns_track[j][i],
                "info": info_i,
            }
            outs_i.append(out_ij)
        out_i = u.aggregate_dictionary(outs_i, combine_func=combine_func)
        outs.append(out_i)
    return outs


def decode_strict_side_fixations(
    data,
    ns,
    keys=("chose_right", "white_right"),
    offset=0,
    gap=0,
    test_prop=0.2,
    n_folds=100,
    model=skm.LinearSVC,
    **kwargs,
):
    reps = get_fixation_pops(data, ns, keys)
    outs = []
    for rep in reps:
        out_r = {}
        for j, k in enumerate(keys):
            targ = rep["info"][:, j]
            mask = np.logical_not(pd.isna(targ))
            pop = rep["pop"][mask]
            targ = targ[mask].astype(int)
            if len(pop) > 0:
                out_j = {}
                for i, n in enumerate(ns):
                    fix = rep["ns"][mask] == n
                    left = rep["start_xy"][mask][:, 0] < offset + gap / 2
                    right = rep["start_xy"][mask][:, 0] > offset + gap / 2
                    ls_mask = np.logical_and(fix, left)
                    rs_mask = np.logical_and(fix, right)

                    out_ji = na.fold_skl_shape(
                        pop[rs_mask],
                        targ[rs_mask],
                        n_folds,
                        mean=False,
                        c_gen=pop[ls_mask],
                        l_gen=targ[ls_mask],
                        test_prop=test_prop,
                        model=model,
                        **kwargs,
                    )
                    out_j[n] = out_ji
                out_r[k] = out_j
        outs.append(out_r)
    return outs


def decode_side_fixation(
    data,
    labels,
    n,
    tzf="choice_start",
    n_folds=100,
    test_prop=0.2,
    lr_thresh=0,
    **kwargs,
):
    fix_starts, xy_starts = npav.get_nth_fixation(data, n, tzf=tzf)
    fix_ends, xy_ends = npav.get_nth_fixation(
        data, n + 1, start_or_end="start", tzf=tzf
    )

    pops = data.get_bounded_firing_rates(fix_starts, fix_ends)
    out_dicts = []
    for i, pop in enumerate(pops):
        out = na.fold_skl_shape(
            pop,
            np.array(labels[i]).astype(float),
            n_folds,
            test_prop=test_prop,
            mean=False,
            **kwargs,
        )
        out["X"] = pop
        out["y"] = labels
        out_dicts.append(out)
    return out_dicts


def decode_pairs(
    data,
    tbeg,
    tend,
    binsize=500,
    binstep=50,
    rot_only=False,
    time_zero_field="choice_start",
    info_fields=("correct_trial", "white_right"),
    **kwargs,
):
    pops, xs = data.get_populations(
        binsize,
        tbeg,
        tend,
        binstep,
        time_zero_field=time_zero_field,
    )
    conds = npa.make_unique_conds(data, rot_only=rot_only)
    info = data[list(info_fields)]
    outs = []
    for i, pop in enumerate(pops):
        out = decode_pairs_pop(pop, conds[i], info=info[i], **kwargs)
        outs.append(out)
    return outs, xs


def get_distance_to_nearest_waypoint(pos, waypoint_locs):
    wp_dists = []
    for i, pos_i in enumerate(pos):
        wps = waypoint_locs[i]
        wps_pos = np.concatenate((pos_i[:1], wps, pos_i[-1:]), axis=0)
        min_dist = np.min(
            skmp.euclidean_distances(pos_i, wps_pos), axis=1, keepdims=True
        )
        wp_dists.append(min_dist)
    return wp_dists


def _bin_spiketimes(spks, ts, n_neurs, start=None, end=None, binsize=100, binstep=100):
    if start is None:
        start = 0
    step = np.gcd(binsize, binstep)
    filt = np.ones((int(binsize / binstep), 1))
    filt = filt / np.sum(filt)
    neur_bins = np.arange(n_neurs + 1)
    if end is None:
        end = ts[-1]
    b_edges = np.arange(start - binsize / 2, end + 2 * binsize, step)
    b_cents = b_edges[:-1] + np.diff(b_edges)[0] / 2

    grouped, _ = np.histogramdd(spks, bins=(b_edges, neur_bins))
    grouped = sig.convolve(grouped, filt, mode="valid")
    b_cents = sig.convolve(b_cents, filt[:, 0], mode="valid")
    return grouped, b_cents


def position_populations(
    data,
    start_field=None,
    end_field=None,
    binsize=100,
    binstep=100,
    pos_keys=("pos_x", "pos_y"),
    spks="spikeTimes",
    n_neurs="n_neurs",
    concat_trials=True,
    regions=None,
    regions_key="neur_regions",
    **kwargs,
):
    spks_all = data[spks]
    pos_all = data[list(pos_keys)]
    if start_field is not None:
        start_ts = data[start_field]
    if end_field is not None:
        end_ts = data[end_field]
    sess_out = []
    n_neurs = data[n_neurs]

    for i, spks in enumerate(spks_all):
        trl_spks_all = []
        trl_fields_all = []
        nn_i = n_neurs[i]

        if regions is not None:
            rs = data[regions_key][i].iloc[0]
            r_mask = np.isin(rs, regions)
        else:
            r_mask = np.ones(nn_i, dtype=bool)

        for j, spk_ij in enumerate(spks):
            spk_ij = _format_spikes(spk_ij)
            pos_ij = np.stack(pos_all[i].iloc[j]).T
            if start_field is not None:
                s_ij = start_ts[i].iloc[j]
            else:
                s_ij = None
            if end_field is not None:
                e_ij = end_ts[i].iloc[j]
            else:
                e_ij = None
            ts = np.arange(pos_ij.shape[0])
            b_spk_ij, ts = _bin_spiketimes(
                spk_ij, ts, nn_i, s_ij, e_ij, binsize=binsize, binstep=binstep
            )
            f_ij = pos_ij[np.round(ts).astype(int)]
            b_spk_ij = b_spk_ij[:, r_mask]
            trl_spks_all.append(b_spk_ij)
            trl_fields_all.append(f_ij)
        if concat_trials:
            trl_spks_all = np.concatenate(trl_spks_all, axis=0)
            trl_fields_all = np.concatenate(trl_fields_all, axis=0)
        sess_out.append((trl_spks_all, trl_fields_all))
    return sess_out


def get_all_conjunctive_pops(
    data, mask_dict, tbeg, tend, winsize=500, stepsize=50, tzfs="choice_start", **kwargs
):
    out_dict = {}
    for k, masks in mask_dict.items():
        xs, pops = data.get_dec_pops(
            winsize, tbeg, tend, stepsize, *masks, tzfs=tzfs, **kwargs
        )
        out_dict[k] = pops
    return out_dict, xs


def format_conjunctions_for_decoding(pop_dict, sess_ind):
    labels = []
    pops = []
    label_dict = {}
    cond_num = 0
    for i, (k, pd_) in enumerate(pop_dict.items()):
        pd_sess = list(x[sess_ind] for x in pd_)
        for j, pd_ij in enumerate(pd_sess):
            n_trls = pd_ij.shape[2]
            labels.extend((cond_num,) * n_trls)
            pops.append(pd_ij)
            label_dict[cond_num] = (k, j)
            cond_num += 1
    pops_all = np.concatenate(pops, axis=2)
    return pops_all, np.array(labels)


def decode_conjunctions(pop, labels, folds_n=20, **kwargs):
    pop = np.squeeze(pop, axis=1)
    out = na.fold_skl_flat(pop, labels, folds_n, mean=False, **kwargs)
    return out


full_quads = (
    (1, 1),
    (1, 0),
    (0, 1),
    (0, 0),
)


def direction_position_conjunction_masks(data, quads=full_quads, rots=(0, 1, 2, 3)):
    ns = data["IsNorth"]
    ew = data["IsEast"]
    rot = data["choice_rotation"]

    if rots is None:
        rots = (None,)
    full_dict = {}
    for ns_i, ew_i in quads:
        quad_list = []
        for rot_i in rots:
            if ns_i is not None:
                m1 = ns == ns_i
            else:
                m1 = ns > -10
            if ew_i is not None:
                m2 = ew == ew_i
            else:
                m2 = ew > -10
            if rot_i is not None:
                m3 = rot == rot_i
            else:
                m3 = rot > -10
            mask = m1.rs_and(m2).rs_and(m3)
            quad_list.append(mask)
        full_dict[(ns_i, ew_i)] = quad_list
    return full_dict


def _boundary_vis_wrap(fields):
    ns = gio.ResultSequence(x["IsNorth"] for x in fields)
    ew = gio.ResultSequence(x["IsEast"] for x in fields)
    rot = gio.ResultSequence(x["choice_rotation"] for x in fields)
    out = _boundary_vis(ns, ew, rot)
    return out


def _boundary_vis(ns, ew, rot):
    sw = (ns == 0).rs_and(ew == 0)
    nw = (ns == 1).rs_and(ew == 0)
    se = (ns == 0).rs_and(ew == 1)
    ne = (ns == 1).rs_and(ew == 1)
    sw_bound = sw.rs_and(rot.one_of((0, 1)))
    nw_bound = nw.rs_and(rot.one_of((1, 2)))
    se_bound = se.rs_and(rot.one_of((0, 3)))
    ne_bound = ne.rs_and(rot.one_of((2, 3)))
    bound = (sw_bound + nw_bound + se_bound + ne_bound) > 0
    no_bound = bound.rs_not()
    return bound, no_bound


default_funcs = {
    "choice orientation": less_than_greater_than_2,
    "boundary visible": _boundary_vis_wrap,
}


reduced_time_dict = {
    "pre_rotation_end": (-1000, 0),
    "nav_start": (-500, 1000),
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
    "boundary visible": ["IsNorth", "IsEast", "choice_rotation"],
    "rule": "Float9_RuleEW0NS1",
    "rewarded": "correct_trial",
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
        func = func_dict.get(k, equals_one_zero)
        m1, m2 = func(data[v])
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
        tend - tbeg,
        tbeg,
        tend,
        time_zero_field=tzf,
        regions=regions,
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
            func = func_dict.get(cv, equals_one_zero)
            c1, c2 = func(data[cv])
            m1_p = m1.rs_and(c1)
            m2_p = m2.rs_and(c1)
            m1_n = m1.rs_and(c2)
            m2_n = m2.rs_and(c2)
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
        data,
        mask1,
        mask2,
        *args,
        gen_mask1=gen_mask1,
        gen_mask2=gen_mask2,
        **kwargs,
    )
    out2 = decode_masks(
        data,
        gen_mask1,
        gen_mask2,
        *args,
        gen_mask1=mask1,
        gen_mask2=mask2,
        **kwargs,
    )
    dec = np.stack((out1[0], out2[0]), axis=1)
    gen = np.stack((out1[-1], out2[-1]), axis=1)
    xs = out1[1]
    return dec, xs, gen


def _make_shifted_data_masks(shift, data, *masks):
    fwd_mask = []
    assert shift > 0

    new_masks = tuple([] for _ in masks)
    for i, trls in enumerate(data["Trial"]):
        m_fwd = np.ones(len(trls), dtype=bool)
        m_fwd[:shift] = False
        fwd_mask.append(m_fwd)

        m_bwd = np.ones(len(trls), dtype=bool)
        m_bwd[-shift:] = False
        for j, m_j in enumerate(masks):
            m_ij = m_j[i].to_numpy()
            new_masks[j].append(m_ij[m_bwd])

    data_masked = data.mask(gio.ResultSequence(fwd_mask))
    new_masks = list(gio.ResultSequence(m) for m in new_masks)
    return data_masked, new_masks


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
    shift=0,
    **kwargs,
):
    if use_nearest_neighbors:
        kwargs["model"] = sknn.KNeighborsClassifier
        kwargs["params"] = {"n_neighbors": n_neighbors}
    if shift != 0:
        data, (mask1, mask2) = _make_shifted_data_masks(shift, data, mask1, mask2)

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

    masks = make_variable_masks(data, dec_variables=dec_vars)
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
