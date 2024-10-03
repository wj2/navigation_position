import numpy as np
import matplotlib.pyplot as plt

import general.plotting as gpl
import general.utility as u
import general.data_io as gio
import navigation_position.analysis.representations as npra


default_dec_vars = ("choice side", "correct side", "rule", "white side", "pink side")
default_balance_vars = {
    "rule": None,
    "white side": None,
    "pink side": None,
    "choice side": ("target_right",),
    "correct side": ("chose_right",),
}


def decode_change_of_mind_regions(
    data,
    *args,
    dec_vars=default_dec_vars,
    balance_vars=default_balance_vars,
    eps=0.1,
    dist_thr=6,
    balance=False,
    **kwargs,
):
    change_mask = distance_change_masks(data, dist_thr=dist_thr, eps=eps)
    out_dict = {}
    if not balance:
        balance_vars = {}
    for i, dv in enumerate(dec_vars):
        m1_full = data[npra.default_dec_variables[dv]] == 1
        m2_full = data[npra.default_dec_variables[dv]] == 0
        m1_tr = m1_full.rs_and(change_mask.rs_not())
        m2_tr = m2_full.rs_and(change_mask.rs_not())
        m1_te = m1_full.rs_and(change_mask)
        m2_te = m2_full.rs_and(change_mask)
        bv_i = balance_vars.get(dv)
        out_regions = npra.decode_regions(
            npra.decode_masks,
            data,
            m1_tr,
            m2_tr,
            *args,
            gen_mask1=m1_te,
            gen_mask2=m2_te,
            balance_fields=bv_i,
            **kwargs,
        )
        out_dict[dv] = out_regions
    return out_dict


def visualize_change_of_mind_dec(
    out_dict,
    tzf="",
    fwid=3,
    axs=None,
    indiv_alpha=0.2,
    chance=0.5,
    proj_cm="hsv",
):
    proj_cm = plt.get_cmap(proj_cm)
    n_vars = len(out_dict)
    n_regions = len(list(out_dict.values())[0])
    if axs is None:
        f, axs = plt.subplots(
            n_vars,
            n_regions,
            figsize=(fwid * n_regions, fwid * n_vars),
            squeeze=False,
        )
    else:
        f = None
    for i, (dv, out_regions) in enumerate(out_dict.items()):
        for j, (use_region, out_ij) in enumerate(out_regions.items()):
            dec, xs, gen = out_ij[:3]
            if len(out_ij) == 4:
                dec_dicts = out_ij[-1]
            else:
                dec_dicts = None
            dec_l = gpl.plot_trace_werr(
                xs,
                np.nanmean(dec, axis=0),
                ax=axs[i, j],
                label="consistent",
                confstd=True,
            )
            gen_l = gpl.plot_trace_werr(
                xs,
                np.nanmean(gen, axis=0),
                ax=axs[i, j],
                label="change of mind",
                confstd=True,
            )
            for k in range(dec.shape[0]):
                axs[i, j].plot(
                    xs,
                    np.mean(dec[k], axis=0),
                    color=dec_l[0].get_color(),
                    zorder=-1,
                    alpha=indiv_alpha,
                )
                axs[i, j].plot(
                    xs,
                    np.mean(gen[k], axis=0),
                    color=gen_l[0].get_color(),
                    zorder=-1,
                    alpha=indiv_alpha,
                )
                if dec_dicts is not None:
                    proj = np.mean(dec_dicts[k]["projection_gen"], axis=0)
                    labels = dec_dicts[k]["labels_gen"]
                    flip_proj = proj * np.expand_dims(
                        np.sign(labels - np.mean(labels)), 1
                    )
                    for z, fp in enumerate(flip_proj):
                        color = proj_cm((z + 1) / (len(flip_proj) + 1))
                        axs[i, j].plot(xs, flip_proj, color=color)

            if j == 0:
                axs[i, j].set_ylabel("decoding {}".format(dv))
            gpl.clean_plot(axs[i, j], j)
            if i == 0:
                axs[i, j].set_title(use_region)
            if i < n_vars - 1:
                gpl.clean_plot_bottom(axs[i, j])
            else:
                axs[i, j].set_xlabel("time from {}".format(tzf))
            gpl.add_hlines(chance, axs[i, j])
            gpl.add_vlines(0, axs[i, j])
    return f, axs


def change_of_mind_trials(
    data,
    subj_pos=("pos_x", "pos_y"),
    time_start="pre_choice_start",
    time_end="approach_end",
    target_right="target_right",
    rotation="pre_choice_rotation",
    add_rot_deg=22,
    dist=5,
):
    rots = data[rotation]
    subjs = list(data.get_field_window(sp, time_start, time_end) for sp in subj_pos)
    targ_right = data[target_right]

    distances_targ = []
    distances_dist = []
    for i, rots_i in enumerate(rots):
        rots_i = rots_i.to_numpy()
        targ_right_i = targ_right[i].to_numpy()
        subj_i = list(subjs[j][i] for j in range(len(subjs)))
        d_targ_i = []
        d_dist_i = []
        for j in range(len(subj_i[0])):
            rots_ij = rots_i[j]
            left_off = dist * u.radian_to_sincos(np.radians(rots_ij - add_rot_deg))
            right_off = dist * u.radian_to_sincos(np.radians(rots_ij + add_rot_deg))
            subj_ij = np.stack(list(subj_i[k][j] for k in range(len(subj_i))), axis=1)

            left_pos = np.expand_dims(subj_ij[0] + left_off, 0)
            right_pos = np.expand_dims(subj_ij[0] + right_off, 0)
            if targ_right_i[j]:
                targ_ij = right_pos
                dist_ij = left_pos
            else:
                targ_ij = left_pos
                dist_ij = right_pos
            d_targ_i.append(np.sqrt(np.sum((subj_ij - targ_ij) ** 2, axis=1)))
            d_dist_i.append(np.sqrt(np.sum((subj_ij - dist_ij) ** 2, axis=1)))
        distances_targ.append(d_targ_i)
        distances_dist.append(d_dist_i)
    return distances_targ, distances_dist


def distance_change_masks(*args, eps=0.1, dist_thr=4.5, **kwargs):
    targ_dists, dist_dists = change_of_mind_trials(*args, **kwargs)
    masks = []
    for i, t_i in enumerate(targ_dists):
        d_i = dist_dists[i]
        dist_mask = []
        for j, t_ij in enumerate(t_i):
            d_ij = d_i[j]
            diff = d_ij - t_ij
            ij_mask = np.logical_or(t_ij < dist_thr, d_ij < dist_thr)
            diff = diff[ij_mask]
            eps_mask = np.abs(diff) < eps
            diff[eps_mask] = 0
            diff = np.unique(np.sign(diff))
            dist_mask.append(1 in diff and -1 in diff)
        dist_mask = np.array(dist_mask)
        masks.append(dist_mask)
    return gio.ResultSequence(masks)


def visualize_distance_trajectories(data, sess_ind, mask, axs=None, fwid=3, **kwargs):
    d_targ, d_dist = change_of_mind_trials(data, **kwargs)
    t_mask = np.array(d_targ[sess_ind], dtype=object)[mask]
    d_mask = np.array(d_dist[sess_ind], dtype=object)[mask]
    f, axs = plt.subplots(len(t_mask), 1, figsize=(fwid, fwid * len(t_mask)))
    for i, t_ij in enumerate(t_mask):
        d_ij = d_mask[i]
        print(data["correct_trial"][sess_ind][mask].iloc[i])
        tnum = data["Trial"][sess_ind][mask].iloc[i]
        axs[i].set_title("trial = {}".format(tnum))

        axs[i].plot(t_ij, label="distance to target")
        axs[i].plot(d_ij, ls="dashed", label="distance to distractor")
        axs[i].set_ylabel("distance")
        gpl.clean_plot(axs[i], 0)
        if i < len(axs) - 1:
            gpl.clean_plot_bottom(axs[i])
        axs[i].legend(frameon=False)
        t_start = data["approach_start"][sess_ind][mask].iloc[i]
        t_off = data["pre_choice_start"][sess_ind][mask].iloc[i]
        t_end = data["approach_end"][sess_ind][mask].iloc[i]
        gpl.add_vlines(t_start - t_off, axs[i])
        gpl.add_vlines(t_end - t_off, axs[i])
    axs[-1].set_xlabel("time from choice period (ms)")
