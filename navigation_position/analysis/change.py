import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as skm
import sklearn.model_selection as skms
import scipy.signal as sig
import jax
import jax.random as jr
import functools as ft

import general.plotting as gpl
import general.utility as u
import general.neural_analysis as na
# import general.unsupervised_analysis as gua
import general.data_io as gio
import navigation_position.analysis.representations as npra
import navigation_position.auxiliary as npa


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
                if (
                    dec_dicts is not None
                    and dec_dicts[k].get("projection_gen") is not None
                ):
                    proj = np.mean(dec_dicts[k]["projection_gen"], axis=0)
                    labels = dec_dicts[k]["labels_gen"]
                    flip_proj = proj * np.expand_dims(
                        np.sign(labels - np.mean(labels)), 1
                    )
                    for z, fp in enumerate(flip_proj):
                        color = proj_cm((z + 1) / (len(flip_proj) + 1))
                        axs[i, j].plot(xs, fp, color=color)

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


def _norm_cond_predictors(preds, conds):
    out = []
    for i, pred in enumerate(preds):
        pred = pred - conds[i][..., None]
        out.append(pred)
    return out


def _rad_to_xy_predictors(preds, mask):
    out = []
    for i, pred in enumerate(preds):
        dims = []
        for j in range(pred.shape[1]):
            if mask[j]:
                pred_dim = np.swapaxes(u.radian_to_sincos(np.radians(pred[:, j])), 1, 2)
            else:
                pred_dim = np.expand_dims(pred[:, j], 1)
            dims.append(pred_dim)
        out.append(np.concatenate(dims, axis=1))
    return out


def _get_transition_times(diff, xs, thr):
    out = np.zeros(len(diff))
    for i, d in enumerate(diff):
        last_wrong = np.where(d > thr)[0][-1]
        right = np.where(d < 0)[0]
        late_right = right[right > last_wrong]
        if len(late_right) > 0:
            trans = xs[late_right[0]]
        else:
            trans = np.nan
        out[i] = trans
    return out


def _detect_switch_time(
    proj, y, xs, thr=None, fill_noncom_times=True, commit_time=200, thr_mult=0.8
):
    commit_bins = commit_time / np.diff(xs)[0]
    thr = np.std(proj) if thr is None else thr
    thr = thr * thr_mult

    targs = np.unique(y)
    proj_diffs = []
    for targ in targs:
        mu = np.mean(proj[y == targ], axis=0, keepdims=True)
        proj_diffs.append(np.sqrt((proj - mu) ** 2))
    p1, p2 = proj_diffs
    t1, t2 = targs
    diff12 = p1 - p2
    c1_to_2_mask = np.logical_and(
        np.logical_and(np.sum(diff12 > thr, axis=1) >= commit_bins, diff12[:, -1] < 0),
        y == t1,
    )
    c2_to_1_mask = np.logical_and(
        np.logical_and(
            np.sum(-diff12 > thr, axis=1) >= commit_bins, -diff12[:, -1] < 0
        ),
        y == t2,
    )
    times = np.zeros(len(y))
    times[:] = np.nan
    times[c1_to_2_mask] = _get_transition_times(diff12[c1_to_2_mask], xs, thr)
    times[c2_to_1_mask] = _get_transition_times(-diff12[c2_to_1_mask], xs, thr)
    mask = np.logical_not(np.isnan(times))
    if fill_noncom_times:
        times[np.isnan(times)] = np.nanmean(times)
    return diff12, mask, times


def com_hmm_states(
        *args, **kwargs, ):
    return gua.hmm_states_cv(*args, **kwargs)


def com_dynamics_diff(X, y, com_mask, cv=skms.LeaveOneOut, recode=True):
    out_traj = np.zeros((4, X.shape[0], X.shape[-1]))
    for i, (tr_inds, te_inds) in enumerate(cv().split(X, y)):
        X_tr = X[tr_inds]
        y_tr = y[tr_inds]
        m_s_tr = com_mask[tr_inds]
        m0_tr = np.logical_and(y_tr == 0, ~m_s_tr)
        m1_tr = np.logical_and(y_tr == 1, ~m_s_tr)
        m0_c_tr = np.logical_and(y_tr == 0, m_s_tr)
        m1_c_tr = np.logical_and(y_tr == 1, m_s_tr)

        X_0s = u.make_unit_vector(np.mean(X_tr[m0_tr], axis=0).T).T
        X_1s = u.make_unit_vector(np.mean(X_tr[m1_tr], axis=0).T).T
        X_0c = u.make_unit_vector(np.mean(X_tr[m0_c_tr], axis=0).T).T
        X_1c = u.make_unit_vector(np.mean(X_tr[m1_c_tr], axis=0).T).T

        diff_uv = u.make_unit_vector((X_0s - X_1s).T).T[None]
        comb_uv = u.make_unit_vector(np.mean((X_0s, X_1s), axis=0).T).T[None]
        cs1_uv = u.make_unit_vector((X_0c - X_1s).T).T[None]
        cs2_uv = u.make_unit_vector((X_1c - X_0s).T).T[None]
        uv_group = np.stack((diff_uv, comb_uv, cs1_uv, cs2_uv), axis=0)
        out_traj[:, i] = np.sum(uv_group * X[te_inds][None], axis=-2)[:, 0]
    return out_traj, y, com_mask


def com_dynamics(X, y, com_mask, cv=skms.LeaveOneOut, recode=True):
    X_s = X[~com_mask]
    y_s = y[~com_mask]
    X_c = X[com_mask]
    y_c = y[com_mask]
    c_traj = np.zeros(
        (
            len(X_s),
            2,
        )
        + (X_c.shape[0], X_c.shape[-1])
    )
    s_traj = np.zeros((2,) + (X_s.shape[0], X_s.shape[-1]))
    for i, (tr_inds, te_inds) in enumerate(cv().split(X_s, y_s)):
        X_s_tr = X_s[tr_inds]
        y_s_tr = y_s[tr_inds]

        X_s_tr1 = u.make_unit_vector(np.mean(X_s_tr[y_s_tr == 0], axis=0).T).T
        X_s_tr2 = u.make_unit_vector(np.mean(X_s_tr[y_s_tr == 1], axis=0).T).T
        if recode:
            X_s_r1 = u.make_unit_vector(np.mean((X_s_tr1, X_s_tr2), axis=0).T).T
            X_s_r2 = u.make_unit_vector((X_s_tr1 - X_s_tr2).T).T
            X_s_tr1 = X_s_r2
            X_s_tr2 = X_s_r1
        X_s_tr1 = X_s_tr1[None]
        X_s_tr2 = X_s_tr2[None]

        s_traj[0, i] = np.sum(X_s[te_inds] * X_s_tr1, axis=1)
        s_traj[1, i] = np.sum(X_s[te_inds] * X_s_tr2, axis=1)

        c_traj[i, 0] = np.sum(X_c * X_s_tr1, axis=1)
        c_traj[i, 1] = np.sum(X_c * X_s_tr2, axis=1)
    return {"stable": (s_traj, y_s), "change": (c_traj, y_c)}


@gpl.ax_adder()
def plot_com_dynamics(
    X_s,
    y_s,
    X_c,
    y_c,
    ax=None,
    alpha_bg=0.1,
    ms=5,
    lw_bg=0.1,
    cdict=None,
    plot_com=True,
):
    if cdict is None:
        cdict = {0: "r", 1: "b"}
    for i in range(X_s.shape[1]):
        ax.plot(*X_s[:, i], color=cdict[y_s[i]], lw=lw_bg, alpha=alpha_bg)
    s_y0 = np.mean(X_s[:, y_s == 0], axis=1)
    gpl.plot_trace_ends(*s_y0, color=cdict[0], ms=ms, ax=ax)

    s_y1 = np.mean(X_s[:, y_s == 1], axis=1)
    gpl.plot_trace_ends(*s_y1, color=cdict[1], ms=ms, ax=ax)

    X_c = np.mean(X_c, axis=0)

    if plot_com:
        c_y0 = np.mean(X_c[:, y_c == 0], axis=1)
        c_y1 = np.mean(X_c[:, y_c == 1], axis=1)
        gpl.plot_trace_ends(*c_y0, ax=ax, ms=ms, color=cdict[0], ls="dashed")
        gpl.plot_trace_ends(*c_y1, ax=ax, ms=ms, color=cdict[1], ls="dashed")

    gpl.make_xaxis_scale_bar(ax)
    gpl.make_yaxis_scale_bar(ax, double=False)
    gpl.clean_plot(ax, 0)

    # for i in range(X_c.shape[1]):
    #     ax.plot(*X_c[:, i], color=cdict[y_c[i]], lw=.5, ls="dashed")
    #     ax.plot(*X_c[:, i, 0], "o", color=cdict[y_c[i]], lw=.5, ls="dashed")
    #     ax.plot(*X_c[:, i, -1], "o", color=cdict[y_c[i]], lw=.5, ls="dashed")
    # ax.plot(*c_y0, color=cdict[0], lw=1, ls="dashed")
    # ax.plot(*c_y1, color=cdict[1], lw=1, ls="dashed")


CHANGE_OF_MIND_START = "approach_start"


def change_of_mind_populations(
    data,
    time_start=CHANGE_OF_MIND_START,
    time_zeros=None,
    time_begin=-1000,
    time_end=1000,
    window=200,
    binstep=20,
    pca_pre=0.8,
    choice_field="chose_right",
    **kwargs,
):
    pops, xs = data.get_populations(
        window,
        time_begin,
        time_end,
        binstep=binstep,
        time_zero_field=time_start,
        time_zero=time_zeros,
        **kwargs,
    )
    choice = data[choice_field]
    outs = []
    for i, pop_i in enumerate(pops):
        targ_i = choice[i].to_numpy()
        if pop_i.shape[1] > 0:
            out = na.targeted_dimensionality_reduction(
                pop_i,
                targ_i,
                model=na.LinearSVCWrapper,
                cv=skms.LeaveOneGroupOut(),
                pre_pca=pca_pre,
            )
            out["X"] = pop_i
            out["y"] = targ_i
            diff_i, mask_i, time_i = _detect_switch_time(
                np.squeeze(out["test_projection"]), targ_i, xs
            )
            out["diff"] = diff_i
        else:
            out = None
        outs.append(out)
    return outs, xs


@gpl.ax_adder()
def plot_flux_heatmap(
    proj,
    targ,
    xs,
    bins=None,
    ax=None,
    cmap="PiYG",
    count_thr=2,
    n_x_bins=20,
    n_y_bins=50,
    y_max=5,
    lw=0.5,
):
    """Plot the average direction of neural activity from a particular point.

    Parameters
    ----------
    proj : array_like, N x T
       Projection along target dimension.
    targ : array_like, N
       Target values for each trial.
    xs : array_like, T
       The time points for each observation.
    bins : tuple, default=None
       The bins to use for the histogram. If None, they will be chosen according to
       n_x_bins and n_y_bins.
    ax : matplotlib.Axes, default=None
       The axes to use for plotting.
    cmap : string, default="bwr"
       The colormap for the heatmap.
    count_thr : int, default=2
       The number of trials needed in a bin to keep it rather than set it to nan in the
       plot.
    n_x_bins, n_y_bins : int, default=20
       The number of bins to use for the x and y axes, respectively.

    Returns
    -------
    None
    """
    proj_diff = proj[:, 1:] - proj[:, :-1]
    xs = xs[:-1]
    proj = proj[:, :-1]
    if bins is None:
        proj_ext = np.min([np.max(np.abs(proj)), y_max])
        bins = (
            np.linspace(-proj_ext, proj_ext, n_y_bins + 1),
            np.linspace(xs[0], xs[-1], n_x_bins + 1),
        )
    targ_tiled = np.tile(targ[:, None], (1, proj.shape[1]))
    pd_flat = proj_diff.flatten()
    xs_flat = np.tile(xs[None], (proj.shape[0], 1)).flatten()
    p_flat = proj.flatten()
    t_flat = targ_tiled.flatten()
    t_flat[t_flat == 0] = -1
    hmap, bins = np.histogramdd((p_flat, xs_flat), bins=bins, weights=pd_flat)
    counts, _ = np.histogramdd((p_flat, xs_flat), bins=bins)
    hmap[counts < count_thr] = np.nan
    hmap = hmap / counts

    extreme = np.nanstd(hmap)
    y_bins, x_bins = bins
    x_bins = x_bins[:-1] + np.diff(x_bins)[0] / 2
    y_bins = y_bins[:-1] + np.diff(y_bins)[0] / 2
    gpl.pcolormesh(x_bins, y_bins, hmap, cmap=cmap, vmin=-extreme, vmax=extreme, ax=ax)


@gpl.ax_adder()
def plot_com_heatmap(
    proj,
    targ,
    xs,
    bins=None,
    ax=None,
    cmap="Grays",
    count_thr=2,
    n_x_bins=20,
    n_y_bins=50,
    y_max=5,
    lw=0.5,
):
    """Plot the density of trials projected along a particular dimension.

    Parameters
    ----------
    proj : array_like, N x T
       Projection along target dimension.
    targ : array_like, N
       Target values for each trial.
    xs : array_like, T
       The time points for each observation.
    bins : tuple, default=None
       The bins to use for the histogram. If None, they will be chosen according to
       n_x_bins and n_y_bins.
    ax : matplotlib.Axes, default=None
       The axes to use for plotting.
    cmap : string, default="bwr"
       The colormap for the heatmap.
    count_thr : int, default=2
       The number of trials needed in a bin to keep it rather than set it to nan in the
       plot.
    n_x_bins, n_y_bins : int, default=20
       The number of bins to use for the x and y axes, respectively.

    Returns
    -------
    None
    """
    if bins is None:
        proj_ext = np.min([np.max(np.abs(proj)), y_max])
        bins = (
            np.linspace(-proj_ext, proj_ext, n_y_bins + 1),
            np.linspace(xs[0], xs[-1], n_x_bins + 1),
        )
    targ_tiled = np.tile(targ[:, None], (1, proj.shape[1]))
    xs_flat = np.tile(xs[None], (proj.shape[0], 1)).flatten()
    p_flat = proj.flatten()
    t_flat = targ_tiled.flatten()
    t_flat[t_flat == 0] = -1
    hmap, bins = np.histogramdd((p_flat, xs_flat), bins=bins, weights=t_flat)
    counts, _ = np.histogramdd((p_flat, xs_flat), bins=bins)
    hmap[counts < count_thr] = np.nan

    extreme = np.nanmax(np.abs(hmap))
    y_bins, x_bins = bins
    x_bins = x_bins[:-1] + np.diff(x_bins)[0] / 2
    y_bins = y_bins[:-1] + np.diff(y_bins)[0] / 2
    gpl.pcolormesh(
        x_bins, y_bins, counts, cmap=cmap, vmin=-extreme, vmax=extreme, ax=ax
    )


@gpl.ax_adder()
def plot_com_heatmap_and_averages(
    proj,
    targ,
    xs,
    com_mask,
    ax=None,
    heatmap=True,
    corr1_color="r",
    corr2_color="b",
    com1_color="m",
    com2_color="g",
    errorbar=False,
    **kwargs,
):
    if heatmap:
        plot_com_heatmap(proj, targ, xs, ax=ax, **kwargs)
    proj_com = proj[com_mask]
    targ_com = targ[com_mask]
    proj_corr = proj[~com_mask]
    targ_corr = targ[~com_mask]

    if errorbar:
        gpl.plot_trace_werr(
            xs,
            proj_com[targ_com == 0],
            color=com1_color,
            conf95=True,
            ax=ax,
        )
        gpl.plot_trace_werr(
            xs,
            proj_com[targ_com == 1],
            color=com2_color,
            conf95=True,
            ax=ax,
        )

        gpl.plot_trace_werr(
            xs,
            proj_corr[targ_corr == 0],
            color=corr1_color,
            conf95=True,
            ax=ax,
        )
        gpl.plot_trace_werr(
            xs,
            proj_corr[targ_corr == 1],
            color=corr2_color,
            conf95=True,
            ax=ax,
        )
    else:
        ax.plot(xs, np.mean(proj_com[targ_com == 0], axis=0), color=com1_color)
        ax.plot(xs, np.mean(proj_com[targ_com == 1], axis=0), color=com2_color)
        ax.plot(xs, np.mean(proj_corr[targ_corr == 0], axis=0), color=corr1_color)
        ax.plot(xs, np.mean(proj_corr[targ_corr == 1], axis=0), color=corr2_color)


@gpl.ax_adder()
def plot_com_heatmap_and_examples(
    proj,
    targ,
    xs,
    com_mask,
    ax=None,
    heatmap=True,
    lw_corr=0.5,
    lw_com=0.9,
    corr_cmap="bwr",
    com_cmap="vanimo",
    **kwargs,
):
    if heatmap:
        plot_com_heatmap(proj, targ, xs, ax=ax, **kwargs)
    proj_com = proj[com_mask]
    targ_com = targ[com_mask]
    corr_mask = ~com_mask
    c1, c2 = plt.get_cmap(com_cmap)([0.0, 1.0])
    ax.plot(
        xs,
        proj_com[targ_com == 0].T,
        color=c1,
        lw=lw_com,
    )
    ax.plot(
        xs,
        proj_com[targ_com == 1].T,
        color=c2,
        lw=lw_com,
    )

    proj_corr = proj[corr_mask]
    targ_corr = targ[corr_mask]
    c1, c2 = plt.get_cmap(corr_cmap)([0.0, 1.0])
    ax.plot(
        xs,
        proj_corr[targ_corr == 0].T,
        color=c1,
        lw=lw_corr,
    )
    ax.plot(
        xs,
        proj_corr[targ_corr == 1].T,
        color=c2,
        lw=lw_corr,
    )


def _rotate_xy_predictors(preds, xy_ind, angs, correct_180=True):
    out = []
    for i, pred in enumerate(preds):
        angs_i = angs[i] + correct_180 * 180
        rads = -np.radians(angs_i)[:, None]
        xys = pred[:, xy_ind]
        x_rot = xys[:, 0] * np.cos(rads) + xys[:, 1] * np.sin(rads)
        y_rot = -xys[:, 0] * np.sin(rads) + xys[:, 1] * np.cos(rads)
        new_pred = np.zeros_like(pred)
        new_pred[:] = pred
        new_pred[:, xy_ind] = np.stack((x_rot, y_rot), axis=1)
        out.append(new_pred)
    return out


fields_all = (
    "pos_x",
    "pos_y",
    "rotation_tc",
    "UserVars.RestructuredVRData.Joystick_Position_X",
    "UserVars.RestructuredVRData.Joystick_Position_Y",
    "eye_x",
    "eye_y",
)
cond_fields_all = (True, True, True, False, False, False, False)
rot_fields_all = (False, False, True, False, False, False, False)


def make_com_predictors(
    data,
    timing_key=None,
    time_begin=-1000,
    time_end=1000,
    window=50,
    binstep=10,
    time_zero_field="approach_start",
    fields=fields_all,
    cond_fields=cond_fields_all,
    rot_fields=rot_fields_all,
):
    predictors_orig, xs = data.get_field_timeseries(
        fields,
        timing_key=timing_key,
        begin=time_begin,
        end=time_end,
        binsize=window,
        binstep=binstep,
        time_zero_field=time_zero_field,
    )
    conds = npa.make_unique_conds(data)
    cond_fields = np.array(cond_fields)
    predictors_conds = _norm_cond_predictors(
        list(po[:, cond_fields] for po in predictors_orig),
        conds,
    )
    predictors_conds = _rad_to_xy_predictors(predictors_conds, (False, False, True))
    predictors_conds = _rotate_xy_predictors(
        predictors_conds, (0, 1), list(c[:, -1] for c in conds)
    )
    predictors = []
    for i, pc in enumerate(predictors_conds):
        predictors.append(
            np.concatenate(
                (pc, predictors_orig[i][:, np.logical_not(cond_fields)]), axis=1
            )
        )
    return predictors, xs


@gpl.ax_adder()
def plot_traj(pred, y, colors=None, ax=None, **kwargs):
    u_y = np.unique(y)
    if colors is None:
        colors = (None,) * len(u_y)
    for i, y_i in enumerate(u_y):
        m_i = y == y_i
        ax.plot(pred[m_i, 0].T, pred[m_i, 1].T, color=colors[i], **kwargs)


def template_change_of_mind(
    data,
    time_start=CHANGE_OF_MIND_START,
    time_end=1000,
    time_begin=-1000,
    window=20,
    binstep=10,
    choice_field="chose_right",
):
    """Determine which trials have predictors indicating change of mind.

    Parameters
    ----------
    data : Dataset
       session date in Dataset format
    subj_pos : tuple of strings
       fields to apply change of mind logic to.
    time_start : string
       timing field to start analysis on (default="pre_choice_start")
    time_end : float
       how long to end analysis after time_start (default=1000)
    window : float
       window size for analysis (default=100)
    binstep : float
       step between different bins
    norm_mask : tuple of booleans
       set which predictors to normalize according to their unique condition values
    rad_to_xy_mask : tuple of booleans
       set which predictors to convert from radians to sin-cos
    choice_field : string
       which field indicates the animal's choice on a particular trial

    Returns
    -------
    masks : ResultSequence
       masks for every session where trials with a detected change of mind are true.
    out : list of dictionaries
       list with full analysis results from every session
    xs : array_like
       time points of the bins for the full analysis results
    """
    predictors, xs = make_com_predictors(
        data,
        time_zero_field=time_start,
        time_begin=time_begin,
        time_end=time_end,
        window=window,
        binstep=binstep,
    )
    choice = data[choice_field]
    outs = []
    masks = []
    times = []
    offset_times = data[time_start]
    for i, pred_i in enumerate(predictors):
        targ_i = choice[i].to_numpy()
        out = na.targeted_dimensionality_reduction(
            pred_i,
            targ_i,
            model=na.LinearSVCWrapper,
            cv=skms.LeaveOneGroupOut(),
        )
        out["X"] = pred_i
        out["y"] = targ_i
        diff_i, mask_i, time_i = _detect_switch_time(
            np.squeeze(out["test_projection"]), targ_i, xs
        )
        out["diff"] = diff_i
        outs.append(out)
        masks.append(mask_i)
        times.append(time_i)
    masks = gio.ResultSequence(masks)
    times = gio.ResultSequence(times) + offset_times
    return masks, times, outs, xs


def compute_traj_var(proj, xs, n_win=10):
    window = np.ones((1,) * (len(proj.shape) - 1) + (n_win,)) / n_win
    conv_mask2 = sig.convolve(proj**2, window, mode="valid")
    conv_mask = sig.convolve(proj, window, mode="valid")
    std = np.mean(np.sqrt(conv_mask2 - conv_mask**2), axis=-2)

    std = np.squeeze(
        sig.convolve(np.std(proj, axis=-2, keepdims=True), window, mode="valid")
    )
    xs_conv = sig.convolve(xs, np.squeeze(window[0]), mode="valid")
    return std, xs_conv


def compute_avg_activity(proj):
    mu = np.mean(proj, axis=(0, 1))
    return mu


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
