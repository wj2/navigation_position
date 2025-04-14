import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold as skm
import sklearn.decomposition as skd
import itertools as it

import general.plotting as gpl


def plot_distance_distribs(conds, rdm, axs=None, fwid=3, colors=None, **kwargs):
    if axs is None:
        f, axs = plt.subplots(
            1, conds.shape[1], figsize=(fwid * conds.shape[1], fwid), sharey="all"
        )
    conds = conds.astype(float)
    if colors is None:
        colors = (None,) * conds.shape[1]
    comb_lists = list([] for i in range(conds.shape[1]))
    for j, k in it.combinations(range(len(conds)), 2):
        sum_jk = (conds[j] - conds[k]) ** 2
        if sum(sum_jk) == 1:
            ind = np.argmax(sum_jk)
            color_jk = colors[ind]
            distrib = rdm[:, j, k]
            axs[ind].hist(distrib, color=color_jk, **kwargs)
            comb_lists[ind].append(distrib)

    for i, ax in enumerate(axs):
        mus = np.array(list(np.mean(distrib) for distrib in comb_lists[i]))
        gpl.plot_trace_werr(np.expand_dims(mus, 1), [-10], ax=axs[i])
        gpl.clean_plot(ax, i)
        gpl.add_vlines(0, ax)


@gpl.ax_adder()
def plot_session_change_of_mind(*cond_dicts, ax=None):
    corr_traj = []
    gen_traj = []
    for i, cd in enumerate(cond_dicts):
        targ = cd["targets"]
        pc = np.swapaxes(cd["projection"], 1, 2)
        u_labels = np.unique(targ)
        out_lab = np.zeros((len(u_labels), pc.shape[-1]))
        for i, ul in enumerate(u_labels):
            for j in range(pc.shape[-1]):
                mask = targ[..., j] == ul
                out_lab[i, j] = np.mean(pc[mask])
        corr_traj.append(out_lab)

        pg = cd["projection_gen"]
        targ_gen = cd["labels_gen"]
        u_labels = np.unique(targ_gen)
        gen_traj_i = []
        for ul in u_labels:
            mask = ul == targ_gen
            gen_traj_i.append(np.mean(pg[:, mask], axis=(0, 1)))
        gen_traj.append(np.stack(gen_traj_i, axis=0))
    corr_traj = np.stack(corr_traj, axis=0)
    gen_traj = np.stack(gen_traj, axis=0)

    gpl.plot_colored_line(*corr_traj[:, 0], ax=ax, cmap="Blues")
    gpl.plot_colored_line(*corr_traj[:, 1], ax=ax, cmap="Greens")
    gpl.plot_colored_line(*gen_traj[:, 0], ax=ax, cmap="Blues")
    gpl.plot_colored_line(*gen_traj[:, 1], ax=ax, cmap="Greens")
    gpl.add_hlines(0, ax)
    gpl.add_vlines(0, ax)


@gpl.ax_adder(three_dim=True)
def visualize_orientation_positions(
    pop_dict,
    xs,
    x_targ=0,
    sess_ind=0,
    ax=None,
    mu=True,
    dim_red=skd.PCA,
    pt_cmap="magma",
    pd_colors=None,
):
    activity_all = []
    activity_groups = []
    x_ind = np.argmin(np.abs(xs - x_targ))
    if pd_colors is None:
        pd_colors = (
            plt.get_cmap("Blues")(0.5),
            plt.get_cmap("Reds")(0.5),
            plt.get_cmap("Greens")(0.5),
            plt.get_cmap("Oranges")(0.5),
        )
    for k, vs in pop_dict.items():
        vs = list(v[sess_ind] for v in vs)
        if mu:
            vs = list(np.mean(v, axis=2, keepdims=True) for v in vs)
        activity_all.extend(vs)
        com_k = np.concatenate(vs, axis=2)
        activity_groups.append(np.squeeze(com_k[..., x_ind], axis=1).T)
    comb_vs = np.concatenate(activity_all, axis=2)
    comb_vs = np.squeeze(comb_vs[..., x_ind], axis=1)
    dr = dim_red(3)
    dr.fit(comb_vs.T)
    for i, ag in enumerate(activity_groups):
        rep = dr.transform(ag)
        ax.scatter(*rep.T, c=np.linspace(0, 1, len(rep) + 1)[:-1], cmap=pt_cmap)
        ax.plot(*rep[[0, 1, 2, 3, 0]].T, color=pd_colors[i])
    ax.set_aspect("equal")


def visualize_rdms(combs, rdm, **kwargs):
    rdm_avg = np.mean(rdm, axis=0)
    rdm_avg[rdm_avg < 0] = 0
    trs = skm.MDS(n_components=3, dissimilarity="precomputed")
    embedding_avg = trs.fit_transform(rdm_avg)
    gpl.plot_highdim_structure(combs, embedding_avg, **kwargs)


def plot_all_place_fields(
    data,
    session_ind,
    axs=None,
    regions=None,
    spk_key="spikeTimes",
    region_key="neur_regions",
    position_fields=("pos_x", "pos_y"),
    fwid=2,
    t_start=None,
    t_end=None,
    **kwargs,
):
    spks = data[spk_key][session_ind].to_numpy()
    neur_regions = data[region_key][session_ind].iloc[0]
    if regions is not None:
        spks_mask = np.isin(neur_regions, regions)
        spks = list(spk[spks_mask] for spk in spks)
    n_plots = spks[0].shape[0]
    if axs is None:
        n_rcs = int(np.sqrt(n_plots))
        f, axs = plt.subplots(n_rcs, n_rcs, figsize=(fwid * n_rcs, fwid * n_rcs))
        axs = axs.flatten()
    pos = data[list(position_fields)][session_ind].to_numpy()
    if t_start is not None:
        start_inds = np.round(data[t_start][session_ind].to_numpy()).astype(int)
    else:
        start_inds = (None,) * len(pos)
    if t_end is not None:
        end_inds = np.round(data[t_end][session_ind].to_numpy()).astype(int)
    else:
        end_inds = (None,) * len(pos)
    for i, ax_i in enumerate(axs):
        spk_i = list(s[i] for s in spks)
        plot_feat_spks(
            pos, spk_i, ax=ax_i, start_inds=start_inds, end_inds=end_inds, **kwargs
        )
        gpl.clean_plot(ax_i, 1)
        gpl.clean_plot_bottom(ax_i)


def plot_place_field(
    data,
    session_ind,
    neur_ind,
    position_fields=("pos_x", "pos_y"),
    spk_key="spikeTimes",
    rot_key="rotation_tc",
    t_start=None,
    t_end=None,
    **kwargs,
):
    sts = data[spk_key][session_ind].to_numpy()
    neur_spks = list(r[neur_ind] for r in sts)
    pos = data[list(position_fields)][session_ind].to_numpy()
    rots = data[rot_key][session_ind].to_numpy()
    if t_start is not None:
        start_inds = np.round(data[t_start][session_ind].to_numpy()).astype(int)
    else:
        start_inds = (None,) * len(pos)
    if t_end is not None:
        end_inds = np.round(data[t_end][session_ind].to_numpy()).astype(int)
    else:
        end_inds = (None,) * len(pos)

    return plot_feat_spks(
        pos,
        neur_spks,
        color_ind=rots,
        start_inds=start_inds,
        end_inds=end_inds,
        **kwargs,
    )


@gpl.ax_adder(include_fig=False)
def plot_occupancy(
    data,
    session_ind,
    position_fields=("pos_x", "pos_y"),
    t_start=None,
    t_end=None,
    ax=None,
    cm="Blues",
    **kwargs,
):
    pos = data[list(position_fields)][session_ind].to_numpy()
    all_pos = []
    if t_start is not None:
        start_inds = np.round(data[t_start][session_ind].to_numpy()).astype(int)
    else:
        start_inds = (None,) * len(pos)
    if t_end is not None:
        end_inds = np.round(data[t_end][session_ind].to_numpy()).astype(int)
    else:
        end_inds = (None,) * len(pos)

    for i, p in enumerate(pos):
        pos_i = np.stack(p, axis=1)
        pos_i = pos_i[start_inds[i] : end_inds[i]]
        all_pos.append(pos_i)

    pos_cat = np.concatenate(all_pos, axis=0)
    norm_freq, edges = np.histogramdd(pos_cat, **kwargs)
    cents = list(x[:-1] + np.diff(x)[0] / 2 for x in edges)
    gpl.pcolormesh(*cents, norm_freq, ax=ax, cmap=cm)
    return norm_freq


@gpl.ax_adder(include_fig=True)
def plot_feat_spks(
    pos,
    neur_spks,
    cm="Blues",
    ax=None,
    bins=10,
    color_ind=None,
    color_norm=360,
    color_cmap="hsv",
    start_inds=None,
    end_inds=None,
    fig=None,
    colorbar=False,
    ms=1,
    low_freq_thr=1000,
    plot_pts=True,
):
    if start_inds is None:
        start_inds = (None,) * len(pos)
    if end_inds is None:
        end_inds = (None,) * len(pos)
    fire_pos = []
    all_pos = []
    pt_colors = []
    for i, ns in enumerate(neur_spks):
        pos_i = np.stack(pos[i], axis=1)
        pos_i = pos_i[start_inds[i] : end_inds[i]]
        if start_inds[i] is not None:
            ns = ns - start_inds[i]
        ms_times = np.round(ns).astype(int)
        filt_times = np.logical_and(ms_times > 0, ms_times < len(pos_i))
        ms_times = ms_times[filt_times]
        fire_pos.append(pos_i[ms_times])
        all_pos.append(pos_i)
        if color_ind is not None:
            pt_colors.append(color_ind[i][ms_times])
    pos_cat = np.concatenate(fire_pos, axis=0)
    pos_all_cat = np.concatenate(all_pos, axis=0)
    norm_freq, edges = np.histogramdd(pos_all_cat, bins=bins)
    norm_freq[norm_freq < low_freq_thr] = 0
    spk_freq, _ = np.histogramdd(pos_cat, bins=edges)
    cents = list(x[:-1] + np.diff(x)[0] / 2 for x in edges)
    if color_ind is not None:
        colors = np.concatenate(pt_colors, axis=0) / color_norm
        pt_colors = plt.get_cmap(color_cmap)(colors)
    else:
        pt_colors = None
    if len(cents) == 1:
        ax.plot(cents[0], 1000 * spk_freq / norm_freq)
    elif len(cents) == 2:
        cm = plt.get_cmap(cm)
        img = gpl.pcolormesh(*cents, 1000 * spk_freq / norm_freq, ax=ax, cmap=cm)
        if colorbar and fig is not None:
            fig.colorbar(img)
        if plot_pts:
            if pt_colors is not None:
                for i, pc in enumerate(pos_cat):
                    ax.plot(*pc, "o", ms=ms, color=pt_colors[i])
            else:
                ax.plot(
                    *pos_cat.T,
                    "o",
                    ms=ms,
                    color=pt_colors,
                )
        ax.set_aspect("equal")
    return ax


def plot_pops_all_units(
    xs,
    *pops,
    pop_ind=0,
    axs=None,
    fwid=2,
    regions=None,
    colors=None,
):
    n_neurs = pops[0][pop_ind].shape[0]
    if axs is None:
        n_rc = int(np.ceil(np.sqrt(n_neurs)))
        f, axs = plt.subplots(n_rc, n_rc, figsize=(fwid * n_rc, fwid * n_rc))
    else:
        f = None
    if regions is None:
        regions = (None,) * n_neurs
    if colors is None:
        colors = {}

    axs_flat = axs.flatten()
    for ind in range(n_neurs):
        col_ind = colors.get(regions[ind], (None,) * len(pops))
        for i, p_i in enumerate(pops):
            gpl.plot_trace_werr(
                xs, p_i[pop_ind][ind, 0], ax=axs_flat[ind], color=col_ind[i]
            )
        axs_flat[ind].set_title(ind)
    return f, axs


def visualize_decoding_dict(dec_dict, color_dict=None, axs=None, fwid=2, **kwargs):
    if axs is None:
        n_rows = len(list(dec_dict.values())[0])
        n_cols = len(dec_dict)
        f, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fwid * n_cols, fwid * n_rows),
            sharey=True,
            sharex="col",
        )
    else:
        f = None
    if color_dict is None:
        color_dict = {}
    default_colors = list(gpl.get_prop_cycler())

    for i, (time_k, var_decs) in enumerate(dec_dict.items()):
        for j, (var_k, dec_group) in enumerate(var_decs.items()):
            dec, xs = dec_group[:2]
            for k, dec_k in enumerate(dec):
                col_ijk = default_colors[j % len(default_colors)].get("color")
                l_ = gpl.plot_trace_werr(
                    xs,
                    dec_k,
                    ax=axs[j, i],
                    color=color_dict.get(var_k, col_ijk),
                    confstd=True,
                    **kwargs,
                )
                color_dict[k] = l_[0].get_color()
            gpl.clean_plot(axs[j, i], i)
            if j < n_rows - 1:
                gpl.clean_plot_bottom(axs[j, i])
            else:
                axs[j, i].set_xlabel(time_k)
            if i == 0:
                axs[j, i].set_ylabel(var_k)
            gpl.add_hlines(0.5, axs[j, i])
    return f, axs
