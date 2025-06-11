import numpy as np
import sklearn.svm as skm
import matplotlib.pyplot as plt
import itertools as it

import general.plotting as gpl
import general.neural_analysis as na
import general.tf.networks as gtfn
import general.data_io as gio
import navigation_position.auxiliary as npa


def get_network_view_representations(views, model_tuple=None, flatten=True, **kwargs):
    if model_tuple is None:
        model_tuple = gtfn.default_pre_model
    model = gtfn.GenericPretrainedNetwork(*model_tuple, **kwargs)
    reps = model.get_representation(views).numpy()
    if flatten:
        reps = np.reshape(reps, (reps.shape[0], -1))
    return reps


def get_view_trajectories(data, tzf="choice_start", **kwargs):
    eps = npa.combine_temporal_keys(
        data, ("eye_x", "eye_y"), tzf="choice_start", **kwargs
    )
    return eps


def get_nth_fixation(data, n, start_or_end="end", tzf="choice_start", **kwargs):
    starts, ends, start_xy, end_xy = npa.get_saccade_info(
        data, tzf=tzf, sub_tzf=False, **kwargs
    )
    if start_or_end == "end":
        ts = ends
        xys = end_xy
    else:
        ts = starts
        xys = start_xy
    ts_out = []
    xys_out = []

    for i, t_i in enumerate(ts):
        out_t_i = np.zeros(len(t_i))
        out_xy_i = np.zeros((len(t_i), 2))
        for j, t_ij in enumerate(t_i):
            if len(t_ij) > n:
                out_t_i[j] = t_ij[n]
                out_xy_i[j] = xys[i][j][n]
            else:
                out_t_i[j] = np.nan
                out_xy_i[j] = np.nan
        ts_out.append(out_t_i)
        xys_out.append(out_xy_i)
    ts_res = gio.ResultSequence(ts_out)
    xy_res = gio.ResultSequence(xys_out)
    return ts_res, xy_res


def get_high_view_mask(*args, thresh=25, tbeg=-500, tend=500, **kwargs):
    eps = get_view_trajectories(*args, **kwargs)
    high_masks = []
    for ep in eps:
        high_mask = np.array(list(np.any(x[:, 1] > 25) for x in ep))
        high_masks.append(high_mask)
    return gio.ResultSequence(high_masks)


def eye_position_map(
    data,
    sess_ind=0,
    fwid=2,
    cmap="hsv",
    cross_len=2,
    axs=None,
    s=0.1,
    feat_ind=-1,
    make_map=False,
    **kwargs,
):
    conds = npa.make_unique_conds(data)[sess_ind][:, feat_ind][:, None]
    eye_pos = np.array(get_view_trajectories(data, **kwargs)[sess_ind], dtype=object)
    cm = plt.get_cmap(cmap)
    conds_u = np.unique(conds, axis=0)
    if axs is None:
        n_conds = len(conds_u)
        n = int(np.ceil(np.sqrt(n_conds)))
        f, axs = plt.subplots(
            n, n, sharex=True, sharey=True, figsize=(n * fwid, n * fwid)
        )
        axs = axs.flatten()
    for i, cond in enumerate(conds_u):
        mask = np.all(cond == conds, axis=1)
        trls = eye_pos[mask]
        c = cm(i / len(conds_u))
        cm_i = gpl.make_linear_cmap(c)
        if make_map:
            all_pos = np.concatenate(trls, axis=0)
            all_pos = all_pos[~np.any(np.isnan(all_pos), axis=1)]
            weights, (xs, ys) = np.histogramdd(all_pos, bins=20)
            xs_cent = xs[:-1] + np.diff(xs)[0] / 2
            ys_cent = ys[:-1] + np.diff(ys)[0] / 2

            gpl.pcolormesh(
                xs_cent, ys_cent, np.log(weights + 1).T, ax=axs[i], cmap=cm_i
            )
        else:
            for t in trls:
                axs[i].scatter(*t.T, c=np.arange(len(t)), cmap=cm_i, s=s)
        axs[i].plot([-cross_len, cross_len], [0, 0], color="k")
        axs[i].plot([0, 0], [-cross_len, cross_len], color="k")
        axs[i].set_aspect("equal")
        gpl.clean_plot(axs[i], 1)
        gpl.clean_plot_bottom(axs[i])


def decode_feature(
    reps,
    labels,
    n_folds=20,
    test_frac=0.2,
    model=skm.SVC,
    return_estimator=False,
    **kwargs,
):
    if len(reps.shape) > 2:
        reps = np.reshape(reps, (reps.shape[0], -1))
    pipe = na.make_model_pipeline(model=model, **kwargs)
    out = na.cv_wrapper(
        pipe,
        reps,
        np.array(labels),
        test_frac=test_frac,
        n_folds=n_folds,
        return_indices=True,
        return_estimator=return_estimator,
    )
    return out


def generalize_feature_masks(
    reps,
    labels,
    train_mask,
    test_mask,
    **kwargs,
):
    if len(reps.shape) > 2:
        reps = np.reshape(reps, (reps.shape[0], -1))
    reps_tr = reps[train_mask]
    reps_te = reps[test_mask]
    labels_tr = labels[train_mask]
    labels_te = labels[test_mask]

    out_dec = decode_feature(reps_tr, labels_tr, return_estimator=True, **kwargs)
    ests = out_dec["estimator"]
    gen = np.zeros(len(ests))
    gen_preds = np.zeros((len(ests), sum(test_mask)))
    gen_targs = np.zeros_like(gen_preds)
    indices_gen = (np.where(test_mask)[0],) * len(ests)
    for i, est in enumerate(ests):
        gen[i] = est.score(reps_te, labels_te)
        gen_preds[i] = est.predict(reps_te)
        gen_targs[i] = labels_te
    out_dec["gen"] = gen
    out_dec["indices_gen"] = indices_gen
    out_dec["predictions_gen"] = gen_preds
    out_dec["targets_gen"] = gen_targs
    return out_dec


def generalize_feature(
    reps,
    labels,
    info,
    train_test_column="isTestCondition",
    **kwargs,
):
    if len(reps.shape) > 2:
        reps = np.reshape(reps, (reps.shape[0], -1))
    train_mask = info[train_test_column] == 0
    test_mask = info[train_test_column] == 1
    return generalize_feature_masks(reps, labels, train_mask, test_mask)


def combined_generalization_decoding(
    reps, img_info, dec_fields=("xPosition", "yPosition"), threshold=500, **kwargs
):
    out_all = {}
    for field in dec_fields:
        field_dict = {}
        out_dec = decode_feature(reps, img_info[field] > threshold, **kwargs)
        out_gen = generalize_feature(
            reps, img_info[field] > threshold, img_info, **kwargs
        )
        field_dict["dec_all"] = out_dec["test_score"]
        field_dict["dec_gen"] = out_gen["gen"]
        out_all[field] = field_dict
    return out_all


def plot_decoding(combined_dict, axs=None, fwid=3):
    if axs is None:
        f, axs = plt.subplots(
            len(combined_dict), 1, figsize=(1 * fwid, len(combined_dict) * fwid)
        )
    for i, (field, field_dict) in enumerate(combined_dict.items()):
        gpl.violinplot([field_dict["dec_all"]], [0], ax=axs[i])
        gpl.violinplot([field_dict["dec_gen"]], [1], ax=axs[i])
        gpl.add_hlines(0.5, axs[i])
        gpl.clean_plot(axs[i], 0)
        if i < len(combined_dict) - 1:
            gpl.clean_plot_bottom(axs[i])
        axs[i].set_title(field)
    axs[-1].set_xticks([0, 1])
    axs[-1].set_xticklabels(["full decoding", "test generalization"])


def repeated_condition_mask(info, repeated_field="isTestCondition"):
    f = info[repeated_field]
    mask = f == 0
    not_mask = f == 1
    return mask, not_mask


def sixteen_condition_mask(
    info,
    position_fields=("xPosition_rounded", "yPosition_rounded"),
    rotation_field="rotation_rounded",
    offset=20,
    center=(500, 500),
    rotations=(-270, -180, -90, 0, 90, 180, 270),
):
    offsets = np.array(list(it.product((-1, 1), repeat=2))) * offset
    positions = np.expand_dims(center, 0) + offsets
    trl_positions = info[list(position_fields)].to_numpy()

    x_in = np.isin(trl_positions[:, 0], positions[:, 0])
    y_in = np.isin(trl_positions[:, 1], positions[:, 1])
    rotation_in = np.isin(info[rotation_field], rotations)
    mask = x_in * y_in * rotation_in
    not_mask = np.logical_not(mask)
    return mask, not_mask


def condition_errors(
    reps,
    labels,
    info,
    n_folds=200,
    test_frac=0.1,
    model=skm.LinearSVC,
    condition_field="condition_number",
    **kwargs,
):
    conditions = info[condition_field].to_numpy()
    conds, inds = np.unique(conditions, axis=0, return_inverse=True)
    pipe = na.make_model_pipeline(model=model, dual="auto", **kwargs)

    out = na.cv_wrapper(
        pipe,
        reps,
        labels,
        test_frac=test_frac,
        n_folds=n_folds,
        return_indices=True,
    )
    test_inds = np.stack(out["indices"]["test"], axis=0).flatten()
    preds = out["predictions"].flatten()
    targs = out["targets"].flatten()
    cond_accuracy = np.zeros((len(conds), 2))
    for i in range(len(conds)):
        cond_inds = np.where(inds == i)[0]
        use_mask = np.isin(test_inds, cond_inds)
        cond_accuracy[i, 0] = np.sum(preds[use_mask] == targs[use_mask])
        cond_accuracy[i, 1] = np.sum(use_mask)
    return cond_accuracy, conds, out


def get_cond_fields(
    info,
    cond_fields=("xPosition_rounded", "yPosition_rounded", "rotation_rounded"),
    cond_field="condition_number",
):
    conds, inds = np.unique(info[cond_field], return_index=True)
    return conds, info[list(cond_fields)].iloc[inds].to_numpy()


def organize_condition_results(
    info,
    inds,
    preds,
    targs,
    cond_field="condition_number",
):
    conds = info[cond_field]
    u_conds, pos_rot = get_cond_fields(info)
    corr = np.zeros((len(u_conds), 2))
    for i, c in enumerate(u_conds):
        c_inds = np.where(conds == c)[0]
        mask = np.isin(inds, c_inds)
        corr[i, 0] = np.sum(preds[mask] == targs[mask])
        corr[i, 1] = np.sum(mask)
    return pos_rot, corr


@gpl.ax_adder()
def visualize_condition_errors(
    conds,
    cond_accuracy,
    cm="coolwarm",
    ax=None,
    arrow_len=3,
    arrow_wid=1,
    add_lines=True,
):
    cm = plt.get_cmap(cm)
    xy = conds[:, :2]
    cond_rads = np.radians(conds[:, 2])
    xy_delt = np.stack((np.sin(cond_rads), np.cos(cond_rads)), axis=1)
    for i, cond in enumerate(conds):
        if cond_accuracy[i, 1] > 0:
            acc = cond_accuracy[i, 0] / cond_accuracy[i, 1]
            color = cm(acc)
            ax.arrow(*xy[i], *xy_delt[i] * arrow_len, ec="k", fc=color, width=arrow_wid)
    ax.set_aspect("equal")
    if add_lines:
        gpl.add_hlines(500, ax)
        gpl.add_vlines(500, ax)


@gpl.ax_adder()
def visualize_tr_gen_condition_errors(
    info,
    tr_mask,
    te_mask,
    out_dec,
    tr_cm="coolwarm",
    te_cm="PiYG",
    ax=None,
):
    tr_conds, tr_corr = organize_condition_results(
        info[tr_mask],
        np.stack(out_dec["indices"]["test"], axis=0),
        out_dec["predictions"],
        out_dec["targets"],
    )
    visualize_condition_errors(
        tr_conds,
        tr_corr,
        cm=tr_cm,
        ax=ax,
        add_lines=False,
    )

    te_conds, te_corr = organize_condition_results(
        info[te_mask],
        np.stack(out_dec["indices_gen"], axis=0),
        out_dec["predictions_gen"],
        out_dec["targets_gen"],
    )
    visualize_condition_errors(te_conds, te_corr, cm=te_cm, ax=ax)


def _von_mises_tuning(rots, wid=1, n_units=50):
    cents = np.linspace(-np.pi, np.pi, n_units + 1)[:-1]
    rep = np.exp(np.cos(np.radians(rots)[:, None] - cents[None]) / wid)
    return rep


def summarize_view_results(
    mask_func,
    info,
    *reps,
    field="xPosition",
    thresh=500,
    axs=None,
    f=None,
    fwid=3,
    color=None,
    add_rotation=True,
    rotation_weight=1000,
    rotation_key="rotation_rounded",
    **kwargs,
):
    mask, mask_not = mask_func(info)
    if axs is None:
        f, axs = plt.subplots(len(reps), 2, figsize=(fwid * 2, fwid * len(reps)))
    for i, rep in enumerate(reps):
        targ = info[field] > thresh
        if len(rep.shape) > 2:
            rep = np.reshape(rep, (rep.shape[0], -1))

        if add_rotation:
            rot = _von_mises_tuning(info[rotation_key].to_numpy(), n_units=rep.shape[1])
            rep = np.concatenate((rep, rot), axis=1)

        out_gen = generalize_feature_masks(
            rep,
            targ,
            mask,
            mask_not,
            **kwargs,
        )
        visualize_tr_gen_condition_errors(info, mask, mask_not, out_gen, ax=axs[i, 1])
        gpl.violinplot(
            [out_gen["test_score"], out_gen["gen"]],
            [0, 1],
            ax=axs[i, 0],
            color=(color, color),
        )
        gpl.add_hlines(0.5, axs[i, 0])
        gpl.clean_plot(axs[i, 0], 0)
        gpl.clean_plot(axs[i, 1], 1)
        gpl.clean_plot_bottom(axs[i, 1])
        if i < len(reps) - 1:
            gpl.clean_plot_bottom(axs[i, 0])
        axs[i, 0].set_xticks([0, 1])
        axs[i, 0].set_xticklabels(["within", "across"])
    return f, axs
