import numpy as np
import scipy.special as sps
import functools as ft
import re
import pyro
import pyro.distributions as distribs
import torch
import torch.nn as nn
import awkward as ak
import ragged

import sklearn.preprocessing as skp
import sklearn.model_selection as skms

import general.pyro_utility as gpu
import general.torch.feedforward as gtf
import general.utility as u
import general.neural_analysis as na
import general.plotting as gpl
import navigation_position.auxiliary as npa
import navigation_position.analysis.representations as npra


@gpl.ax_adder()
def plot_pred_traj(xs, preds, targs=None, ax=None, labels=None):
    if targs is None:
        targs = preds
        preds = xs
        xs = np.arange(preds.shape[-1])
    if labels is None:
        labels = (None,) * len(preds[0])
    for j, pred_ij in enumerate(preds):
        targ_ij = targs[j]
        l_ = ax.plot(xs, pred_ij, ls="dashed", label=labels[j])
        ax.plot(xs, targ_ij, color=l_[0].get_color())
    ax.legend(frameon=False)


def equalize_last_dim(x, conv_func=None):
    arrs = []
    dims = []
    for i, arr in enumerate(x):
        if conv_func is not None:
            arr = conv_func(arr)
        di = arr.shape[-1]
        arrs.append(arr)
        dims.append(di)
    md = np.max(dims)
    diffs = md - np.array(dims)
    new_arrs = []
    for i, arr in enumerate(arrs):
        dims = ((0, 0),) * (len(arr.shape) - 1) + ((0, diffs[i]),)
        new_arrs.append(np.pad(arr, dims, constant_values=np.nan))
    return np.stack(new_arrs, axis=0)


def convert_ragged_to_numpy(x):
    return equalize_last_dim(x, conv_func=ak.to_numpy)


@gpl.ax_adder()
def plot_most_likely_corr_ragged(preds, targs, **kwargs):
    pred_all = []
    targ_all = []
    for i, pred_i in enumerate(preds):
        pred_i = convert_ragged_to_numpy(pred_i)
        targ_i = convert_ragged_to_numpy(targs[i])
        pred_all.append(pred_i)
        targ_all.append(targ_i)
    preds = equalize_last_dim(pred_all)
    targs = equalize_last_dim(targ_all)
    return plot_most_likely_corr(preds, targs, **kwargs)


@gpl.ax_adder()
def plot_most_likely_corr_diff(
    preds,
    targs,
    ax=None,
    gray_color=(0.8,) * 3,
    chance_ci=False,
    xs=None,
    chance_ls="solid",
    **kwargs,
):
    corr = np.zeros((len(preds), preds[0].shape[-1]))
    chance = np.zeros((preds[0].shape[-2], len(preds), preds[0].shape[-1]))
    for i, pred in enumerate(preds):
        pred_ind = np.argmax(pred, axis=-2)
        targ_ind = np.argmax(targs[i], axis=-2)
        corr[i] = np.mean((pred_ind == targ_ind).astype(float), axis=0)
        for j in range(pred.shape[-2]):
            chance[j, i] = np.mean(np.argmax(targs[i], axis=-2) == j, axis=0)
    if xs is None:
        xs = np.arange(corr.shape[-1])
    gpl.plot_trace_werr(xs, corr, confstd=True, ax=ax, **kwargs)
    for c_i in chance:
        ax.plot(xs, np.mean(c_i, axis=0), color=gray_color, ls=chance_ls)


@gpl.ax_adder()
def plot_most_likely_corr(
    xs,
    preds,
    targs=None,
    ax=None,
    gray_color=(0.9,) * 3,
    chance_ls="solid",
    chance=None,
    thr=None,
    chance_dims=(0, 1),
    **kwargs,
):
    if targs is None:
        targs = preds
        preds = xs
        xs = np.arange(preds.shape[-1])
    if chance is None:
        chance = 1 / preds.shape[-2]
    if thr is not None:
        mask = np.all(preds < thr, axis=-2)
    pred_ind = np.argmax(preds, axis=-2)
    targ_ind = np.argmax(targs, axis=-2)
    nan_mask = np.all(np.isnan(targs), axis=-2)
    corr = (pred_ind == targ_ind).astype(float)
    corr[nan_mask] = np.nan
    if thr is not None:
        corr[mask] = np.nan
    mu_corr = np.nanmean(corr, axis=-2)
    gpl.plot_trace_werr(xs, mu_corr, ax=ax, confstd=True, **kwargs)
    gpl.add_hlines(chance, ax)
    for i in range(targs.shape[-2]):
        occur_i = (targ_ind == i).astype(float)
        occur_i[nan_mask] = np.nan
        ax.plot(
            xs,
            np.nanmean(occur_i, axis=chance_dims),
            color=gray_color,
            ls=chance_ls,
        )


@gpl.ax_adder()
def plot_correct_thrs(
    preds,
    targs,
    ax=None,
    thr_range=(1e-5, 3),
    n_thr=100,
    chance=None,
    min_count=2,
    **kwargs,
):
    thrs = np.linspace(*thr_range, n_thr)
    n_folds = preds.shape[0]
    corr_te = np.zeros((n_folds, n_thr))
    if chance is None:
        chance = 1 / preds.shape[-2]
    for i in range(n_folds):
        for j, thr in enumerate(thrs):
            mask_ij = preds[i] > thr
            if np.sum(mask_ij) < min_count:
                corr_te[i, j] = np.nan
            else:
                corr_te[i, j] = np.mean(targs[i][mask_ij])
    gpl.plot_trace_werr(thrs, corr_te, confstd=True, ax=ax, **kwargs)
    gpl.add_hlines(chance, ax)


COND_CENTS_X = (
    (-1, 0, 1, 0),
    (1, 0, 1, 0),
    (-1, 0, -1, 0),
    (1, 0, -1, 0),
)
COND_CENTS_Y = (
    (0, -1, 0, 1),
    (0, 1, 0, 1),
    (0, -1, 0, -1),
    (0, 1, 0, -1),
)
COND_CENTS_FULL = (
    (-1, 1, 1, 0),
    (-1, -1, 1, 0),
    (-1, 1, 0, 1),
    (-1, -1, 0, 1),
    (-1, 1, -1, 0),
    (-1, -1, -1, 0),
    (-1, 1, 0, -1),
    (-1, -1, 0, -1),
    (1, 1, 1, 0),
    (1, -1, 1, 0),
    (1, 1, 0, 1),
    (1, -1, 0, 1),
    (1, 1, -1, 0),
    (1, -1, -1, 0),
    (1, 1, 0, -1),
    (1, -1, 0, -1),
)
COND_CENTS_FULL_ALT = (
    (-1, 0, 1, 0),
    (0, -1, 1, 0),
    (-1, 0, 0, 1),
    (0, -1, 0, 1),
    (-1, 0, -1, 0),
    (0, -1, -1, 0),
    (-1, 0, 0, -1),
    (0, -1, 0, -1),
    (1, 0, 1, 0),
    (0, 1, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 0, 1),
    (1, 0, -1, 0),
    (0, 1, -1, 0),
    (1, 0, 0, -1),
    (0, 1, 0, -1),
)
COND_CENTS_POSX = (
    (-1, 0, 0, 0),
    (1, 0, 0, 0),
)
COND_CENTS_POSY = (
    (0, -1, 0, 0),
    (0, 1, 0, 0),
)
COND_CENTS_POSXY = (
    (-1, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, -1, 0, 0),
)
COND_CENTS_ROTX = (
    (0, 0, 1, 0),
    (0, 0, -1, 0),
)
COND_CENTS_ROTY = (
    (0, 0, 0, 1),
    (0, 0, 0, -1),
)


def recode_hd_position_soft_jagged(pos_tc, hd_tc, **kwargs):
    out = []
    for i, pos_tc_i in enumerate(pos_tc):
        pos_tc_i = ak.to_numpy(pos_tc_i)
        hd_tc_i = ak.to_numpy(hd_tc[i])
        out.append(recode_hd_position_soft(pos_tc_i[None], hd_tc_i[None], **kwargs)[0])
    return ragged.array(out)


def recode_hd_position_soft(
    pos_tc,
    hd_tc,
    cond_cents=COND_CENTS_X,
    sub=500,
    div=40,
    temperature=1,
):
    pos_tc = (pos_tc - sub) / div
    conds = np.expand_dims(np.concatenate((pos_tc, hd_tc), axis=1), 1)
    cents = np.expand_dims(cond_cents, (0, -1))
    dists = np.sum((conds - cents) ** 2, axis=2)
    probs = sps.softmax(-dists * temperature, axis=1)
    return probs


def recode_hd_position(
    pos_tc,
    hd_tc,
    binary_ind=None,
    pos_thr=500,
    rot_cents=(45, 135, 225, 315),
):
    discret_hd = npa.discretize_rotation_quadrants_tc(hd_tc, cents=rot_cents)
    discret_pos = pos_tc > pos_thr
    if binary_ind is None:
        dp = u.combine_dimensions(discret_pos, 0, -1)
        _, ind = np.unique(dp, axis=0, return_inverse=True)
        discret_pos = u.uncombine_dimensions(ind, 0, -1, discret_pos.shape[-1])
    else:
        discret_pos = discret_pos[:, binary_ind]
    hd_oh = onehot_y(discret_hd)
    pos_oh = onehot_y(discret_pos)
    m = skp.PolynomialFeatures(degree=(2, 2), interaction_only=True, include_bias=False)
    comb_feats = np.concatenate((pos_oh, hd_oh), axis=1)
    flat_feats = u.combine_dimensions(comb_feats, 0, -1)
    feats = m.fit_transform(flat_feats)
    mask = np.var(feats, axis=0) > 0
    feats = feats[:, mask]
    return u.uncombine_dimensions(feats, 0, -1, comb_feats.shape[-1])


default_ts_dict = {
    "rotation": (("rotation_tc_sin", "rotation_tc_cos"), None),
    "position": (("pos_x", "pos_y"), None),
    "eye": (("eye_x", "eye_y"), None),
    "joy": (("joy_x", "joy_y"), None),
}


@gpl.ax_adder()
def plot_all_pos(pos_tc, ax=None, len_thr=None):
    ends = []
    for i, tc in enumerate(pos_tc):
        tc = ak.to_numpy(tc)
        if len_thr is None or tc.shape[-1] > len_thr:
            gpl.plot_colored_line(*tc, ax=ax)
            ends.append(tc[:, -1])
    ax.set_xlim([450, 550])
    ax.set_ylim([450, 550])
    gpl.add_hlines(500, ax)
    gpl.add_vlines(500, ax)
    return np.array(ends)


def get_all_jagged_data(*args, **kwargs):
    return get_all_data(*args, **kwargs, jagged=True)


DLC_TEMPLATE = ".*manual.*[xy]$"


def get_dlc_features(data, dlc_template=DLC_TEMPLATE):
    return list(
        filter(lambda x: re.match(dlc_template, x) is not None, data.session_keys)
    )


def get_all_data(
    data,
    binsize=200,
    binstep=100,
    before=0,
    after=0,
    tzf1="nav_start",
    tzf2="nav_end",
    region_dict=npra.default_region_dict,
    ts_dict=default_ts_dict,
    include_dlc=True,
    include_neural=True,
    dlc_template=DLC_TEMPLATE,
    dlc_timing_key="video_frames",
    temperature=2,
    jagged=False,
):
    if include_dlc:
        dlc_fields = get_dlc_features(data, dlc_template=dlc_template)
        ts_dict["dlc"] = (dlc_fields, dlc_timing_key)
    if jagged:
        pop_getter = data.get_jagged_populations
        pop_args = (binsize, tzf1, tzf2)
        pop_kwargs = {"before": before, "after": after, "binstep": binstep}
        ts_kwargs = {"tzf1": tzf1, "tzf2": tzf2, "before": before, "after": after}
    else:
        pop_getter = data.get_populations
        pop_args = (binsize, before, after)
        pop_kwargs = {"binstep": binstep, "time_zero_field": tzf1}
        ts_kwargs = {"begin": before, "end": after, "time_zero_field": tzf1}
    timeseries_all = {}
    for k, (ts, timing_key) in ts_dict.items():
        timeseries_all[k] = data.get_field_timeseries(
            ts,
            timing_key=timing_key,
            binstep=binstep,
            binsize=binsize,
            jagged=jagged,
            **ts_kwargs,
        )

    if include_neural:
        pop_data = {
            k: pop_getter(*pop_args, **pop_kwargs, regions=rs)
            for k, rs in npra.default_region_dict.items()
        }
        timeseries_all.update(pop_data)
    corr = list(x.to_numpy() for x in data["correct_trial"])
    return timeseries_all, corr


def sequence_chopper(*args, length=10, start=0):
    out_seqs = tuple([] for _ in args)
    n_trls = args[0].shape[0]
    for i in range(n_trls):
        splitted = list(False for _ in args)
        for j, arg in enumerate(args):
            arr = ak.to_numpy(arg[i])
            arr = arr[..., start:]
            n_splits = int(np.floor(arr.shape[-1] / length))
            take_pts = n_splits * length
            if n_splits > 0:
                splitted[j] = True
                arr_splits = np.split(arr[..., :take_pts], n_splits, axis=-1)
                out_seqs[j].append(np.stack(arr_splits, axis=0))
        assert np.all(splitted) or np.all(np.logical_not(splitted))
    out_seqs = list(np.concatenate(seq, axis=0) for seq in out_seqs)
    return out_seqs


def sequence_multi_chopper(*args, length=10, n_steps=3, step_size=None):
    if step_size is None:
        step_size = int(np.floor(length / n_steps))
    n_steps = int(np.floor(length / step_size))
    out_seqs = list([] for _ in range(len(args)))
    for i in range(n_steps):
        seqs = sequence_chopper(*args, length=length, start=step_size * i)
        list(out_seqs[i].append(s) for i, s in enumerate(seqs))
    return list(np.concatenate(s, axis=0) for s in out_seqs)


class GenericCustomDecoder:
    def predict(self, X):
        return self.net.get_detached_output(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def predict_tc(self, X):
        pred = self.net.get_detached_output(u.combine_dimensions(X, 0, -1))
        return u.uncombine_dimensions(pred, 0, -1, X.shape[-1])


def _make_nested_tensor(x):
    return torch.nested.as_nested_tensor(
        list(torch.tensor(x_i) for x_i in x),
        layout=torch.jagged,
    )


class DimCrossEntropyLoss:
    def __init__(self, dim=-1, **kwargs):
        self.dim = dim
        self.loss = nn.CrossEntropyLoss(**kwargs)

    def __call__(self, targ, pred, **kwargs):
        targ_use = torch.swapaxes(targ, 1, self.dim)
        pred_use = torch.swapaxes(pred, 1, self.dim)
        return self.loss(targ_use, pred_use, **kwargs)


class ModuleDecoder(GenericCustomDecoder):
    def __init__(
        self,
        batch_size=50,
        lr=1e-3,
        weight_decay=0.001,
        lr_stepping=False,
        dropout=0.1,
        transform_layer_dim=100,
        is_causal=True,
        # loss_fn=nn.CrossEntropyLoss,
        loss_fn=nn.MSELoss,
        norm=True,
        ragged=False,
        include_positional=False,
        **kwargs,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.lr_stepping = lr_stepping
        self.kwargs = kwargs
        self.dropout = dropout
        self.loss_fn = loss_fn
        self.is_causal = is_causal
        self.transform_layer_dim = transform_layer_dim
        self.norm = norm
        self.ragged = ragged
        self.include_positional = include_positional
        self.weight_decay = weight_decay

    def fit(self, X, y, val_set=None, **kwargs):
        if self.norm:
            if self.ragged:
                self.norm_pipes = list(
                    na.RaggedPipelineTC(norm=True, pca=None) for _ in X
                )
            else:
                self.norm_pipes = list(
                    na.make_model_pipeline(norm=True, pca=None, single_tc=True)
                    for _ in X
                )
            X = list(self.norm_pipes[i].fit_transform(xi) for i, xi in enumerate(X))
        if self.ragged:
            X = list(_make_nested_tensor(xi) for xi in X)
            y = _make_nested_tensor(y)
        X = list(xi.swapaxes(1, 2) for xi in X)
        y = y.swapaxes(1, 2)
        if val_set is not None:
            X_v, y_v = val_set
            if self.norm:
                X_v = list(
                    self.norm_pipes[i].transform(xvi) for i, xvi in enumerate(X_v)
                )
            if self.ragged:
                X_v = list(_make_nested_tensor(x) for x in X_v)
                y_v = _make_nested_tensor(y_v)
            X_v = list(x.swapaxes(1, 2) for x in X_v)
            y_v = y_v.swapaxes(1, 2)
            val_set = (X_v, y_v)

        self.net = gtf.ModuleAttentionNetwork(
            len(X),
            list(xi.shape[-1] for xi in X),
            self.transform_layer_dim,
            y.shape[-1],
            dropout=self.dropout,
            include_positional=self.include_positional,
        )
        optim_kwargs = {"weight_decay": self.weight_decay}
        info = self.net.fit(
            X,
            y,
            lr_stepping=self.lr_stepping,
            batch_size=self.batch_size,
            lr=self.lr,
            val_set=val_set,
            loss=self.loss_fn,
            ragged=self.ragged,
            optim_kwargs=optim_kwargs,
            **self.kwargs,
            **kwargs,
        )
        self.fit_info = info
        return self

    def predict_tc(self, X):
        if self.norm:
            X = list(self.norm_pipes[i].transform(xi) for i, xi in enumerate(X))
        if self.ragged:
            X = list(_make_nested_tensor(xi) for xi in X)
        X = list(xi.swapaxes(1, 2) for xi in X)
        return self.predict(X)


class AttentionDecoder(GenericCustomDecoder):
    def __init__(
        self,
        include_positional=False,
        batch_size=50,
        lr=1e-3,
        lr_stepping=False,
        weight_decay=0.001,
        dropout=0.1,
        transform_layer_dim=100,
        is_causal=True,
        n_heads=1,
        loss_fn=DimCrossEntropyLoss,
        # loss_fn=nn.MSELoss,
        norm=True,
        ragged=False,
        **kwargs,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.lr_stepping = lr_stepping
        self.kwargs = kwargs
        self.include_positional = include_positional
        self.dropout = dropout
        self.loss_fn = loss_fn
        self.is_causal = is_causal
        self.transform_layer_dim = transform_layer_dim
        self.weight_decay = weight_decay
        self.norm = norm
        self.ragged = ragged
        self.n_heads = n_heads

    def fit(self, X, y, val_set=None, **kwargs):
        if self.norm:
            if self.ragged:
                self.norm_pipe = na.RaggedPipelineTC(norm=True, pca=None)
            else:
                self.norm_pipe = na.make_model_pipeline(
                    norm=True, pca=None, single_tc=True
                )
            X = self.norm_pipe.fit_transform(X)
        if self.ragged:
            X = _make_nested_tensor(X)
            y = _make_nested_tensor(y)
        X = X.swapaxes(1, 2)
        y = y.swapaxes(1, 2)
        if val_set is not None:
            if self.norm:
                vs0 = self.norm_pipe.transform(val_set[0])
                val_set = [vs0] + list(val_set[1:])
            if self.ragged:
                val_set = tuple(_make_nested_tensor(x) for x in val_set)
            val_set = tuple(x.swapaxes(1, 2) for x in val_set)

        self.net = gtf.AttentionNetwork(
            X.shape[-1],
            y.shape[-1],
            transform_layer_dim=self.transform_layer_dim,
            include_positional=self.include_positional,
            dropout=self.dropout,
            is_causal=self.is_causal,
            n_heads=self.n_heads,
        )
        info = self.net.fit(
            X,
            y,
            lr_stepping=self.lr_stepping,
            batch_size=self.batch_size,
            lr=self.lr,
            val_set=val_set,
            loss=self.loss_fn,
            ragged=self.ragged,
            optim_kwargs={"weight_decay": self.weight_decay},
            **self.kwargs,
            **kwargs,
        )
        self.fit_info = info
        return self

    def predict_tc(self, X):
        if self.norm:
            X = self.norm_pipe.transform(X)
        if self.ragged:
            X = _make_nested_tensor(X)
        X = X.swapaxes(1, 2)
        return self.predict(X)


class SparseDecoder(GenericCustomDecoder):
    def __init__(
        self,
        l1_weight=5,
        lr=0.01,
        batch_size=200,
        nonlinear=True,
        lr_stepping=False,
        weight_decay=0.001,
        **kwargs,
    ):
        self.lr = lr
        self.nonlinear = nonlinear
        self.batch_size = batch_size
        self.l1_weight = l1_weight
        self.lr_stepping = lr_stepping
        self.kwargs = kwargs
        self.weight_decay = weight_decay

    def fit(self, X, y, **kwargs):
        if self.nonlinear:
            output_function = nn.ReLU
        else:
            output_function = nn.Identity
        self.net = gtf.FeedForwardNetwork(
            X.shape[1],
            (),
            y.shape[1],
            transfer_function=nn.Identity,
            output_function=output_function,
        )

        def custom_loss(l1=self.l1_weight):
            mse = nn.MSELoss(reduction="none")

            def l_(x, y):
                return torch.mean(mse(x, y)) + l1 * torch.mean(
                    mse(x, y) * torch.abs(x) ** 1
                )

            return l_

        info = self.net.fit(
            X,
            y,
            lr_stepping=self.lr_stepping,
            batch_size=self.batch_size,
            loss=custom_loss,
            optim_kwargs={"weight_decay": self.weight_decay},
            **self.kwargs,
            **kwargs,
        )
        self.fit_info = info
        return self


def sparse_decoders_pops(
    X_tc_dict,
    y_tc,
    corr=None,
    num_epochs=20,
    batch_size=1000,
    n_folds=50,
    **kwargs,
):
    out = {}

    for k, (X_tc, xs) in X_tc_dict.items():
        outs_k = []
        for i, X_tc_i in enumerate(X_tc):
            y_tc_i = y_tc[i]
            if corr is not None:
                X_tc_i = X_tc_i[corr[i]]
                y_tc_i = y_tc[i][corr[i]]
            dec_i = sparse_decoder(
                X_tc_i,
                y_tc_i,
                num_epochs=num_epochs,
                batch_size=batch_size,
                n_folds=n_folds,
                **kwargs,
            )
            outs_k.append(dec_i)
        out[k] = (u.aggregate_dictionary(outs_k), xs)
    return out


def transformer_module_decoder(
    X_tc,
    y_tc,
    splitter=skms.ShuffleSplit,
    test_frac=0.1,
    n_folds=10,
    num_epochs=10,
    batch_size=50,
    lr_stepping=False,
    include_positional=False,
    is_causal=True,
    norm=True,
    ragged=False,
    **kwargs,
):
    outs = []
    splitter = splitter(n_folds, test_size=test_frac)
    for tr_inds, te_inds in splitter.split(X_tc[0], y_tc):
        X_tc_tri = list(xi[tr_inds] for xi in X_tc)
        y_tc_tri = y_tc[tr_inds]

        X_tc_tei = list(xi[te_inds] for xi in X_tc)
        y_tc_tei = y_tc[te_inds]

        model = ModuleDecoder(
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr_stepping=lr_stepping,
            is_causal=is_causal,
            norm=norm,
            ragged=ragged,
            include_positional=include_positional,
        )

        model.fit(X_tc_tri, y_tc_tri, val_set=(X_tc_tei, y_tc_tei))
        tr_pred = model.predict_tc(X_tc_tri)
        te_pred = model.predict_tc(X_tc_tei)

        out_i = {}
        out_i.update(model.fit_info)
        out_i["train_preds"] = tr_pred
        out_i["train_targs"] = y_tc_tri
        out_i["test_preds"] = te_pred
        out_i["test_targs"] = y_tc_tei
        out_i["train_inds"] = tr_inds
        out_i["test_inds"] = te_inds
        out_i["model"] = model
        outs.append(out_i)
    return u.aggregate_dictionary(outs)


def onehot_y(y_tc):
    y = u.combine_dimensions(y_tc, 0, -1)
    y = skp.OneHotEncoder().fit_transform(y[:, None]).todense()
    y_tc = u.uncombine_dimensions(np.asarray(y), 0, -1, y_tc.shape[-1])
    return y_tc


default_targ_conds = {
    "pos_x": COND_CENTS_POSX,
    "pos_y": COND_CENTS_POSY,
    "int_x": COND_CENTS_X,
    "int_y": COND_CENTS_Y,
}


def transformer_decoder(
    X_tc,
    y_tc,
    splitter=skms.ShuffleSplit,
    test_frac=0.1,
    n_folds=10,
    num_epochs=10,
    batch_size=50,
    lr_stepping=False,
    include_positional=False,
    is_causal=True,
    norm=True,
    ragged=False,
    chop_length=None,
    n_steps=4,
    **kwargs,
):
    if len(y_tc.shape) == 2 or y_tc.shape[1] == 1:
        y_tc = onehot_y(y_tc)

    outs = []
    splitter = splitter(n_folds, test_size=test_frac)
    for tr_inds, te_inds in splitter.split(X_tc, y_tc):
        X_tc_tri = X_tc[tr_inds]
        y_tc_tri = y_tc[tr_inds]

        X_tc_tei = X_tc[te_inds]
        y_tc_tei = y_tc[te_inds]
        if chop_length is not None:
            X_tc_tri, y_tc_tri = sequence_multi_chopper(
                X_tc_tri,
                y_tc_tri,
                length=chop_length,
                n_steps=n_steps,
            )
            X_tc_tei, y_tc_tei = sequence_multi_chopper(
                X_tc_tei,
                y_tc_tei,
                length=chop_length,
                n_steps=n_steps,
            )

        model = AttentionDecoder(
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr_stepping=lr_stepping,
            include_positional=include_positional,
            is_causal=is_causal,
            norm=norm,
            ragged=ragged,
        )

        model.fit(X_tc_tri, y_tc_tri, val_set=(X_tc_tei, y_tc_tei))
        tr_pred = model.predict_tc(X_tc_tri)
        te_pred = model.predict_tc(X_tc_tei)

        out_i = {}
        out_i.update(model.fit_info)
        out_i["train_preds"] = tr_pred
        out_i["train_targs"] = y_tc_tri
        out_i["test_preds"] = te_pred
        out_i["test_targs"] = y_tc_tei
        out_i["train_inds"] = tr_inds
        out_i["test_inds"] = te_inds
        out_i["model"] = model
        outs.append(out_i)
    return u.aggregate_dictionary(outs)


def sparse_decoder(
    X_tc,
    y_tc,
    splitter=skms.ShuffleSplit,
    test_frac=0.1,
    n_folds=10,
    num_epochs=10,
    batch_size=200,
    pca=0.99,
    l1_weight=1,
    lr_stepping=False,
    norm=True,
    **kwargs,
):
    if len(y_tc.shape) == 2 or y_tc.shape[1] == 1:
        y_tc = onehot_y(y_tc)

    outs = []
    splitter = splitter(n_folds, test_size=test_frac)
    for tr_inds, te_inds in splitter.split(X_tc, y_tc):
        X_tc_tri = X_tc[tr_inds]
        y_tc_tri = y_tc[tr_inds]

        pipe_i = na.make_model_pipeline(pca=pca, norm=norm, single_tc=True, **kwargs)
        X_tc_tri = pipe_i.fit_transform(X_tc[tr_inds])
        X_tri = u.combine_dimensions(X_tc_tri, 0, -1)
        y_tri = u.combine_dimensions(y_tc_tri, 0, -1)

        X_tc_tei = X_tc[te_inds]
        y_tc_tei = y_tc[te_inds]
        X_tc_tei = pipe_i.transform(X_tc_tei)

        model = SparseDecoder(
            num_epochs=num_epochs,
            batch_size=batch_size,
            l1_weight=l1_weight,
            lr_stepping=lr_stepping,
        )
        val_set = (
            u.combine_dimensions(X_tc_tei, 0, -1),
            u.combine_dimensions(y_tc_tei, 0, -1),
        )
        model.fit(X_tri, y_tri, val_set=val_set)
        tr_pred = model.predict_tc(X_tc_tri)
        te_pred = model.predict_tc(X_tc_tei)

        out_i = {}
        out_i.update(model.fit_info)
        out_i["train_preds"] = tr_pred
        out_i["train_targs"] = y_tc_tri
        out_i["test_preds"] = te_pred
        out_i["test_targs"] = y_tc_tei
        out_i["model"] = model
        out_i["pipe"] = pipe_i
        outs.append(out_i)
    return u.aggregate_dictionary(outs)


def session_cond_decoder(
    data_types,
    pos_tc,
    rot_tc,
    corr=None,
    targ_conds=default_targ_conds,
    decoder_func=transformer_decoder,
    num_epochs=50,
    n_folds=30,
    cond_temperature=2,
    ragged_trials=False,
    **kwargs,
):
    if ragged_trials:
        con = ragged.concat
        cond_func = recode_hd_position_soft_jagged
        if "chop_length" not in kwargs.keys():
            raise AttributeError("chop_length must be set for ragged input")
    else:
        con = np.concatenate
        cond_func = recode_hd_position_soft
    n_sessions = len(pos_tc)
    outs = []
    for i in range(n_sessions):
        out_sess_dict = {}
        for j, (k_j, cond_j) in enumerate(targ_conds.items()):
            X_use = con(tuple(x[i] for x in data_types), axis=1)
            corr_si = corr[i] if corr is not None else np.ones(len(X_use), dtype=bool)
            X_use = X_use[corr_si]

            y_pos_pre = pos_tc[i][corr_si]
            y_rot_pre = rot_tc[i][corr_si]

            y_use = cond_func(
                y_pos_pre,
                y_rot_pre,
                temperature=cond_temperature,
                cond_cents=cond_j,
            )

            out_sess_dict[k_j] = decoder_func(
                X_use,
                y_use,
                n_folds=n_folds,
                num_epochs=num_epochs,
                **kwargs,
            )
        outs.append(out_sess_dict)
    return outs


@pyro.infer.config_enumerate
def sparse_generative_model(
    y,
    X=None,
    X_dims=None,
    batch_size=None,
    vecs_hn_width=3,
    alpha_hn_width=0.1,
    hn_width=1,
):
    """fit sparse decoder model

    X(t) ~ a(t) w(y) + \\epsilon
    where
    X(t) is the D-dim neural activity at a single time point t
    a(t) is the attentional weight at t
    w(y(t)) is a D-dim vector, corresponding to the position (y) at t

    a has an exponential prior
    w has a normal prior

    start by assuming t is non-sequential and random
    """
    alpha_prior = pyro.sample("alpha_prior", distribs.HalfNormal(alpha_hn_width))
    alpha_distrib = distribs.Dirichlet(
        torch.cat([1 / alpha_prior.squeeze().unsqueeze(0), torch.ones(1)]),
    )
    eps_sig_mu = pyro.sample("eps_sig_mu", distribs.HalfNormal(hn_width))
    eps_sig_sig = pyro.sample("eps_sig_sig", distribs.HalfNormal(hn_width))
    epsilon = pyro.sample(
        "epsilon",
        distribs.Normal(torch.zeros(X_dims), eps_sig_mu).to_event(1),
    )

    vecs_sig_mu = pyro.sample("vecs_sig_mu", distribs.HalfNormal(vecs_hn_width))
    with pyro.plate("sparse_factors", 2):
        vecs = pyro.sample(
            "vecs",
            distribs.Normal(torch.zeros(X_dims), vecs_sig_mu).to_event(1),
        )
    with pyro.plate("alphas", len(y)):
        alpha = pyro.sample(
            "alpha",
            alpha_distrib,
        )
        print(alpha.shape)
        sig = pyro.sample("sig", distribs.Categorical(alpha))
        print(sig.shape)

    with pyro.plate("data", size=len(y), subsample_size=batch_size) as inds:
        print(sig.shape, alpha.shape, epsilon.shape, vecs.shape)

        mu_spont = epsilon.unsqueeze(0)
        mu_sig = vecs.squeeze()[y[inds]]
        mu_use = (
            sig[inds].unsqueeze(-1) * mu_spont + (1 - sig[inds].unsqueeze(-1)) * mu_sig
        )
        distr = distribs.Normal(mu_use, eps_sig_sig).to_event(1)
        if X is not None:
            X_use = X[inds]
        else:
            X_use = X
        out = pyro.sample("obs", distr, obs=X_use)
        return out


def fit_sparse_generative_model(
    X,
    y,
    n_samps=500,
    fixed_params=None,
    batch_size=200,
    model=sparse_generative_model,
    **kwargs,
):
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.int)
    if fixed_params is None:
        fixed_params = {}
    model = ft.partial(model, batch_size=batch_size, X_dims=X.shape[1], **fixed_params)

    loss = pyro.infer.TraceEnum_ELBO()
    out = gpu.fit_model(
        (y,),
        (X,),
        model,
        loss=loss,
        block_vars=["sig"],
        **kwargs,
    )
    # preds = gpu.sample_fit_model(
    #     (y,),
    #     model_sample,
    #     out["samples"],
    #     out["guide"],
    #     n_samps=n_samps,
    # )
    # out["predictive_samples"] = preds
    return out
