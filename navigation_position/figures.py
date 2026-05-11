import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as skm

import general.paper_utilities as pu
import general.plotting as gpl
import general.utility as u
import navigation_position.auxiliary as npa
import navigation_position.analysis.view as npav
import navigation_position.analysis.representations as npra
import navigation_position.visualization as npv
import navigation_position.analysis.change as npac

config_path = "navigation_position/navigation_position/figures.conf"


class NavigationFigure(pu.Figure):
    def __init__(self, fsize, fig_key, *args, **kwargs):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]

        self.params = params
        super().__init__(fsize, params, *args, **kwargs)

    def load_all_data(self):
        if self.data.get("exper_data") is None:
            data_full = npa.load_sessions()
            self.data["exper_data"] = data_full
        return self.data["exper_data"]

    def load_date_data(self, date):
        if self.data.get("exper_data") is None:
            data_full = npa.load_dated_session(date)
            self.data["exper_data"] = data_full
        return self.data["exper_data"]

    def get_uninstructed_data(self):
        data = self.get_exper_data()
        return npa.mask_uninstructed_trials(data)

    def get_instructed_data(self):
        data = self.get_exper_data()
        return npa.mask_uninstructed_trials(data, targ=1)

    def get_correct_data(self):
        data = self.get_exper_data()
        return npa.mask_completed_trials(data, correct_only=True)

    def get_exper_data(self):
        data = self.load_all_data()
        return data

    def get_trial_string_data(self):
        match self.trial_string:
            case "uninstructed":
                data = self.get_uninstructed_data()
            case "instructed":
                data = self.get_instructed_data()
            case "correct":
                data = self.get_correct_data()
            case "all":
                data = self.get_exper_data()
        return data


class ViewFigure(pu.Figure):
    def __init__(
        self,
        fig_key="view_fig",
        **kwargs,
    ):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        fsize = (7, 5)

        self.params = params
        super().__init__(fsize, params, **kwargs)

    def make_gss(self):
        gss = {}
        dec_grid = pu.make_mxn_gridspec(
            self.gs,
            2,
            3,
            0,
            100,
            0,
            100,
            4,
            4,
        )
        dec_axs = self.get_axs(
            dec_grid, squeeze=False, sharex="vertical", sharey="vertical"
        )
        gss["panel_view_decoding"] = dec_axs

        self.gss = gss

    def panel_old_view_decoding(self):
        folder = self.params.get("folder_old")
        if self.params.getboolean("use_training_spec_old"):
            spec = npa.training_spec_file
        else:
            spec = npa.default_spec_file
        color = self.params.getcolor("old_color")
        self._view_decoding(folder, spec, "old", ax_cols=(0, 1), color=color)

    def panel_new_view_decoding(self):
        folder = self.params.get("folder_new")
        if self.params.getboolean("use_training_spec_new"):
            spec = npa.training_spec_file
        else:
            spec = npa.default_spec_file
        color = self.params.getcolor("new_color")
        self._view_decoding(folder, spec, "new", ax_cols=(0, 2), color=color)

    def _view_decoding(self, folder, spec, kind, ax_cols=(0, 1), color=None):
        key = "panel_view_decoding"
        axs = self.gss[key][:, ax_cols]
        if self.data.get((key, kind)) is None:
            img_size = self.params.getint("img_size")
            imgs, img_info = npa.load_views_session(
                folder,
                spec_file_template=spec,
            )
            reps = npav.get_network_view_representations(imgs)
            imgs_low = tf.image.resize_with_pad(imgs, img_size, img_size).numpy()
            self.data[(key, kind)] = (img_info, reps, imgs_low)
        img_info, reps, imgs_low = self.data[(key, kind)]

        pca = self.params.getfloat("pca")
        npav.summarize_view_results(
            npav.sixteen_condition_mask,
            img_info,
            reps,
            imgs_low,
            pca=pca,
            axs=axs,
            color=color,
        )


class ChangeOfMindBehavior(NavigationFigure):
    def __init__(self, trial_string="uninstructed", fig_key="com_behavior", **kwargs):
        fsize = (8, 8)
        self.trial_string = trial_string
        super().__init__(fsize, fig_key, **kwargs)

    def make_gss(self):
        gss = {}

        gss["panel_behavioral_traj"] = self.get_axs(
            pu.make_mxn_gridspec(self.gs, 2, 1, 0, 100, 0, 40, 5, 5),
            squeeze=True,
        )
        gss["panel_stats"] = self.get_axs(
            pu.make_mxn_gridspec(self.gs, 1, 3, 0, 40, 45, 100, 0, 8),
            squeeze=True,
        )
        self.gss = gss

    def panel_behavioral_traj(self, recompute=False):
        key = "panel_behavioral_traj"
        ax_pos, ax_dec = self.gss[key]

        sess_ind = self.params.getint("traj_session")

        time_start = self.params.get("time_zero_field")
        time_begin = self.params.getfloat("time_begin")
        time_end = self.params.getfloat("time_end")
        window = self.params.getfloat("time_window")
        binstep = self.params.getfloat("binstep")
        choice_field = self.params.get("choice_field")
        corr_field = self.params.get("correct_field")
        data = self.get_trial_string_data()

        if self.data.get(key) is None or recompute:
            corr_trls = list(x.to_numpy() for x in data[corr_field])
            out = npac.template_change_of_mind(
                data,
                time_start=time_start,
                time_begin=time_begin,
                time_end=time_end,
                window=window,
                binstep=binstep,
                choice_field=choice_field,
            )
            self.data[key] = out + (corr_trls,)
        masks, times, out_bhv, xs, corr_trls = self.data[key]
        xy = out_bhv[sess_ind]["X"][:, :2]
        side = out_bhv[sess_ind]["y"]
        l_color = "r"
        r_color = "b"
        corr_lw = 0.1
        corr_alpha = 0.5
        npac.plot_traj(
            xy, side, colors=(l_color, r_color), ax=ax_pos, lw=corr_lw, alpha=corr_alpha
        )
        npac.plot_traj(
            xy[masks[sess_ind]],
            side[masks[sess_ind]],
            colors=(l_color, r_color),
            ax=ax_pos,
        )
        ts = (times - data[time_start])[sess_ind][masks[sess_ind]].to_numpy()
        trl_inds = np.where(masks[sess_ind])[0]
        x_inds = np.argmin(np.abs(ts[:, None] - xs[None]), axis=1)
        ax_pos.plot(xy[trl_inds, 0, x_inds], xy[trl_inds, 1, x_inds], "o", color="k")
        # to plot decision time
        # decision_ind = np.argmin(np.abs(xs))
        # ax.plot(xy[:, 0, decision_ind], xy[:, 1, decision_ind], "o", color=(.8,) * 3)
        gpl.clean_plot(ax_pos, 0)
        gpl.make_xaxis_scale_bar(ax_pos, magnitude=1)
        gpl.make_yaxis_scale_bar(ax_pos, magnitude=1)
        ax_pos.set_aspect("equal")

        tp = np.squeeze(out_bhv[sess_ind]["test_projection"])
        ax_dec.plot(xs, tp[side == 0].T, color=l_color, lw=corr_lw, alpha=corr_alpha)
        ax_dec.plot(xs, tp[side == 1].T, color=r_color, lw=corr_lw, alpha=corr_alpha)

        tp_com = tp[masks[sess_ind]]
        side_com = side[masks[sess_ind]]
        ax_dec.plot(xs, tp_com[side_com == 0].T, color=l_color, lw=corr_lw)
        ax_dec.plot(xs, tp_com[side_com == 1].T, color=r_color, lw=corr_lw)
        ax_dec.plot(xs[x_inds], tp[trl_inds, x_inds], "o", color="k")
        gpl.make_xaxis_scale_bar(ax_dec, magnitude=200)
        gpl.make_yaxis_scale_bar(ax_dec, magnitude=2)
        gpl.clean_plot(ax_dec, 0)

    def panel_stats(self, recompute=False):
        key = "panel_stats"
        ax_frac, ax_corr, ax_time = self.gss[key]
        data_key = "panel_behavioral_traj"

        if self.data.get(data_key) is None or recompute:
            self.panel_behavioral_traj(recompute=recompute)
        masks, times, out_bhv, xs, corr_trls = self.data[data_key]
        data = self.get_trial_string_data()
        fracs = np.array(list(np.mean(x) for x in masks))
        gpl.plot_trace_werr([0], fracs[:, None], conf95=True, ax=ax_frac)
        ax_frac.scatter(np.zeros_like(fracs), fracs)
        gpl.add_hlines(0, ax_frac)

        stay_corr = np.array(
            list(np.mean(x[~masks[i]]) for i, x in enumerate(corr_trls))
        )
        com_corr = np.array(list(np.mean(x[masks[i]]) for i, x in enumerate(corr_trls)))
        corrs = np.stack((stay_corr, com_corr), axis=1)
        ax_corr.plot([0, 1], corrs.T)
        gpl.add_hlines(0.5, ax_corr)
        gpl.clean_plot(ax_corr, 0)

        time_start = self.params.get("time_zero_field")
        times = times - data[time_start]
        comb_times = np.concatenate(
            list(t[masks[i]].to_numpy() for i, t in enumerate(times))
        )
        ax_time.hist(comb_times)
        gpl.clean_plot(ax_time, 0)


class ChangeOfMindNeural(NavigationFigure):
    def __init__(
        self,
        trial_string="uninstructed",
        region_dict=npra.default_region_dict,
        fig_key="com_neural",
        **kwargs,
    ):
        fsize = (5, 8)
        self.region_dict = region_dict
        self.trial_string = trial_string
        super().__init__(fsize, fig_key, **kwargs)

    def make_gss(self):
        gss = {}

        gss["panel_neural_traj"] = self.get_axs(
            pu.make_mxn_gridspec(self.gs, 1, 2, 0, 40, 0, 100, 0, 5),
            squeeze=True,
            sharey="all",
        )
        gss["panel_neural_stats"] = self.get_axs(
            pu.make_mxn_gridspec(self.gs, 3, 1, 45, 100, 55, 100, 5, 5),
            sharex="all",
            squeeze=True,
        )
        self.gss = gss

    def panel_neural_traj(self, recompute=False):
        key = "panel_neural_traj"
        axs = self.gss[key]

        sess_ind = self.params.getint("traj_session")

        time_start = self.params.get("time_zero_field")
        time_begin = self.params.getfloat("time_begin")
        time_end = self.params.getfloat("time_end")

        time_begin_short = self.params.getfloat("time_begin_short")
        time_end_short = self.params.getfloat("time_end_short")

        region = self.params.get("region")

        window = self.params.getfloat("time_window")
        binstep = self.params.getfloat("binstep")
        choice_field = self.params.get("choice_field")
        # corr_field = self.params.get("correct_field")
        data = self.get_trial_string_data()

        if self.data.get(key) is None or recompute:
            out_bhv = npac.template_change_of_mind(
                data,
                time_start=time_start,
                time_begin=time_begin,
                time_end=time_end,
                window=window,
                binstep=binstep,
                choice_field=choice_field,
            )
            times = out_bhv[1]
            out_dict = {}
            for k, regions in self.region_dict.items():
                out_neur_tzf = npac.change_of_mind_populations(
                    data,
                    time_start=time_start,
                    time_begin=time_begin,
                    time_end=time_end,
                    window=window,
                    binstep=binstep,
                    choice_field=choice_field,
                    regions=regions,
                )
                out_neur_com_tz = npac.change_of_mind_populations(
                    data,
                    time_start=time_start,
                    time_begin=time_begin_short,
                    time_end=time_end_short,
                    window=window,
                    binstep=binstep,
                    choice_field=choice_field,
                    time_zeros=times,
                    regions=regions,
                )
                out_dict[k] = out_neur_tzf, out_neur_com_tz
            self.data[key] = out_bhv, out_dict
        masks, times, out_bv, xs = self.data[key][0]
        pops_tzf, xs_tzf_r = self.data[key][1][region][0]

        proj = np.squeeze(pops_tzf[sess_ind]["test_projection"])
        targ = pops_tzf[sess_ind]["y"]
        com_mask = masks[sess_ind]
        npac.plot_com_heatmap_and_averages(proj, targ, xs_tzf_r, com_mask, ax=axs[0])
        # npac.plot_flux_heatmap(proj[~com_mask], targ[~com_mask], xs_tzf_r, ax=axs[0])
        m1 = np.logical_and(targ == 0, com_mask)
        m2 = np.logical_and(targ == 1, com_mask)
        eg_lw = 0.3
        axs[0].plot(xs_tzf_r, proj[m1].T, color="m", lw=eg_lw)
        axs[0].plot(xs_tzf_r, proj[m2].T, color="g", lw=eg_lw)
        gpl.make_xaxis_scale_bar(axs[0], 200)
        gpl.make_yaxis_scale_bar(axs[0], 2)

        pops_tz, xs_tz_r = self.data[key][1][region][1]
        proj = np.squeeze(pops_tz[sess_ind]["test_projection"])
        targ = pops_tz[sess_ind]["y"]
        com_mask = masks[sess_ind]
        npac.plot_com_heatmap_and_averages(
            proj, targ, xs_tz_r, com_mask, n_x_bins=15, ax=axs[1]
        )
        m1 = np.logical_and(targ == 0, com_mask)
        m2 = np.logical_and(targ == 1, com_mask)
        axs[1].plot(xs_tz_r, proj[m1].T, color="m", lw=eg_lw)
        axs[1].plot(xs_tz_r, proj[m2].T, color="g", lw=eg_lw)

        gpl.make_xaxis_scale_bar(axs[1], 200)
        gpl.make_yaxis_scale_bar(axs[1], 2)

    def panel_neural_stats(
        self,
        recompute=False,
    ):
        key = "panel_neural_stats"
        ax_avg, ax_std, ax_std_null = self.gss[key]
        region = self.params.get("region")

        key_data = "panel_neural_traj"
        if self.data.get(key_data) is None or recompute:
            self.panel_neural_traj(recompute=recompute)
        masks = self.data[key_data][0][0]
        pops_app, xs_r_app = self.data[key_data][1][region][0]
        pops_com, xs_r = self.data[key_data][1][region][0]
        masks = list(x for i, x in enumerate(masks) if pops_app[i] is not None)
        pops_app = list(x for x in pops_app if x is not None)
        pops_com = list(x for x in pops_com if x is not None)

        n_var_window = self.params.getint("var_window_width")
        n_ts = pops_com[0]["X"].shape[-1]

        n_nulls = 100
        avgs = np.zeros((2, len(pops_com), n_ts))
        std_tc = np.zeros((2, len(pops_com), n_ts - n_var_window + 1))
        std_null_tc = np.zeros_like(std_tc)
        std_null_tc = np.zeros((n_nulls,) + std_tc.shape)
        xs_app_mask = np.logical_and(xs_r_app >= xs_r[0], xs_r_app <= xs_r[-1])
        for i, pop in enumerate(pops_com):
            r_i = pop["X"]
            r_i_app = pops_app[i]["X"][..., xs_app_mask]
            p_i = np.squeeze(pop["test_projection"])
            p_i_app = np.squeeze(pops_app[i]["test_projection"])[..., xs_app_mask]
            m_i = masks[i]
            rng = np.random.default_rng()
            null_vec = u.make_unit_vector(
                rng.normal(0, 1, size=(n_nulls, r_i_app.shape[1]))
            )[:, None, :, None]

            p_null_i = np.sum(r_i[None] * null_vec, axis=-2)
            std_null_tc[:, 0, i], _ = npac.compute_traj_var(
                p_null_i[:, m_i],
                xs_r,
                n_win=n_var_window,
            )
            std_null_tc[:, 1, i], _ = npac.compute_traj_var(
                p_null_i[:, ~m_i],
                xs_r,
                n_win=n_var_window,
            )

            std_tc[0, i], xs_conv = npac.compute_traj_var(
                p_i[m_i], xs_r, n_win=n_var_window
            )
            std_tc[1, i], xs_conv = npac.compute_traj_var(
                p_i_app[~m_i], xs_r, n_win=n_var_window
            )
            avgs[0, i] = npac.compute_avg_activity(r_i[m_i])
            avgs[1, i] = npac.compute_avg_activity(r_i_app[~m_i])
        sess_lw = .4
        l_ = gpl.plot_trace_werr(xs_conv, std_tc[0], ax=ax_std)
        ax_std.plot(xs_conv, std_tc[0].T, color=l_[0].get_color(), lw=sess_lw)
        l_ = gpl.plot_trace_werr(xs_conv, std_tc[1], ax=ax_std)
        ax_std.plot(xs_conv, std_tc[1].T, color=l_[0].get_color(), lw=sess_lw)

        std_null_tc = np.mean(std_null_tc, axis=0)
        l_ = gpl.plot_trace_werr(xs_conv, std_null_tc[0], ax=ax_std_null)
        ax_std_null.plot(xs_conv, std_null_tc[0].T, color=l_[0].get_color(), lw=sess_lw)
        l_ = gpl.plot_trace_werr(xs_conv, std_null_tc[1], ax=ax_std_null)
        ax_std_null.plot(xs_conv, std_null_tc[1].T, color=l_[0].get_color(), lw=sess_lw)

        l_ = gpl.plot_trace_werr(xs_r, avgs[0], ax=ax_avg)
        ax_avg.plot(xs_r, avgs[0].T, color=l_[0].get_color(), lw=sess_lw)
        l_ = gpl.plot_trace_werr(xs_r, avgs[1], ax=ax_avg)
        ax_avg.plot(xs_r, avgs[1].T, color=l_[0].get_color(), lw=sess_lw)
        gpl.add_vlines(0, ax_avg)
        gpl.add_vlines(0, ax_std)


class FixationAnalysis(NavigationFigure):
    def __init__(
        self,
        date=None,
        fig_key="fixation_fig",
        dec_keys=("chose_right", "white_right", "pink_right"),
        trial_string="uninstructed",
        regions=None,
        fixations=(-1, 0, 1, 2),
        balance_correct=False,
        **kwargs,
    ):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        fsize = (5, 12)

        self.trial_string = trial_string
        self.params = params
        self.date = date
        self.regions = regions
        self.dec_keys = dec_keys
        self.fixations = fixations
        if balance_correct:
            self.balance_field = "correct_trial"
        else:
            self.balance_field = None
        super().__init__(fsize, params, **kwargs)

    def make_gss(self):
        gss = {}

        n_axs = len(self.fixations)
        fix_grid = pu.make_mxn_gridspec(
            self.gs,
            1,
            n_axs,
            0,
            20,
            0,
            100,
            2,
            2,
        )
        gss["panel_fixations"] = self.get_axs(
            fix_grid, sharex="all", sharey="all", squeeze=True
        )

        n_plots = len(self.dec_keys)

        dec_ax = self.get_axs((self.gs[25:45, :55],), squeeze=False)[0, 0]
        gen_grid = pu.make_mxn_gridspec(
            self.gs,
            1,
            n_plots,
            50,
            65,
            0,
            100,
            4,
            5,
        )
        gen_axs = self.get_axs(gen_grid, sharex="all", sharey="all", squeeze=True)

        gss["panel_dec"] = dec_ax, gen_axs

        cf_grid = pu.make_mxn_gridspec(
            self.gs,
            1,
            n_plots,
            80,
            100,
            0,
            100,
            4,
            3,
        )
        cf_axs = self.get_axs(cf_grid, sharex="all", sharey="all", squeeze=True)
        gss["panel_cross_fixation"] = cf_axs

        eye_ax = self.get_axs(
            (self.gs[25:45, 60:100],), squeeze=False, share_ax_y=gss["panel_dec"][0]
        )[0, 0]
        gss["panel_eye_decoding"] = eye_ax
        self.gss = gss

    def get_exper_data(self):
        if self.date is not None:
            data = self.load_date_data(self.date)
        else:
            data = self.load_all_data()
        return data

    def _make_full_key(self, key):
        return (
            key,
            tuple(self.fixations),
            tuple(self.dec_keys),
            tuple(self.regions) if u.check_list(self.regions) else self.regions,
        )

    def panel_fixations(self):
        key = "panel_fixations"
        axs = self.gss[key]

        eyebound = 15
        cmap = "magma"
        colors = plt.get_cmap(cmap)(np.linspace(0.2, 0.9, len(self.fixations)))

        full_key = self._make_full_key(key)
        if self.data.get(full_key) is None:
            outs = npra.get_fixation_pops(
                self.get_trial_string_data(),
                self.fixations,
                self.dec_keys,
                combine_func=np.stack,
                regions=self.regions,
            )
            self.data[full_key] = outs[0]
        out = self.data[full_key]
        xy_pos = out["end_xy"]
        prev = None
        for i, xy_i in enumerate(xy_pos):
            if prev is not None:
                axs[i].scatter(*prev.T, s=1, color=colors[i - 1], alpha=0.5)
                comb = np.stack((prev.T, xy_i.T), axis=1)
                axs[i].plot(*comb, color=colors[i - 1], alpha=0.5, lw=0.1, zorder=-1)

                xy1 = np.nanmedian(prev, axis=0).T
                xy2 = np.nanmedian(xy_i, axis=0).T
                axs[i].plot(*xy2, "o", color="k")
                axs[i].annotate(
                    "",
                    xy1,
                    xytext=xy2,
                    arrowprops=dict(arrowstyle="<-"),
                )
            axs[i].scatter(*xy_i.T, s=1, color=colors[i])
            prev = xy_i
            gpl.clean_plot(axs[i], 0)
        for ax in axs:
            ax.set_xlim([-eyebound, eyebound])
            ax.set_ylim([-eyebound, eyebound])
            gpl.make_xaxis_scale_bar(ax, magnitude=5)
            gpl.make_yaxis_scale_bar(ax, magnitude=5)

    def get_color_feature_dict(self):
        features = ("white_right", "pink_right", "chose_right")
        return {f: self.params.getcolor("{}_color".format(f)) for f in features}

    def panel_dec(self, recompute=False):
        key = "panel_dec"
        ax, axs_gen = self.gss[key]
        colors = self.get_color_feature_dict()

        full_key = self._make_full_key(key)
        if self.data.get(full_key) is None or recompute:
            data = self.get_trial_string_data()
            out = {}
            for k in self.dec_keys:
                dec_k = npra.decode_strict_fixation_seq(
                    data,
                    data[k],
                    model=skm.LinearSVC,
                    n=len(self.fixations) - 1,
                    regions=self.regions,
                    balance_field=self.balance_field,
                )
                gen_k = npra.generalize_strict_fixation_pops(
                    dec_k,
                    data[k],
                )
                out[k] = dec_k, gen_k
            self.data[full_key] = out

        res = self.data[full_key]
        maxes = []
        for _, res_k in res.values():
            try:
                vmax_k = np.max(
                    np.mean(
                        np.stack(list(i for i in res_k if i is not None), axis=0),
                        axis=(0, -1, -2),
                    )
                )
            except ValueError:
                vmax_k = 0.5
            maxes.append(vmax_k)
        vmax = np.max(maxes)
        if self.regions is None:
            rs = ("all regions",)
        else:
            rs = self.regions
        ax.set_xlabel("-".join(rs))
        for i, k in enumerate(self.dec_keys):
            dec, gen = res[k]
            gens = []
            for j, dec_j in enumerate(dec):
                if dec_j is not None:
                    if len(gens) == 0:
                        label = k
                    else:
                        label = ""
                    npv.plot_dec_fix_seq(
                        dec_j["score"],
                        ax=ax,
                        label=label,
                        color=colors[k],
                    )
                    gens.append(gen[j])

            ax.set_ylabel("decoding performance")
            if len(gens) > 0:
                gen_plot = np.mean(np.stack(gens, axis=0), axis=(0, -1, -2))
                m = gpl.pcolormesh(
                    gen_plot,
                    cmap="Blues",
                    vmin=0.5,
                    vmax=vmax,
                    ax=axs_gen[i],
                )
            if i == 0:
                axs_gen[i].set_ylabel("trained saccade")
            axs_gen[i].set_xlabel("tested saccade")
            axs_gen[i].set_aspect("equal")
        plt.colorbar(m, ax=axs_gen, label="decoding performance")

    def panel_cross_fixation(self):
        key = "panel_cross_fixation"
        axs = self.gss[key]

        if self.data.get(key) is None:
            data = self.get_trial_string_data()
            out = npra.decode_strict_side_fixations(
                data,
                self.fixations,
                keys=self.dec_keys,
                regions=self.regions,
                balance_field=self.balance_field,
            )
            self.data[key] = out
        out = self.data[key]

        dec_color = self.params.getcolor("dec_color")
        gen_color = self.params.getcolor("gen_color")

        for i, res_i in enumerate(out):
            npv.visualize_strict_side_fixations(
                res_i,
                axs=axs,
                colors=(dec_color, gen_color),
                add_label=i == 0,
                fill=False,
            )
        for i, ax in enumerate(axs):
            ax.set_xlabel("fixation number")
            ax.set_title(self.dec_keys[i])
            if i == 0:
                ax.set_ylabel("decoding performance")
            gpl.clean_plot(ax, i)

    def get_session_colors(self, n_sessions=None):
        if n_sessions is None:
            n_sessions = len(self.get_exper_data())
        cmap = plt.get_cmap(self.params.get("session_cmap"))
        s_pts = np.linspace(0, 1, n_sessions + 1)[:-1]
        colors = cmap(s_pts)
        return colors

    def panel_eye_decoding(self):
        key = "panel_eye_decoding"
        ax = self.gss[key]

        if self.data.get(key) is None:
            out_white = npra.decode_eye(
                self.get_trial_string_data(),
                self.fixations,
                "white_right",
                regions=self.regions,
                balance_field=self.balance_field,
            )
            out_pink = npra.decode_eye(
                self.get_trial_string_data(),
                self.fixations,
                "pink_right",
                regions=self.regions,
                balance_field=self.balance_field,
            )
            self.data[key] = out_white, out_pink

        offset = 0.1
        white_pts = [0 - offset, 1 - offset]
        pink_pts = [0 + offset, 1 + offset]
        (white_sides, white_views), (pink_sides, pink_views) = self.data[key]
        colors = self.get_session_colors(len(white_sides))
        for i, w_side in enumerate(white_sides):
            if w_side is not None:
                gpl.violinplot(
                    [w_side["score"], white_views[i]["score"]],
                    white_pts,
                    ax=ax,
                    color=(colors[i], colors[i]),
                )
            if pink_sides[i] is not None:
                gpl.violinplot(
                    [pink_sides[i]["score"], pink_views[i]["score"]],
                    pink_pts,
                    ax=ax,
                    color=(colors[i], colors[i]),
                )
        gpl.add_hlines(0.5, ax)
        ax.set_xticks(np.mean((white_pts, pink_pts), axis=0))
        ax.set_xticklabels(("view side", "view target"))
        gpl.clean_plot(ax, 0)
