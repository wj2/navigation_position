import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.svm as skm

import general.paper_utilities as pu
import general.plotting as gpl
import general.utility as u
import navigation_position.auxiliary as npa
import navigation_position.analysis.view as npav
import navigation_position.analysis.representations as npra
import navigation_position.visualization as npv

config_path = "navigation_position/navigation_position/figures.conf"


class NavigationFigure(pu.Figure):
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

            gen_plot = np.mean(np.stack(gens, axis=0), axis=(0, -1, -2))
            ax.set_ylabel("decoding performance")
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
