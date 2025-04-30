import tensorflow as tf

import general.paper_utilities as pu
import general.utility as u
import navigation_position.auxiliary as npa
import navigation_position.analysis.view as npav

config_path = "navigation_position/navigation_position/figures.conf"


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
