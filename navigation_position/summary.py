
import os

import navigation_position.visualization as npv
import navigation_position.analysis.representations as npra
import navigation_position.auxiliary as npaux


default_regions = {
    "dlPFC": ("DLPFCd", "DLPFCv",), 
    "HPC": ("HPC",),
    "PMd": ("PMd"),
    "all": None,
}


def generate_summary_plots(
        data,
        out_all=None,
        regions=default_regions,
        base_folder=npaux.FIGFOLDER,
        colors=None,
        pop_ind=0,
        **kwargs,
):
    if out_all is None:
        out_all = {}
        for label, r in regions.items():
            out_all[label] = npra.decode_times(data, regions=r, **kwargs)

    neur_regions = data["neur_regions"][pop_ind].iloc[0]

    for region, dec in out_all.items():
        f, axs = npv.visualize_decoding_dict(dec)
        f.suptitle(region)
        fname = os.path.join(base_folder, "decoding_{}.pdf").format(region)
        f.savefig(
            fname, bbox_inches="tight", transparent=True
        )

        if region == "all":
            for i, (time_k, var_decs) in enumerate(dec.items()):
                for j, (var_k, dec_list) in enumerate(var_decs.items()):
                    xs_pops = dec_list[1:]
                    f, _ = npv.plot_pops_all_units(
                        *xs_pops, pop_ind=pop_ind, regions=neur_regions, colors=colors,
                    )
                    fname = os.path.join(base_folder, "units_{}-at-{}.pdf").format(
                        var_k, time_k
                    )

                    f.savefig(
                        fname,
                        bbox_inches="tight", 
                        transparent=True
                    )
    return out_all
