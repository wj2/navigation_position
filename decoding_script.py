import argparse
import pickle
import os
from datetime import datetime

import sklearn.svm as skm

import general.data_io as gio
import navigation_position.auxiliary as npa
import navigation_position.analysis.representations as npra
import navigation_position.visualization as npv


def create_parser():
    parser = argparse.ArgumentParser(description="decoding analysis on navigation data")
    parser.add_argument(
        "-o",
        "--output_folder",
        default=".",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument(
        "--output_template", default="dec_{region}{conds}-{date}_{jobid}"
    )
    parser.add_argument("--winsize", default=500, type=int)
    parser.add_argument("--stepsize", default=50, type=int)
    parser.add_argument("--jobid", default="0000")
    parser.add_argument("--use_inds", default=None, nargs="+", type=int)
    parser.add_argument("--correct_only", default=False, action="store_true")
    parser.add_argument("--include_instructed", default=False, action="store_true")
    parser.add_argument("--regions", default=None, nargs="+")
    parser.add_argument("--decoder", default="linear")
    parser.add_argument("--balance_fields", default=None, nargs="+")
    return parser


decoder_dict = {
    "linear": {},
    "RBF": {"model": skm.SVC},
    "neighbors": {"use_nearest_neighbors": True},
}


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()

    data = gio.Dataset.from_readfunc(
        npa.load_gulli_hashim_data_folder,
        npa.BASEFOLDER,
        load_only_nth_files=args.use_inds,
    )
    data_use = npa.mask_completed_trials(data, correct_only=args.correct_only)
    cond_s = ""
    if args.correct_only:
        cond_s = cond_s + "_correct"
    if not args.include_instructed:
        data_use = npa.mask_uninstructed_trials(data_use)
    else:
        cond_s = cond_s + "_instructed"
    if args.balance_fields is not None:
        args.balance_fields = list(args.balance_fields)
        cond_s = cond_s + "_" + "-".join(args.balance_fields)

    decoder_kwargs = decoder_dict.get(args.decoder, {})
    cond_s = cond_s + "_{}".format(args.decoder)
    out_all = npra.decode_times(
        data_use,
        regions=args.regions,
        balance_fields=args.balance_fields,
        winsize=args.winsize,
        stepsize=args.stepsize,
        **decoder_kwargs,
    )
    f, _ = npv.visualize_decoding_dict(out_all)
    if args.regions is None:
        args.regions = ("all",)

    dates = data_use["date"].to_numpy()

    out_fn = args.output_template.format(
        region="-".join(args.regions),
        date="-".join(dates),
        conds=cond_s,
        jobid=args.jobid,
    )

    out_fig_path = os.path.join(args.output_folder, out_fn + ".pdf")
    f.savefig(out_fig_path, transparent=True, bbox_inches="tight")

    out_arg_path = os.path.join(args.output_folder, out_fn + ".pkl")
    pickle.dump(vars(args), open(out_arg_path, "wb"))
