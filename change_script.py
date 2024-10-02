import argparse
import pickle
import os
from datetime import datetime

import sklearn.svm as skm

import general.data_io as gio
import navigation_position.auxiliary as npa
import navigation_position.analysis.representations as npra
import navigation_position.analysis.change as npac


def create_parser():
    parser = argparse.ArgumentParser(
        description="change of mind analysis on navigation data",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default=".",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument(
        "--output_template",
        default="change_{conds}-w{winsize}-s{stepsize}_{date}_{jobid}",
    )
    parser.add_argument("--time_zero_field", default="choice_start")
    parser.add_argument("--winsize", default=500, type=int)
    parser.add_argument("--stepsize", default=50, type=int)
    parser.add_argument("--change_eps", default=.1, type=float)    
    parser.add_argument("--jobid", default="0000")
    parser.add_argument("--use_inds", default=None, nargs="+", type=int)
    parser.add_argument("--correct_only", default=False, action="store_true")
    parser.add_argument("--include_instructed", default=False, action="store_true")
    parser.add_argument("--decoder", default="linear")
    parser.add_argument("--balance", default=False, action="store_true")
    parser.add_argument("--projection", default=False, action="store_true")
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
    if args.balance:
        cond_s = cond_s + "_balanced"

    decoder_kwargs = decoder_dict.get(args.decoder, {})
    cond_s = cond_s + "_{}".format(args.decoder)

    ts = npra.reduced_time_dict[args.time_zero_field]
    out = npac.decode_change_of_mind_regions(
        data_use,
        *ts,
        args.time_zero_field,
        eps=args.change_eps,
        winsize=args.winsize,
        stepsize=args.stepsize,
        ret_projections=args.projection,
        **decoder_kwargs,
    )

    f, _ = npac.visualize_change_of_mind_dec(out, tzf=args.time_zero_field)

    dates = data_use["date"].to_numpy()

    out_fn = args.output_template.format(
        date="-".join(dates),
        conds=cond_s,
        jobid=args.jobid,
        winsize=args.winsize,
        stepsize=args.stepsize,
    )

    out_fig_path = os.path.join(args.output_folder, out_fn + ".pdf")
    f.savefig(out_fig_path, transparent=True, bbox_inches="tight")

    save_dict = vars(args)
    save_dict["decoding"] = out
    out_arg_path = os.path.join(args.output_folder, out_fn + ".pkl")
    pickle.dump(save_dict, open(out_arg_path, "wb"))
