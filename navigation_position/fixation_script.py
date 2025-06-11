import argparse
import pickle
import os
from datetime import datetime

import sklearn.svm as skm

import general.data_io as gio
import navigation_position.auxiliary as npa
import navigation_position.analysis.representations as npra
import navigation_position.visualization as npv
import navigation_position.figures as npf


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
        "--output_template",
        default="fix_{region}_{date}_{jobid}",
    )
    parser.add_argument("--winsize", default=500, type=int)
    parser.add_argument("--stepsize", default=50, type=int)
    parser.add_argument("--jobid", default="0000")
    parser.add_argument("--use_ind", default=None, type=int)
    parser.add_argument("--correct_only", default=False, action="store_true")
    parser.add_argument("--include_instructed", default=False, action="store_true")
    parser.add_argument("--instructed_only", default=False, action="store_true")
    parser.add_argument("--regions", default=None, nargs="+")
    parser.add_argument("--decoder", default="linear")
    parser.add_argument("--balance_fields", default=None, nargs="+")
    parser.add_argument("--shift_trials", default=0, type=int)
    parser.add_argument("--balance_training", default=False, action="store_true")
    return parser


decoder_dict = {
    "linear": {},
    "RBF": {"model": skm.SVC},
    "neighbors": {"use_nearest_neighbors": True},
}


def main():
    parser = create_parser()
    args = parser.parse_args()

    use_date = npa.get_date_list()[args.use_ind]
    args.date = datetime.now()

    fig = npf.FixationAnalysis(
        use_date, regions=args.regions,
    )
    fig.panel_fixations()
    fig.panel_dec()

    fn = args.output_template.format(
        region="-".join(args.regions),
        date=use_date,
        jobid=args.jobid,
    )
    fig.save(fn, use_bf=args.output_folder)
