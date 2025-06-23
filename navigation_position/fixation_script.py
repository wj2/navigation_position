import argparse
from datetime import datetime

import sklearn.svm as skm

import navigation_position.auxiliary as npa
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
        default="fix_{region}_{trial_string}{balanced}_{date}_{jobid}",
    )
    parser.add_argument("--jobid", default="0000")
    parser.add_argument("--use_ind", default=None, type=int)
    parser.add_argument("--correct_only", default=False, action="store_true")
    parser.add_argument("--include_instructed", default=False, action="store_true")
    parser.add_argument("--instructed_only", default=False, action="store_true")
    parser.add_argument("--balance_correct", default=False, action="store_true")
    parser.add_argument("--regions", default=None, nargs="+")

    return parser


decoder_dict = {
    "linear": {},
    "RBF": {"model": skm.SVC},
    "neighbors": {"use_nearest_neighbors": True},
}


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.use_ind is not None:
        use_date = npa.get_date_list()[args.use_ind]
    else:
        use_date = None
    args.date = datetime.now()

    trial_string = "uninstructed"
    if args.correct_only:
        trial_string = "correct"
    elif args.include_instructed:
        trial_string = "all"
    elif args.instructed_only:
        trial_string = "instructed"

    fig = npf.FixationAnalysis(
        date=use_date,
        regions=args.regions,
        trial_string=trial_string,
        balance_correct=args.balance_correct,
    )
    fig.panel_fixations()
    fig.panel_dec()
    fig.panel_cross_fixation()
    fig.panel_eye_decoding()

    if args.regions is None:
        regions = ("all",)
    else:
        regions = args.regions

    
    balance_s = ""
    if args.balance_correct:
        balance_s = "-balanced"
    fn = args.output_template.format(
        region="-".join(regions),
        date=use_date,
        trial_string=trial_string,
        balanced=balance_s,
        jobid=args.jobid,
    )
    fig.save(fn + ".pdf", use_bf=args.output_folder)
