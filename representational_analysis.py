
import numpy as np
import sklearn.neighbors as sknn

import general.neural_analysis as na 

import navigation_position.auxiliary as npa


def equals_one(x):
    return x == 1


def equal_0(x):
    return x == 0

def less_than_180(x):
    return x < 180


default_funcs = {
    "choice orientation": less_than_180,
    "rewarded": equal_0,
}


reduced_time_dict = {
    "pre_rotation_end": (-1000, 0),
    "nav_end": (-1000, 0),
    "relevant_crossing_x":, (-500, 500),
    "relevant_crossing_y":, (-500, 500),
    "post_rotation_end": (-1000, 0),
    "choice_approach_end": (-1000, 0),
    "choice_start": (-500, 500),
    "approach_end": (-1000, 0),
}


default_dec_variables = {
    "relevant position": "IsEast",
    "irrelevant position": "IsNorth",
    "white side": "white_right",
    "correct side": "target_right",
    "choice side": "chose_right",
    "choice color": "chose_white",
    "choice orientation": "pre_choice_rotation",
    "rewarded": "TrialError",
}


def make_variable_masks(data, dec_variables=default_dec_variables, func_dict=None):
    """
    The variables of interest are:
    position (binary and continuous), correct side, view direction, choice
    """
    if func_dict is None:
        func_dict = default_funcs
    masks = {}
    for k, v in dec_variables.items():
        func = func_dict.get(k, equals_one)
        m1 = func(data[v])
        m2 = func(data[v]).rs_not()
        masks[k] = (m1, m2)
    return masks


default_contrast_variables = {
    "relevant_position": ("IsNorth",),
    "irrelevant_position": ("IsEast",),
}


def make_variable_generalization_masks(
        data, contrast_variables=default_contrast_variables, func_dict=None,
):
    masks = make_variable_masks(data)
    masks_gen = {}
    for k, (m1, m2) in masks.items():
        for cv in contrast_variables[k]:
            func = func_dict.get(cv, equals_one)
            m1_p = m1.rs_and(func(data[cv]))
            m2_p = m2.rs_and(func(data[cv]))
            m1_n = m1.rs_and(~func(data[cv]))
            m2_n = m2.rs_and(~func(data[cv]))
            fcv = "{} x {}".format(k, cv)
            masks_gen[fcv] = ((m1_p, m2_p), (m1_n, m2_n))
    return masks_gen
    

def decode_masks(
        data,
        mask1,
        mask2,
        tbeg,
        tend,
        tzf,
        winsize=500,
        stepsize=50,
        gen_mask1=None,
        gen_mask2=None,
        use_nearest_neighbors=False,
        n_neighbors=5,
        **kwargs,
):
    if use_nearest_neighbors:
        kwargs["model"] = sknn.KNeighborsClassifier
        kwargs["params"] = {"n_neighbors": n_neighbors}
    out = data.decode_masks(
        mask1,
        mask2,
        winsize,
        tbeg,
        tend,
        stepsize,
        decode_m1=gen_mask1,
        decode_m2=gen_mask2,
        time_zero_field=tzf,
        **kwargs,
    )
    return out


def decode_times(data, time_dict=None, dec_vars=None, **kwargs):
    if time_dict is None:
        time_dict = reduced_time_dict
    if dec_vars is None:
        dec_vars = default_dec_variables
    out_dict = {}

    masks = make_variable_masks(data)
    for time_k, ts in time_dict.items():
        out_dict[time_k] = {}
        for var_k, k_masks in masks.items():
            out = decode_masks(data, *k_masks, *ts, time_k, **kwargs, ret_pops=True)
            out_dict[time_k][var_k] = out
    return out_dict
    
