import numpy as np


def simulate_stay_switch_strategy(
    data, session_ind=0, pos_keys=["IsEast", "IsNorth"], corr_key="IsEast"
):
    pos = data[pos_keys][session_ind].to_numpy()
    targ = data[corr_key][session_ind].to_numpy()
    predicted_choice = np.zeros(len(pos))
    rng = np.random.default_rng()
    last_choice = rng.choice((0, 1), 1)[0]
    last_targ = None
    last_pos_x, last_pos_y = None, None
    last_corr = None
    for i, pos_i in enumerate(pos):
        expect_corr = last_choice
        if last_corr is not None and not last_corr:
            expect_corr = 1 - last_choice
        if pos_i[0] != last_pos_x or pos_i[1] != last_pos_y:
            expect_corr = 1 - expect_corr

        predicted_choice[i] = expect_corr
        last_choice = expect_corr
        last_targ = targ[i]
        last_corr = last_choice == last_targ
        last_pos_x, last_pos_y = pos_i
    return predicted_choice


def simulate_see_boundary_strategy(
    data,
    session_ind=0,
    pos_keys=["IsEast", "IsNorth"],
    corr_key="IsEast",
    rot_key="choice_rotation",
):
    pos = data[pos_keys][session_ind]
    targ = data[corr_key][session_ind]
    rot = data[rot_key][session_ind]

    
    
