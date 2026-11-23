# Asymmetric stick-slip in a robotic grasp on a spinning body

Code and data for the paper

> H. Jahanshahi and Z. H. Zhu, "A stiff robotic grasp stores and returns the spin of a captured body through
> asymmetric stick-slip" (submitted).

A three-finger hand on one robot arm grasps the rim of the engine cone of a spacecraft mock-up that a second
arm spins. The repository holds the fitted grasp-torque law, the processed measurements, and the scripts that
produce every data figure and table of the paper.

## Reproduce the figures and tables

Python 3.10 or later with numpy, scipy and matplotlib.

    pip install -r requirements.txt
    python make_all.py

This takes about one minute and writes the figures and tables to `figures/` with the numbers used in the
paper. The Inter font is used if it is installed.

| Output | Script |
|---|---|
| Fig. 1 | `free_body/fig1.py` |
| Figs. 3 to 6, figs. S1, S2 and S5, Table 1, tables S1 to S3 | `one_axis/make_figures.py` |
| Fig. 7, fig. S3, table S4 | `free_body/fig7_figS3_tableS4.py` |
| Fig. S4 and the energy loss of the grasp | `free_body/figS4_grasp_loss.py` |
| Data S1 | `supplementary/make_data_S1.py` |
| Movie S1 | `supplementary/movie_S1.py` (needs the camera recording) |

Fig. 2 is a set of photographs.

## Contents

    one_axis/        target arm turns the rim about one fixed axis (sessions 1 to 3)
        fit_law.py           fits the law on the sweeps, with bootstrap, and builds the validation set
        force_model.py       grasp-force model used when the law is replayed
        figs.py              figures of the measurements and of the law
        closed_loop.py       closed-loop captures with a virtual inertia, and their replay with the law
        fig7_replay.py       replay of one capture with the law
        sweep_reversal.py    stiffness before and after each reversal of the rim (needs the packed raw data)
        pipeline.py          packed raw data -> out/metrics.csv and out/signals.npz
        out/                 processed data, law parameters (law_params.json), grasp-force model, tables
        data3/               trial records of the closed-loop captures
    free_body/       captures of a body that is free in all six axes
        law_free_body.py     replays each capture with the law, with no parameter refitted
        data/free_body_trials.npz      68 centered captures and 8 long holds (emulator and controller logs)
        data/block_P_pack.npz          closed-loop captures of session 3 (one-axis emulation)
        data/movie_trials.npz          the three trials recorded for Movie S1
    acquisition/     scripts that ran the one-axis trials on the two arms, trial plans, offline mocks
    packing/         raw trial folders -> packed .npz files
    supplementary/   Data S1 and the script that builds it, script of Movie S1
    figures/         written by make_all.py

## The law

The torque of the grasp about the cone axis is the sum of three elasto-slip elements. With rim rate `w`,
grasp force `F` and `s = F / 50 N`, each element is updated as

    e_j <- clip(e_j - s * k_j * w * dt, -s * c_j_plus, s * c_j_minus)        torque = e_1 + e_2 + e_3

The stiffnesses `k_j` and the limits `c_j_plus` and `c_j_minus` are in `one_axis/out/law_params.json`.

## Data

`free_body/data/free_body_trials.npz` stores, for each trial number, three arrays. `em` is the log of the
free-body emulator at 125 Hz (columns in `em_cols`: time, contact force and torque, gate, pose, velocity,
energy terms). `ctrl` is the log of the chaser arm and hand (columns in `ctrl_cols`). `events` lists the
events of the trial with their times. `free_body_trials_index.csv` gives the conditions of each trial.

The packed raw recordings of the one-axis sessions are larger than this repository allows. They are in the
data record given in the paper.

## License

Code is under the MIT license (`LICENSE`). Data files are under CC BY 4.0.

## Contact

Hadi Jahanshahi, York University, hadij@yorku.ca, https://github.com/HJahanshahi
Zheng H. Zhu (corresponding author), York University, gzhu@yorku.ca
