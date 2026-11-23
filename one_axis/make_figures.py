# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""make_figures.py - regenerates the one-axis figures and tables of the paper and copies the figures, with the
numbers used in the paper, to ../figures.

Run from this folder:
    python make_figures.py            uses the fitted law in out/law_params.json (about 1 min)
    python make_figures.py --refit    refits the law (bootstrap included) and the grasp-force model first
    python make_figures.py --rebuild  also rebuilds out/metrics.csv and out/signals.npz from the packed raw data with
                                      pipeline.py (needs data/pack.npz, data2/pack_s2.npz and data2/pack_s2bad.npz,
                                      which are in the data record, not in this repository)
Needs Python 3.10+ with numpy, scipy and matplotlib (see ../requirements.txt). The Inter font is used if installed."""
import os, sys, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
run = lambda script: subprocess.run([sys.executable, script], check=True)
os.makedirs(os.path.join(HERE, "out", "figs"), exist_ok=True)
if "--rebuild" in sys.argv:
    run("pipeline.py")         # per-trial metrics and aligned signals          -> out/metrics.csv, out/signals.npz
if "--refit" in sys.argv or "--rebuild" in sys.argv:
    run("fit_law.py")          # law fit, bootstrap, Table 1, validation set     -> out/law_params.json, ...
    run("force_model.py")      # grasp-force model used by the law replay        -> out/force_model.json
run("fig3_law_schematic.py")   # fig. S5 law elements
import figs
figs.fig2()                    # fig. S1 one grasp
figs.fig3()                    # fig. S2 force build-up during closure
figs.table2()                  # table S2
figs.fig4_angle()              # Fig. 3 torque against rim angle
figs.fig5_loops()              # Fig. 4 hysteresis loops
figs.fig6_validation()         # Fig. 5 law validation
run("closed_loop.py")          # Fig. 6 closed loop with law replay, table S3
FINAL = os.path.join(HERE, "..", "figures"); os.makedirs(FINAL, exist_ok=True)
F = os.path.join(HERE, "out", "figs")
MAP = {"fig3_law_schematic": "FigS5", "fig2_time_histories": "FigS1", "fig3_closure_map": "FigS2", "fig4_torque_angle": "Fig3",
       "fig5_hysteresis_angle": "Fig4", "fig6_law_validation": "Fig5", "fig7_closed_loop": "Fig6"}
for src, dst in MAP.items():
    for ext in (".pdf", ".png"):
        if os.path.exists(os.path.join(F, src + ext)): shutil.copy(os.path.join(F, src + ext), os.path.join(FINAL, dst + ext))
for src, dst in (("table1_design.md", "TableS1.md"), ("table2_closure.md", "TableS2.md"), ("table3_law.md", "Table1.md"), ("table4_closed_loop.md", "TableS3.md")):
    if os.path.exists(os.path.join(HERE, "out", src)): shutil.copy(os.path.join(HERE, "out", src), os.path.join(FINAL, dst))
print("done: figures in", os.path.abspath(FINAL))
