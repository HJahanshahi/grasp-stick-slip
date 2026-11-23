# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""make_all.py - regenerates every data figure and table of the paper into figures/ (about 1 min).

    python make_all.py
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
def run(folder, script):
    print(">>", folder + "/" + script, flush=True); subprocess.run([sys.executable, script], check=True, cwd=os.path.join(HERE, folder))
run("one_axis", "make_figures.py")            # Figs. 3 to 6, figs. S1, S2, S5, Table 1, tables S1 to S3
run("free_body", "fig1.py")                   # Fig. 1
run("free_body", "fig7_figS3_tableS4.py")     # Fig. 7, fig. S3, table S4
run("free_body", "figS4_grasp_loss.py")       # fig. S4
print("done: see figures/")
