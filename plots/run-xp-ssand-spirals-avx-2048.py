#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if ssand-spirals-avx-2048.csv -v lazy1 -- col=schedule row=label

easypapOptions = {
     "-k": ["ssandPile"],
    "-i": [1000],
    "-v": ["lazy1"],
    "-wt": ["avx_spirals"],
    "-a": ["spirals"],
    "-s": [2048],
    "-ts": [16],
    "--label": ["omp_tiled"],
    "-of": ["ssand-spirals-avx-2048.csv"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static", "static,1"],
    "OMP_NUM_THREADS": [1] + list(range(4, os.cpu_count() + 1, 4)),
}

nbruns = 2
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# del easypapOptions["-ts"]
# easypapOptions["--label"] = ["line"]
# easypapOptions["-th"] = [1, 4, 8, 16, 32]
# easypapOptions["-tw"] = [16, 32, 64, 128, 512, 1024]

# execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

print("Recommended plot:")
print(" plots/easyplot.py -if ssand-spirals-avx-2048.csv -v lazy1 -- col=schedule row=label")