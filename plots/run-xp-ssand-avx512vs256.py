#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if ssand-avx512vs256.csv -v omp_tiled  -- col=schedule row=label

# easypapOptions = {
#      "-k": ["ssandPile"],
#     "-i": [256],
#     "-v": ["omp_tiled"],
#     "-wt": ["avx"],
#     "-s": [1024],
#     "-ts": [16, 32, 64, 128, 512, 1024],
#     "--label": ["omp_tiled"],
#     "-of": ["run-xp-ssand-avx512vs256.csv"],
# }

# # OMP Internal Control Variable
# ompICV = {
#     "OMP_SCHEDULE": ["static", "static,1"],
#     "OMP_NUM_THREADS": [1] + list(range(4, os.cpu_count() + 1, 4)),
# }

# nbruns = 1
# # Lancement des experiences
# execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

easypapOptions = {
     "-k": ["ssandPile"],
    "-i": [5000],
    "-v": ["omp_tiled"],
    "-wt": ["avx"],
    "-s": [512],
    "-ts": [32, 64],
    "--label": ["omp_tiled"],
    "-of": ["ssand-avx512vs256.csv"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static"],
    "OMP_NUM_THREADS": [4, 8, 16, 24]
}

#del easypapOptions["-ts"]
easypapOptions["--label"] = ["line"]

nbruns = 24

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

easypapOptions["--label"] = ["line"]
easypapOptions["-wt"] = ["avx_256"]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

easypapOptions = {
     "-k": ["ssandPile"],
    "-i": [5000],
    "-v": ["omp_tiled"],
    "-wt": ["opt"],
    "-s": [512],
    "-ts": [32, 64],
    "--label": ["omp_tiled"],
    "-of": ["ssand-avx512vs256.csv"],
}

easypapOptions["--label"] = ["line"]

ompICV = {
    "OMP_SCHEDULE": ["static"],
    "OMP_NUM_THREADS": [1]
}

execute("./run ", ompICV, easypapOptions, nbruns = 1, verbose=False, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if ssand-avx512vs256.csv -v omp_tiled -- col=schedule row=label")