#!/usr/bin/env python3
from expTools import *

easypapOptions = {
     "-k": ["ssandPile"],
     "-g":[""],
    "-i": [5000],
    "-v": ["ocl_omp"],
    "-wt": ["avx"],
    "-s": [2048],
    "-th": [1],
    "-tw": [512],
    "-a": [0.75, 0.5, 0.25, 0.125], 
    "--label": ["ocl_omp"],
    "-of": ["ssand-ompcl.csv"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static"],
    "OMP_NUM_THREADS": [16, 32]
}

nbruns = 4

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

#del easypapOptions["-ts"]
easypapOptions["--label"] = ["fullgpu"]
easypapOptions["-v"] = ["ocl"]
del easypapOptions["-a"]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

easypapOptions["--label"] = ["fullcpu"]
easypapOptions["-v"] = ["omp_tiled"]
del easypapOptions["-g"]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")


easypapOptions["--label"] = ["seq"]
del easypapOptions["-v"]

ompICV = {
    "OMP_NUM_THREADS": [1]
}

nbruns = 1

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if ssand-ompcl.csv -v omp_tiled -- col=schedule row=label")