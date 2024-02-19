#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if ssand-omp_tiled_256.csv -v omp_tiled  -- col=schedule row=label

easypapOptions = {
     "-k": ["ssandPile"],
    "-i": [100],
    "-v": ["omp_tiled"],
    "-wt": ["opt"],
    "-s": [256],
    "-ts": [8, 16, 32, 64, 128],
    "--label": ["square"],
    "-of": ["ssand-omp_tiled_256.csv"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static", "static,1"],
    "OMP_NUM_THREADS": [1] + list(range(4, os.cpu_count() + 1, 4)),
}

nbruns = 1
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

del easypapOptions["-ts"]
easypapOptions["--label"] = ["line"]
easypapOptions["-th"] = [1]
easypapOptions["-tw"] = [8, 16, 32, 64, 128, 256]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

easypapOptions = {
    "-k": ["ssandPile"],
    "-i": [100],
    "-v": ["seq"],
    "-s": [256],
    "-of": ["ssand-omp_tiled_256.csv"],
}
ompICV = {"OMP_NUM_THREADS": [1]}
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if ssand-omp_tiled_256.csv -v omp_tiled -- col=schedule row=label")

#OMP_NUM_THREADS=3 OMP_SCHEDULE=static ./run -k ssandPile -v omp_tiled -wt opt -tw 1024 -th 1 -s 4096 -i 64 -n -a 4partout