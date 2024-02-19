#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if ssand-omptask.csv -v omp_taskloop  -- col=schedule row=label

easypapOptions = {
     "-k": ["ssandPile"],
    "-i": [256],
    "-v": ["omp_taskloop"],
    "-wt": ["opt"],
    "-s": [1024],
    "-ts": [16, 32, 64, 128, 512],
    "--label": ["square"],
    "-of": ["ssand-omptask.csv"],
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
easypapOptions["-tw"] = [16, 32, 64, 128, 512, 1024]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

easypapOptions = {
    "-k": ["ssandPile"],
    "-i": [256],
    "-v": ["seq"],
    "-s": [1024],
    "-of": ["ssand-omptask.csv"],
}
ompICV = {"OMP_NUM_THREADS": [1]}
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if ssand-omptask.csv -v omp_taskloop -- col=schedule row=label")

#OMP_NUM_THREADS=3 OMP_SCHEDULE=static ./run -k ssandPile -v omp_tiled -wt opt -tw 1024 -th 1 -s 4096 -i 64 -n -a 4partout