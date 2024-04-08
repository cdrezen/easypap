#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if ssand-avx-omp_1024.csv -v omp_tiled  -- col=schedule row=label

# easypapOptions = {
#      "-k": ["ssandPile"],
#     "-i": [256],
#     "-v": ["omp_tiled"],
#     "-wt": ["avx"],
#     "-s": [1024],
#     "-ts": [16, 32, 64, 128, 512, 1024],
#     "--label": ["omp_tiled"],
#     "-of": ["ssand-avx-omp_1024.csv"],
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
    "-i": [500],
    "-v": ["omp_tiled"],
    "-wt": ["avx"],
    "-s": [1024],
    "--label": ["omp_tiled"],
    "-of": ["ssand-avx-omp_1024.csv"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static", "static,1"],
    "OMP_NUM_THREADS": [1] + list(range(4, os.cpu_count() + 1, 4)),
}

#del easypapOptions["-ts"]
easypapOptions["--label"] = ["carre"]
easypapOptions["-ts"] = [16, 32]

nbruns=4

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

#rect

del easypapOptions["-ts"]

easypapOptions["--label"] = ["rect"]
easypapOptions["-th"] = [1]
easypapOptions["-tw"] = [512]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-th"] = [16]
easypapOptions["-tw"] = [32]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

easypapOptions["--label"] = ["rect1"]

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-th"] = [8]
easypapOptions["-tw"] = [32]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-th"] = [32]
easypapOptions["-tw"] = [16]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions = {
    "-k": ["ssandPile"],
    "-i": [500],
    "-v": ["omp_tiled"],
    "-wt": ["opt"],
    "-s": [1024],
    "-of": ["ssand-avx-omp_1024.csv"],
}
execute("./run ", ompICV, easypapOptions, nbruns=1, verbose=False, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if ssand-avx-omp_1024.csv -v omp_tiled -- col=schedule row=label")

#OMP_NUM_THREADS=3 OMP_SCHEDULE=static ./run -k ssandPile -v omp_tiled -wt opt -tw 1024 -th 1 -s 4096 -i 64 -n -a 4partout
