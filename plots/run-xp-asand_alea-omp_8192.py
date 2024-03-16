#!/usr/bin/env python3
from expTools import *
# expériences avec des tuiles carrées
easypapOptions = {
"-k": ["asandPile"],
"-i": [16],
"-v": ["omp"],
"-wt": ["opt1"],
"-a":["alea"],
"-s": [8192],
"-ts": [512, 1024, 2048],
"--label": ["square"],
"-of": ["ssand_alea-omp_8192.csv"]
}
# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static"],
    "OMP_NUM_THREADS": [1] + list(range(4, os.cpu_count() + 1, 4)),
}
nbrun=1
# Lancement des experiences
execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")
# expériences avec des lignes
del easypapOptions["-ts"]
easypapOptions["--label"] = ["line"]
easypapOptions["-th"] = [4, 8, 16, 32, 64, 128, 256]
easypapOptions["-tw"] = [256, 512, 1024, 2048, 4096]
execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

easypapOptions = {
    "-k": ["asandPile"],
    "-i": [16],
    "-v": ["seq"],
    "-s": [2048],
    "-of": ["ssand_alea-omp_8192.csv"],
}
ompICV = {"OMP_NUM_THREADS": [1]}
execute("./run ", ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

print("Recommended plot:")
print("plots/easyplot.py -if ssand_alea-omp_8192.csv -v omp -- col=schedule row=label")
