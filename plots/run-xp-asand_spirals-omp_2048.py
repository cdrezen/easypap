#!/usr/bin/env python3
from expTools import *
# expériences avec des tuiles carrées
easypapOptions = {
"-k": ["asandPile"],
"-i": [256],
"-v": ["omp"],
"-wt": ["opt1"],
"-a":["spirals"],
"-s": [2048],
"-ts": [64, 256, 512],
"--label": ["square"],
"-of": ["asand-omp_2048.csv"]
}
# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static", "static,1"],
    "OMP_NUM_THREADS": [1] + list(range(4, os.cpu_count() + 1, 4)),
}
nbrun=1
# Lancement des experiences
execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")
# expériences avec des lignes
del easypapOptions["-ts"]
easypapOptions["--label"] = ["line"]
easypapOptions["-th"] = [16, 32, 64, 128]
easypapOptions["-tw"] = [128, 256, 512, 1024, 2048]
execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

easypapOptions = {
    "-k": ["asandPile"],
    "-i": [256],
    "-v": ["seq"],
    "-s": [2048],
    "-of": ["asand-omp_2048.csv"],
}
ompICV = {"OMP_NUM_THREADS": [1]}
execute("./run ", ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

print("Recommended plot:")
print("plots/easyplot.py -if asand-omp_2048.csv -v omp -- col=schedule row=label")

