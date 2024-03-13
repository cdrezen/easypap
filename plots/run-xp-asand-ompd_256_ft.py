#!/usr/bin/env python3
from expTools import *
# expériences avec des tuiles carrées
easypapOptions = {
"-k": ["asandPile"],
"-i": [1024],
"-v": ["ompd"],
"-wt": ["ooo"],
"-s": [256],
"-ts": [16, 32, 64],
"-ft":[""],
"--label": ["square"],
"-of": ["asand-ompd_256.csv"]
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
easypapOptions["-th"] = [1]
easypapOptions["-tw"] = [16, 32, 64]
execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

easypapOptions = {
    "-k": ["asandPile"],
    "-i": [128],
    "-v": ["seq"],
    "-s": [256],
    "-of": ["asand-ompd_256.csv"],
}
ompICV = {"OMP_NUM_THREADS": [1]}
execute("./run ", ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

print("Recommended plot:")
print(" plots/easyplot.py -if asand-ompd_256.csv -v ompd -- col=schedule row=label")


#OMP_NUM_THREADS=32 OMP_SCHEDULE=static ./run -k asandPile -s 256 -ts 16 -v ompd -wt ooo -n -ft