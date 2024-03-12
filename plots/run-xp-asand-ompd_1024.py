#!/usr/bin/env python3
from expTools import *
# expériences avec des tuiles carrées
easypapOptions = {
"-k": ["asandPile"],
"-i": [10],
"-v": ["ompd"],
"-wt": ["ooo"],
"-s": [1024],
"-ts": [8, 16, 32, 64, 128, 256, 512],
"--label": ["square"],
"-of": ["asand-ompd_1024.csv"]
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
easypapOptions["-tw"] = [8, 16, 32, 64, 128, 256, 512]
execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

print("Recommended plot:")
print(" plots/easyplot.py -if asand-ompd_1024.csv -v ompd -- col=schedule row=label")
