"""
Tâche 1 — Génération des données
=================================
Génère un fichier CSV d'au moins 1 000 000 d'étudiants. Chaque étudiant a
trois notes (Maths, Physique, Anglais) tirées aléatoirement dans [0, 20],
arrondies au demi-point pour rester réalistes.

Usage :
    python generate_data.py [N] [chemin_csv]
"""
import sys
import time

import numpy as np
import pandas as pd

from data_io import DEFAULT_CSV

# Valeur par défaut : bien au-dessus du minimum requis (1 000 000), choisie
# pour que la boucle de calcul dure assez longtemps et que le gain de la
# parallélisation soit mesurable sans bruit.
N_DEFAUT = 5_000_000


def generer_csv(n_etudiants, chemin_csv=DEFAULT_CSV, graine=42):
    """Génère n_etudiants lignes et les écrit dans chemin_csv.

    La graine fixe le tirage aléatoire pour que les résultats soient
    reproductibles (le correcteur obtient les mêmes données).
    """
    rng = np.random.default_rng(graine)
    # Notes sur 20, arrondies à 0.5
    maths = np.round(rng.uniform(0, 20, n_etudiants) * 2) / 2
    physique = np.round(rng.uniform(0, 20, n_etudiants) * 2) / 2
    anglais = np.round(rng.uniform(0, 20, n_etudiants) * 2) / 2

    df = pd.DataFrame({
        'id': np.arange(1, n_etudiants + 1, dtype=np.int64),
        'maths': maths,
        'physique': physique,
        'anglais': anglais,
    })
    df.to_csv(chemin_csv, index=False)
    return df


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAUT
    chemin = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CSV

    debut = time.perf_counter()
    generer_csv(n, chemin)
    duree = time.perf_counter() - debut
    print(f"{n:,} étudiants générés dans '{chemin}' en {duree:.2f} s")
