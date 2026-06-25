"""
Constantes et chargement des données (module partagé)
=====================================================
Regroupe au même endroit les coefficients des matières et la lecture du CSV,
afin que les versions séquentielle, parallèle et le benchmark utilisent
exactement les mêmes définitions (principe DRY).
"""
import numpy as np
import pandas as pd

# Coefficients des matières (cf. énoncé du TP)
COEF_MATHS = 5
COEF_PHYSIQUE = 4
COEF_ANGLAIS = 2
TOTAL_COEF = COEF_MATHS + COEF_PHYSIQUE + COEF_ANGLAIS  # = 11

DEFAULT_CSV = "etudiants.csv"


def charger_notes(chemin_csv=DEFAULT_CSV):
    """Lit le CSV et renvoie trois tableaux NumPy (maths, physique, anglais).

    pandas est utilisé pour la lecture car il est beaucoup plus rapide que
    numpy.loadtxt sur des millions de lignes. Les tableaux renvoyés sont en
    float64 et contigus en mémoire, prêts à être consommés par les kernels
    Numba.
    """
    df = pd.read_csv(chemin_csv, usecols=['maths', 'physique', 'anglais'])
    maths = np.ascontiguousarray(df['maths'].to_numpy(dtype=np.float64))
    physique = np.ascontiguousarray(df['physique'].to_numpy(dtype=np.float64))
    anglais = np.ascontiguousarray(df['anglais'].to_numpy(dtype=np.float64))
    return maths, physique, anglais
