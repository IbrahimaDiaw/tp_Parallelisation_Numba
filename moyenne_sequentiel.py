"""
Tâche 2 — Version séquentielle (mono-thread)
============================================
Calcule la moyenne pondérée de chaque étudiant avec Numba, en mode
mono-thread. Cette version sert de référence
pour mesurer le speedup.

    Moyenne = (M*5 + P*4 + A*2) / 11
"""
import sys
import time

import numpy as np
from numba import njit

from data_io import (COEF_MATHS, COEF_PHYSIQUE, COEF_ANGLAIS, TOTAL_COEF,
                     charger_notes, DEFAULT_CSV)


@njit(cache=True, fastmath=True)
def moyenne_sequentielle(maths, physique, anglais):
    """Moyenne pondérée de chaque étudiant, calculée en série.

    La boucle est compilée en code machine par Numba mais s'exécute sur un
    seul thread : les itérations sont traitées les unes après les autres.
    Chaque note étant indépendante des autres, il n'y a aucune dépendance de
    données entre les itérations.
    """
    n = maths.shape[0]
    resultat = np.empty(n, dtype=np.float64)
    for i in range(n):
        resultat[i] = (maths[i] * COEF_MATHS
                       + physique[i] * COEF_PHYSIQUE
                       + anglais[i] * COEF_ANGLAIS) / TOTAL_COEF
    return resultat


if __name__ == '__main__':
    chemin = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    maths, physique, anglais = charger_notes(chemin)

    # 1er appel : compilation JIT (non chronométré)
    moyenne_sequentielle(maths[:10], physique[:10], anglais[:10])

    debut = time.perf_counter()
    moyennes = moyenne_sequentielle(maths, physique, anglais)
    duree = time.perf_counter() - debut

    print(f"Version séquentielle  : {maths.shape[0]:,} moyennes en "
          f"{duree*1000:.2f} ms")
    print(f"  Exemple (5 premières) : {np.round(moyennes[:5], 2)}")
    print(f"  Moyenne générale      : {moyennes.mean():.4f}")
