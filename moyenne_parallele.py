"""
Tâche 3 — Version parallèle (multi-thread)
==========================================
Même calcul que la version séquentielle, mais la boucle est parallélisée
automatiquement par Numba grâce à @njit(parallel=True) et prange. Les
itérations, indépendantes les unes des autres, sont réparties sur tous les
cœurs disponibles.

    Moyenne = (M*5 + P*4 + A*2) / 11
"""
import sys
import time

import numpy as np
from numba import njit, prange

from data_io import (COEF_MATHS, COEF_PHYSIQUE, COEF_ANGLAIS, TOTAL_COEF,
                     charger_notes, DEFAULT_CSV)

@njit(parallel=True, cache=True, fastmath=True)
def moyenne_parallele(maths, physique, anglais):
    """Moyenne pondérée de chaque étudiant, calculée en parallèle.

    prange (au lieu de range) indique à Numba que les itérations sont
    indépendantes : il peut donc les distribuer sur plusieurs threads qui
    s'exécutent simultanément. C'est exactement le cas ici puisque la moyenne
    d'un étudiant ne dépend d'aucun autre.
    """
    n = maths.shape[0]
    resultat = np.empty(n, dtype=np.float64)
    for i in prange(n):
        resultat[i] = (maths[i] * COEF_MATHS
                       + physique[i] * COEF_PHYSIQUE
                       + anglais[i] * COEF_ANGLAIS) / TOTAL_COEF
    return resultat


if __name__ == '__main__':
    import numba

    chemin = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    maths, physique, anglais = charger_notes(chemin)

    # 1er appel : compilation JIT (non chronométré)
    moyenne_parallele(maths[:10], physique[:10], anglais[:10])

    debut = time.perf_counter()
    moyennes = moyenne_parallele(maths, physique, anglais)
    duree = time.perf_counter() - debut

    print(f"Version parallèle ({numba.get_num_threads()} threads) : "
          f"{maths.shape[0]:,} moyennes en {duree*1000:.2f} ms")
    print(f"  Exemple (5 premières) : {np.round(moyennes[:5], 2)}")
    print(f"  Moyenne générale      : {moyennes.mean():.4f}")
