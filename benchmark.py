"""
Tâches 4 & 5 — Mesure du speedup et loi d'Amdahl
================================================
Charge les données, exécute les versions séquentielle et parallèle, mesure
le speedup, puis en déduit la fraction parallélisable via la loi d'Amdahl.

Méthodologie : la version séquentielle et la version parallèle sont
strictement identiques (toutes deux compilées par Numba), à une seule
différence — la boucle (`range` mono-thread contre `prange` multi-thread).
Le rapport des temps mesure donc UNIQUEMENT l'effet de la parallélisation
(passage de 1 à N threads), ce qui est exactement ce qu'exige la loi
d'Amdahl.

Protocole de mesure honnête :
  - Warm-up des kernels Numba AVANT chronométrage (on exclut la compilation
    JIT, qui n'a lieu qu'à la 1re exécution).
  - Plusieurs répétitions, on garde le MEILLEUR temps (le moins pollué par
    les autres processus du système).

Usage :
    python benchmark.py [chemin_csv]
"""
import sys
import time

import numpy as np
import numba

from data_io import charger_notes, DEFAULT_CSV
from moyenne_sequentiel import moyenne_sequentielle
from moyenne_parallele import moyenne_parallele

REPETITIONS = 9       # nombre de lots mesurés
CIBLE_LOT = 0.2       # durée visée pour un lot (s), pour des mesures stables


def chronometrer(fonction, args, repetitions=REPETITIONS, cible=CIBLE_LOT):
    """Renvoie le meilleur temps PAR APPEL (en secondes), mesuré par lots.

    Le kernel est très rapide (quelques ms) : le mesurer en un seul appel est
    trop bruité (latence de réveil des threads, scheduling hybride P/E). On
    chronomètre donc un LOT de `boucles` appels, calibré pour durer ~`cible`
    secondes, puis on divise — méthode standard (cf. timeit). On garde le
    meilleur lot sur `repetitions` mesures.
    """
    # Calibration : estimer le coût d'un appel pour dimensionner le lot
    debut = time.perf_counter()
    fonction(*args)
    cout_unitaire = max(time.perf_counter() - debut, 1e-6)
    boucles = max(1, int(cible / cout_unitaire))

    temps = []
    for _ in range(repetitions):
        debut = time.perf_counter()
        for _ in range(boucles):
            fonction(*args)
        temps.append((time.perf_counter() - debut) / boucles)
    return min(temps)


def fraction_parallelisable(speedup, p):
    """Déduit la fraction parallélisable P à partir de la loi d'Amdahl.

    Loi d'Amdahl :  A = 1 / ((1 - P) + P/N)
    En isolant P :  P = (1 - 1/S) / (1 - 1/N)

    où S est le speedup mesuré et N le nombre de threads utilisés.
    """
    return (1 - 1 / speedup) / (1 - 1 / p)


def main():
    chemin = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

    # --- Chargement des données ---
    print("Chargement des données...")
    t0 = time.perf_counter()
    maths, physique, anglais = charger_notes(chemin)
    t_load = time.perf_counter() - t0
    n = maths.shape[0]
    p = numba.get_num_threads()
    print(f"  {n:,} étudiants chargés en {t_load:.2f} s")
    print(f"  Threads Numba disponibles : {p}\n")

    # --- Warm-up : déclenche la compilation JIT (hors chronométrage) ---
    print("Warm-up (compilation JIT des deux kernels)...")
    r_seq = moyenne_sequentielle(maths, physique, anglais)
    r_par = moyenne_parallele(maths, physique, anglais)

    # --- Vérification : les deux versions donnent-elles le même résultat ? ---
    identiques = np.allclose(r_seq, r_par)
    print(f"  Résultats séquentiel == parallèle : {identiques}")
    print(f"  Exemple (5 premières moyennes)    : {np.round(r_seq[:5], 2)}\n")
    if not identiques:
        print("  ATTENTION : les deux versions divergent, mesure non fiable.")

    # --- Mesures chronométrées ---
    print(f"Mesure des temps (meilleur sur {REPETITIONS} répétitions)...")
    t_seq = chronometrer(moyenne_sequentielle, (maths, physique, anglais))
    t_par = chronometrer(moyenne_parallele, (maths, physique, anglais))

    # --- Calculs ---
    speedup = t_seq / t_par
    efficacite = speedup / p
    P = fraction_parallelisable(speedup, p)
    speedup_max = 1 / (1 - P) if P < 1 else float('inf')

    # --- Affichage du tableau de résultats ---
    print()
    print("=" * 54)
    print("                    RÉSULTATS")
    print("=" * 54)
    print(f"  Nombre d'étudiants             : {n:,}")
    print(f"  Threads utilisés (N)           : {p}")
    print(f"  Temps séquentiel  (1 thread)   : {t_seq*1000:8.2f} ms")
    print(f"  Temps parallèle   ({p:>2} threads) : {t_par*1000:8.2f} ms")
    print("-" * 54)
    print(f"  Speedup   S = T1 / Tp          : {speedup:8.2f} x")
    print(f"  Efficacité  E = S / N          : {efficacite*100:8.1f} %")
    print("-" * 54)
    print(f"  Loi d'Amdahl - fraction //  P  : {P*100:8.2f} %")
    print(f"  Partie séquentielle  (1 - P)   : {(1-P)*100:8.2f} %")
    print(f"  Accélération max  1/(1-P)      : {speedup_max:8.2f} x")
    print("=" * 54)

    return {
        'n': n, 'p': p, 't_seq': t_seq, 't_par': t_par,
        'speedup': speedup, 'efficacite': efficacite,
        'P': P, 'speedup_max': speedup_max,
    }


if __name__ == '__main__':
    main()
