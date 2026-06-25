# TP — Paralléliser un calcul de moyenne avec Numba

> UVS / UNCHK — Module *Programmation parallèle* (Master 1 Big Data) — Enseignant : Osias Noël TOSSOU

Calcul de la moyenne pondérée d'une liste de **5 000 000 d'étudiants**, en
séquentiel puis en parallèle avec [Numba](https://numba.pydata.org/), mesure
du speedup et déduction de la fraction parallélisable via la **loi d'Amdahl**.

> Formule : `Moyenne = (Maths × 5 + Physique × 4 + Anglais × 2) / 11`

## Structure du projet

| Fichier | Rôle |
|---|---|
| `generate_data.py` | **Tâche 1** — génère le CSV (≥ 1 000 000 étudiants) |
| `data_io.py` | Constantes (coefficients) + chargement du CSV (partagé) |
| `moyenne_sequentiel.py` | **Tâche 2** — version séquentielle Numba (`@njit`, 1 thread) |
| `moyenne_parallele.py` | **Tâche 3** — version parallèle Numba (`parallel=True` + `prange`) |
| `benchmark.py` | **Tâches 4 & 5** — speedup + loi d'Amdahl |
| `RAPPORT.md` | **Tâche 6** — rapport détaillé avec résultats |

## Installation

```bash
# Créer et activer un environnement virtuel
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Générer les données (5 000 000 étudiants par défaut -> etudiants.csv)
python generate_data.py
#    ou un nombre précis :  python generate_data.py 1000000

# 2. Lancer le benchmark complet (speedup + Amdahl)
python benchmark.py
```

On peut aussi exécuter chaque version isolément :

```bash
python moyenne_sequentiel.py    # version séquentielle seule
python moyenne_parallele.py     # version parallèle seule
```

> Le fichier `etudiants.csv` n'est pas versionné (trop volumineux) : il est
> régénéré à l'identique par `generate_data.py` grâce à une graine aléatoire fixe.
