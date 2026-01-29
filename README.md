# Projet TSP - Problème du Voyageur de Commerce

## Équipe 1

**Membres :**
- ASTOURY Félix
- BOUSSEAU Victor
- LANNUZEL Elliot
- ZNEDI Yacine

---

## Organisation du Projet

```
team_1/
├── README.md                    # Ce fichier
├── report/
│   └── report_team_1.pdf        # Rapport au format PDF
├── src/
│   ├── model/                   # Code partagé (graphe, fonctions de base)
│   │   ├── load_data.py         # Lecture des fichiers d'instance (.in et .tsp)
│   │   ├── save_solution.py     # Sauvegarde des solutions (.out)
│   │   ├── total_cost.py        # Calcul du coût d'un cycle
│   │   └── export_to_json.py    # Export JSON pour visualisation
│   ├── exact/                   # Algorithme exact (Branch and Bound)
│   │   └── BB.py
│   ├── constructive/            # Heuristique constructive (Nearest Neighbor)
│   │   └── Constructive.py
│   ├── local_search/            # Recherche locale (2-Opt)
│   │   └── LocalSearch.py
│   └── grasp/                   # Métaheuristique GRASP
│       └── GraspTSP.py
└── instances/
    ├── exact/                   # Instances pour l'algorithme exact
    ├── constructive/            # Instances pour le constructif
    ├── local_search/            # Instances pour la recherche locale
    ├── grasp/                   # Instances pour GRASP
    └── new_instances/           # Instances comparatives finales
```

---

## Prérequis

- **Python 3.8+**
- Bibliothèques requises :
  ```bash
  pip install numpy scipy matplotlib
  ```

---

## Compilation / Exécution

### Exécution d'un algorithme individuel

Tous les scripts sont paramétrables et prennent en argument le chemin vers le fichier d'instance (fichiers `.tsp` ou `.in`).

Depuis la racine du projet :

```bash
# Algorithme Exact (Branch & Bound)
python src/exact/BB.py instances/exact/17.in

# Heuristique Constructive (Nearest Neighbor)
python src/constructive/Constructive.py instances/constructive/100.in

# Recherche Locale (2-Opt)
python src/local_search/LocalSearch.py instances/local_search/ali535.tsp

# GRASP
python src/grasp/GraspTSP.py instances/grasp/ali535.tsp
```

Si aucun argument n'est fourni, une instance par défaut (ex: `ali535.tsp` ou `100.in`) est utilisée pour un test rapide.
Les fichiers de sortie (`.out` et `.json`) sont automatiquement générés dans le dossier `Solutions/`.

---

## Format des Fichiers

### Fichier d'entrée (.in)

Le fichier d'entrée contient `n + 1` lignes :
- Ligne 1 : le nombre `n` de sommets
- Lignes 2 à n+1 : la matrice d'adjacence (poids séparés par des espaces)

**Exemple (`test.in`) :**
```
4
0 1 3 2
1 0 2 4
3 2 0 1
2 4 1 0
```

### Fichier de sortie (.out)

Le fichier de sortie doit être nommé `{instance}_{method}.out` et contient :
- Ligne 1 : les numéros des sommets séparés par des espaces (indices commençant à 1)
- Ligne 2 : le poids total du cycle

**Exemple (`test_exact.out`) :**
```
1 2 3 4
6
```

Ce résultat correspond au cycle 1 → 2 → 3 → 4 → 1 avec un coût total de 6.

---

## Emplacement des Fichiers

| Type | Emplacement |
|------|-------------|
| Fichiers d'entrée | `instances/{method}/` |
| Fichiers de sortie (.out) | `Solutions/` |
| Fichiers JSON (visualisation) | `Solutions/` |

---

## Visualisation

Le projet inclut un export JSON pour visualiser les solutions. Les fichiers `.json` sont générés automatiquement avec les fichiers `.out`.

---

## Algorithmes Implémentés

1. **Exact (Branch & Bound)** : Résolution optimale utilisant la programmation linéaire relaxée
2. **Constructive (Nearest Neighbor)** : Heuristique gloutonne du plus proche voisin
3. **Local Search (2-Opt)** : Amélioration locale par échanges d'arêtes
4. **GRASP** : Métaheuristique combinant construction randomisée et amélioration locale

---

## Solutions Optimales de Référence

| Instance | Solution Optimale |
|----------|-------------------|
| ali535   | 202339            |
| att48    | 10628             |
