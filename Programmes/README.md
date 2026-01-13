# Projet TSP - M2 MIASHS

Ce projet implémente plusieurs algorithmes pour résoudre le problème du Voyageur de Commerce (TSP).

## Structure du Projet

Le code source est organisé dans le dossier `src` :

- **`src/model`** : Modèle de données et fonctions utilitaires (Lecture fichier, Distance, Coût).
- **`src/exact`** : Algorithme Branch & Bound (Question 2).
- **`src/constructive`** : Heuristique Nearest Neighbor (Question 3).
- **`src/local_search`** : Recherche locale 2-Opt (Question 4).
- **`src/grasp`** : Métaheuristique GRASP (Question 5).

## Installation

Aucune installation particulière n'est requise en dehors de Python 3.
Les fichiers de données doivent être placés dans `data/Input/`.

## Exécution

Vous pouvez exécuter chaque module indépendamment depuis la racine `Programmes/`.

**Exemple :**
```bash
# Lancer Nearest Neighbor
py src/constructive/nearest_neighbor.py

# Lancer GRASP
py src/grasp/grasp.py
```

## Expérimentations

Le script `Experiments_6.py` permet de lancer un comparatif complet.
```bash
py Experiments_6.py
```

## Auteurs
Équipe Projet TSP
