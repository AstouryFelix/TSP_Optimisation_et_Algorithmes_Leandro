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

## Visualisation des Résultats

Pour visualiser une solution dans l'interface web, suivez cette procédure :

### 1. Générer le fichier Solution (.out)
Modifiez le script de l'algorithme (ex: `src/grasp/grasp.py`) pour choisir votre fichier d'entrée (`filename`).
Ensuite, exécutez le script :
```bash
py src/grasp/grasp.py
```
Cela créera un fichier solution dans `data/Solutions/` (ex: `101_GRASP_Final.out`).

### 2. Générer les Données de Visualisation (.json)
Utilisez le script `Generate_Visu_Data.py` avec le fichier d'entrée et le fichier solution :

**Syntaxe :**
```bash
py Generate_Visu_Data.py <CHEMIN_INPUT> <CHEMIN_OUTPUT>
```

**Exemple concret :**
```bash
py Generate_Visu_Data.py ../data/Input/101.in ../data/Solutions/101_GRASP_Final.out
```

### 3. Afficher
Ouvrez `visualization/index.html` dans votre navigateur et chargez le fichier JSON généré (ex: `101_GRASP_Final.json`).

## Auteurs
Équipe Projet TSP
