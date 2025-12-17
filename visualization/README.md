# 🗺️ TSP Visualizer - Application de Visualisation Interactive

Application web interactive pour visualiser les solutions du Problème du Voyageur de Commerce (TSP).

## 📋 Fonctionnalités

- **Upload de fichiers** : Support des formats `.tsp`, `.out`, et `.json`
- **Visualisation graphique** : Affichage des villes et des chemins sur un canvas HTML5
- **Comparaison** : Possibilité d'afficher la solution initiale et optimisée
- **Interaction** : Zoom et pan pour explorer les grandes instances
- **Statistiques** : Affichage du coût et de l'amélioration

## 🚀 Utilisation

### 1. Ouvrir l'application
Double-cliquez sur `index.html` dans votre navigateur web.

### 2. Charger un fichier .tsp
- Cliquez sur "Fichier .tsp (Coordonnées)"
- Sélectionnez votre fichier `.tsp` (ex: `ali535.tsp`)

### 3. Charger une solution (Optionnel)
**Option A : Fichier .out**
- Cliquez sur "Fichier .out (Solution)"
- Sélectionnez le fichier `.out` généré par votre algorithme

**Option B : Fichier .json complet**
- Cliquez sur "Fichier .json (Solution complète)"
- Sélectionnez le fichier `.json` exporté par le script Python modifié

### 4. Visualiser
- Cliquez sur "🎨 Visualiser"
- Utilisez les options d'affichage pour personnaliser la vue

## 📂 Formats de Fichiers Supportés

### Fichier .tsp (TSPLIB)
```
NAME: ali535
TYPE: TSP
DIMENSION: 535
EDGE_WEIGHT_TYPE: GEO
NODE_COORD_SECTION
1  36.49  7.49
2  57.06  9.51
...
EOF
```

### Fichier .out (Solution simple)
```
0 92 27 66 57 60 ...
23239
```

### Fichier .json (Solution complète - Recommandé)
```json
{
  "instance": "ali535",
  "n_cities": 535,
  "coordinates": [[36.49, 7.49], [57.06, 9.51], ...],
  "initial_path": [0, 1, 2, ...],
  "initial_cost": 50000,
  "optimized_path": [0, 92, 27, ...],
  "optimized_cost": 45000,
  "improvement": 5000
}
```

## 🎨 Options d'Affichage

- **Afficher les villes** : Points représentant les villes
- **Afficher le chemin** : Trajet optimisé
- **Afficher solution initiale** : Trajet avant optimisation (uniquement avec fichier .json)
- **Afficher les numéros** : Identifiants des villes
- **Réinitialiser le zoom** : Revenir à la vue par défaut

## 🖱️ Contrôles

- **Clic + Déplacement** : Pan (déplacer la carte)
- **Molette de la souris** : Zoom avant/arrière

## 🔧 Génération du fichier JSON

Pour générer un fichier `.json` compatible, utilisez le script Python modifié `LocalSearchTSP_FichierTSP.py` avec l'export JSON activé.

## 💡 Conseils

- Pour de meilleures performances, utilisez le fichier `.json` qui contient toutes les données nécessaires
- Pour les grandes instances (>500 villes), désactivez "Afficher les numéros"
- Utilisez le zoom pour explorer les détails des grandes instances

## 🐛 Dépannage

**Le visualiser ne s'active pas ?**
- Vérifiez que vous avez chargé au minimum un fichier `.tsp`

**Les coordonnées sont bizarres ?**
- Vérifiez que le format de votre fichier `.tsp` est correct
- Le format TSPLIB est requis

**La solution ne s'affiche pas ?**
- Vérifiez que le fichier `.out` ou `.json` est au bon format
- Les indices doivent correspondre aux villes du `.tsp`

---

Développé pour le projet TSP - M2 Optimisation
