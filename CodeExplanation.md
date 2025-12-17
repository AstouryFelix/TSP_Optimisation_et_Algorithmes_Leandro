# Explication du Code : Problème du Voyageur de Commerce (TSP)

Ce document explique les algorithmes utilisés dans le fichier `LocalSearchTSP.py` pour résoudre le problème du voyageur de commerce.

## 1. Contexte
Le but est de trouver le chemin le plus court qui passe par toutes les villes une seule fois et revient au point de départ.
Nous utilisons deux approches combinées :
1.  **Heuristique Constructive (Nearest Neighbor)** : Pour trouver rapidement une solution initiale "correcte".
2.  **Recherche Locale (2-Opt)** : Pour améliorer cette solution en "démêlant" le chemin.

---

## 2. Heuristique Constructive : Le Plus Proche Voisin (Nearest Neighbor)

### Principe
L'idée est simple : "Je vais à la ville la plus proche que je n'ai pas encore visitée."

### Exemple
Imaginons 4 villes (A, B, C, D) avec les distances suivantes :
*   A -> B : 10
*   A -> C : 15
*   A -> D : 20
*   B -> C : 35
*   B -> D : 25
*   C -> D : 30

**Algorithme pas à pas (Départ de A) :**
1.  On est à **A**. Villes non visitées : {B, C, D}.
2.  On regarde les distances :
    *   A -> B = 10 (La plus petite)
    *   A -> C = 15
    *   A -> D = 20
3.  On va à **B**. Chemin actuel : [A, B]. Villes non visitées : {C, D}.
4.  On est à **B**. On regarde les voisins restants :
    *   B -> C = 35
    *   B -> D = 25 (La plus petite)
5.  On va à **D**. Chemin actuel : [A, B, D]. Villes non visitées : {C}.
6.  On est à **D**. Il ne reste que **C**.
7.  On va à **C**. Chemin actuel : [A, B, D, C].
8.  On boucle (retour à A).

**Résultat final** : A -> B -> D -> C -> A.
**Coût** : 10 + 25 + 30 + 15 = 80.

---

## 3. Recherche Locale : 2-Opt (Décroisement)

### Principe
Cette méthode essaie d'améliorer un chemin existant. Elle regarde si en croisant deux arêtes, on obtient un chemin plus court.
Si on a un chemin `... A -> B ... C -> D ...`, on regarde si faire `... A -> C ... B -> D ...` est mieux.
Concrètement, cela revient à **inverser un segment du chemin**.

### Exemple Visuel
Imaginez que votre chemin forme un "nœud papillon" (un croisement). C'est souvent inefficace.
Le 2-opt va supprimer les deux arêtes qui se croisent et les reconnecter différemment pour supprimer le croisement.

**Algorithme :**
Supposons le chemin : `[1, 2, 3, 4, 1]` (Coût total inconnu pour l'exemple).

On teste l'échange des arêtes (1-2) et (3-4).
*   Chemin actuel : 1 -> **2 -> 3** -> 4 -> 1
*   On coupe entre 1-2 et 3-4.
*   On reconnecte 1 avec 3, et 2 avec 4.
*   Cela revient à **inverser** le segment entre les coupures (ici le segment [2, 3]).
*   Nouveau chemin candidat : 1 -> **3 -> 2** -> 4 -> 1.

Si `distance(1,3) + distance(2,4) < distance(1,2) + distance(3,4)`, alors on a amélioré le coût ! On garde ce nouveau chemin.

---

## 4. Structure du Code Python

### `read_instance(filename)`
*   Lit le fichier `.in`.
*   Récupère `n` (nombre de villes) et la `matrix` (tableau des distances).

### `constructive_nearest_neighbor(n, matrix)`
*   Crée une liste `unvisited`.
*   Boucle tant qu'il reste des villes.
*   Cherche le minimum dans la ligne de la ville courante.

### `local_search_2opt(path, matrix)`
*   Boucle `while improved`: Tant qu'on trouve mieux, on continue.
*   Deux boucles `for` (i et j) pour tester toutes les paires d'arêtes possibles.
*   `if new_cost < current_cost`: Comparaison simple.
*   `path[i+1 : j+1] = path[i+1 : j+1][::-1]`: L'astuce Python pour inverser le segment et effectuer le recâblage.

### `calculate_total_cost(path, matrix)`
*   Fonction utilitaire pour additionner les distances du parcours.
