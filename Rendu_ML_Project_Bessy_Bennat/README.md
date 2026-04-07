# League of Legends — Match Prediction

Pipeline ML de prédiction des résultats de matchs League of Legends (victoire/défaite de l'équipe bleue) basé sur les données de l'API Riot Games.

---

## Installation

### Prérequis

Python 3.10+ requis.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost lightgbm requests
```

---

## Utilisation

### `main.py` — Pipeline CLI (recommandé)

Lance des tâches spécifiques avec le(s) modèle(s) de ton choix.

```bash
python main.py --tasks <tâches> --ml_method <modèles>
```

#### `--tasks` (séparés par des espaces)

| Valeur | Description |
|---|---|
| `balance` | Graphique d'équilibre des classes (wins vs losses) |
| `train` | Entraînement + évaluation complète (accuracy, F1, precision, recall, AUC, confusion matrix) |
| `pca` | Visualisation PCA (scree plot, variance cumulée, scatter 2D) |
| `cluster` | Clustering K-Means + silhouette + taux de victoire par cluster |
| `learning_curve` | Courbes d'apprentissage (biais/variance) |
| `calibration` | Courbes de calibration des probabilités |
| `all` | Toutes les tâches ci-dessus |

#### `--ml_method` (séparés par des virgules)

| Valeur | Modèle |
|---|---|
| `lr` | Logistic Regression |
| `lr_pca80` | LR + PCA (80% variance) |
| `lr_pca90` | LR + PCA (90% variance) |
| `lr_pca95` | LR + PCA (95% variance) |
| `lr_pca99` | LR + PCA (99% variance) |
| `sgd` | SGD Classifier (log_loss) |
| `knn` | K-Nearest Neighbors |
| `nb` | Gaussian Naive Bayes |
| `lda` | Linear Discriminant Analysis |
| `qda` | Quadratic Discriminant Analysis |
| `rf` | Random Forest |
| `gb` | Gradient Boosting |
| `all` | Tous les modèles |

#### Exemples

```bash
# Évaluation complète avec Random Forest
python main.py --tasks train --ml_method rf

# Tout faire avec LR et Gradient Boosting
python main.py --tasks all --ml_method lr,gb

# Comparer tous les modèles (long)
python main.py --tasks train --ml_method all

# PCA + clustering uniquement
python main.py --tasks pca cluster --ml_method lr

# Éval + calibration sur 3 modèles
python main.py --tasks train calibration --ml_method lr,rf,gb
```

> **Note :** `calibration` et `learning_curve` nécessitent que `train` soit aussi dans `--tasks`.


## Outputs

Tous les fichiers sont sauvegardés dans le dossier `outputs/` créé automatiquement.

| Fichier | Description |
|---|---|
| `results_summary.csv` | Tableau complet des métriques (accuracy, F1, AUC, ...) |
| `balance.png` | Équilibre des classes par configuration |
| `heatmap_auc.png` | Grille AUC — modèles × configurations |
| `heatmap_f1.png` | Grille F1 — modèles × configurations |
| `confusion_matrices.png` | Matrices de confusion (config 25 min) |
| `pca_variance.png` | Scree plot + variance cumulée |
| `pca_scatter2d.png` | Projection PCA 2D colorée par résultat |
| `cluster_silhouette.png` | Courbe silhouette K-Means |
| `cluster_scatter.png` | Clusters + taux de victoire par cluster |
| `learning_curves.png` | Courbes d'apprentissage biais/variance |
| `calibration.png` | Courbes de calibration des probabilités |

---

## Structure du projet

```
LoL_Match_Prediction/
├── main.py                        # Pipeline CLI (à utiliser)
├── pipeline_analysis.py           # Pipeline complet automatique
├── Data_recup1.ipynb              # Collecte des données via API Riot
├── Preprocessing_v3.ipynb         # Prétraitement et feature engineering
├── matchs_analysis.ipynb          # Analyse et modélisation (v1)
├── matchs_analysis_v2.ipynb       # Analyse et modélisation (v2 + visualisations)
├── matchs_preprocessed_*.csv      # Données prêtes à l'emploi (10 configs)
├── outputs/                       # Figures et résultats générés
└── data/                          # Données brutes JSON (API Riot)
```
