"""
main.py
-------
Pipeline CLI de prédiction de matchs League of Legends.

Usage :
    python main.py --tasks all --ml_method rf
    python main.py --tasks train --ml_method lr,rf,gb
    python main.py --tasks train pca --ml_method all

Tâches disponibles (--tasks) :
    balance         Graphique d'équilibre des classes
    train           Entraînement + évaluation complète (F1, precision, recall, AUC, ...)
    pca             Visualisation PCA (scree plot, variance cumulée, scatter 2D)
    cluster         Clustering K-Means + taux de victoire par cluster
    learning_curve  Courbes d'apprentissage (biais/variance)
    calibration     Courbes de calibration des probabilités
    all             Toutes les tâches ci-dessus

Modèles disponibles (--ml_method) :
    lr              Logistic Regression
    lr_pca80        LR + PCA (80% variance)
    lr_pca90        LR + PCA (90% variance)
    lr_pca95        LR + PCA (95% variance)
    lr_pca99        LR + PCA (99% variance)
    sgd             SGD Classifier (log_loss)
    knn             K-Nearest Neighbors
    nb              Gaussian Naive Bayes
    lda             Linear Discriminant Analysis
    qda             Quadratic Discriminant Analysis
    rf              Random Forest
    gb              Gradient Boosting
    all             Tous les modèles
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, precision_score,recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# CONFIGURATION
OUTPUT_DIR   = Path("outputs")
CSV_PATTERN  = "matchs_preprocessed_*.csv"
RANDOM_STATE = 42

ALL_TASKS = ["balance", "train", "pca", "cluster", "learning_curve", "calibration"]

MODEL_REGISTRY = {
    "lr": LogisticRegression(max_iter=1000),
    "lr_pca80": Pipeline([("pca", PCA(n_components=0.80)),("model", LogisticRegression(max_iter=1000))]),
    "lr_pca90": Pipeline([("pca", PCA(n_components=0.90)),("model", LogisticRegression(max_iter=1000))]),
    "lr_pca95": Pipeline([("pca", PCA(n_components=0.95)),("model", LogisticRegression(max_iter=1000))]),
    "lr_pca99" : Pipeline([("pca", PCA(n_components=0.99)),("model", LogisticRegression(max_iter=1000))]),
    "sgd": SGDClassifier(loss="log_loss", max_iter=1000,random_state=RANDOM_STATE),
    "knn" : KNeighborsClassifier(n_neighbors=99),
    "nb" : GaussianNB(),
    "lda": LinearDiscriminantAnalysis(),
    "qda": QuadraticDiscriminantAnalysis(reg_param=0.1),
    "rf": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

MODEL_LABELS = {
    "lr": "LogisticRegression",
    "lr_pca80": "LR + PCA(80%)",
    "lr_pca90" :"LR + PCA(90%)",
    "lr_pca95": "LR + PCA(95%)",
    "lr_pca99": "LR + PCA(99%)",
    "sgd": "SGD (log_loss)",
    "knn": "KNN",
    "nb" :"Gaussian NB",
    "lda": "LDA",
    "qda" : "QDA",
    "rf": "RandomForest",
    "gb": "GradientBoosting",
}



# UTILITAIRES
def save_fig(name: str, dpi: int = 150) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   clone(model)),
    ])


def load_csv_files() -> list:
    files = sorted(glob.glob(CSV_PATTERN))
    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé : '{CSV_PATTERN}'")
    return files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline ML — prédiction de matchs League of Legends",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--tasks", nargs="+", required=True,
        help="Tâches à exécuter (séparées par des espaces ou virgules) :\n"
             "  balance, train, pca, cluster, learning_curve, calibration, all"
    )
    parser.add_argument(
        "--ml_method", type=str, required=True,
        help="Modèle(s) à utiliser (séparés par des virgules) :\n"
             "  lr, lr_pca80, lr_pca90, lr_pca95, lr_pca99,\n"
             "  sgd, knn, nb, lda, qda, rf, gb, all"
    )
    return parser.parse_args()


def resolve_tasks(raw: list) -> list:
    """Aplatit les virgules et résout 'all'."""
    tokens = []
    for t in raw:
        tokens.extend(t.split(","))
    tokens = [t.strip().lower() for t in tokens if t.strip()]
    if "all" in tokens:
        return ALL_TASKS
    invalid = [t for t in tokens if t not in ALL_TASKS]
    if invalid:
        print(f"[ERREUR] Tâches inconnues : {invalid}")
        print(f"Disponibles : {ALL_TASKS + ['all']}")
        sys.exit(1)
    return tokens


def resolve_models(raw: str) -> dict:
    """Retourne un sous-dict de MODEL_REGISTRY."""
    tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
    if "all" in tokens:
        return MODEL_REGISTRY
    invalid = [t for t in tokens if t not in MODEL_REGISTRY]
    if invalid:
        print(f"[ERREUR] Modèles inconnus : {invalid}")
        print(f"Disponibles : {list(MODEL_REGISTRY.keys()) + ['all']}")
        sys.exit(1)
    return {k: MODEL_REGISTRY[k] for k in tokens}


# TÂCHE : BALANCE
def task_balance(csv_files: list) -> None:
    print("\n[BALANCE] Equilibre des classes...")
    rows = []
    for path in csv_files:
        cfg = path.replace("matchs_preprocessed_", "").replace(".csv", "")
        df  = pd.read_csv(path, index_col=0)
        if "win" not in df.columns:
            continue
        wins   = df["win"].sum()
        losses = len(df) - wins
        rows.append({"config": cfg, "wins": int(wins), "losses": int(losses),"total": len(df), "pct_win": round(100 * wins / len(df), 1)})
    df_bal = pd.DataFrame(rows)
    print(df_bal.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df_bal))
    axes[0].bar(x - 0.2, df_bal["wins"], 0.4, label="Victoires",color="#3498db")
    axes[0].bar(x + 0.2, df_bal["losses"], 0.4, label="Defaites",color="#e74c3c")
    axes[0].set_xticks(x); axes[0].set_xticklabels(df_bal["config"], rotation=45, ha="right")
    axes[0].set_title("Wins vs Losses par configuration"); axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, df_bal["pct_win"], color="#2ecc71")
    axes[1].axhline(50, color="black", linestyle="--", alpha=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(df_bal["config"], rotation=45, ha="right")
    axes[1].set_ylabel("% victoires equipe bleue"); axes[1].set_title("Taux de victoire")
    axes[1].set_ylim(40, 60); axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig("balance")


# ──────────────────────────────────────────────────────────────────────────────
# TÂCHE : TRAIN
# ──────────────────────────────────────────────────────────────────────────────
def task_train(csv_files: list, models: dict) -> tuple:
    """
    Entraîne chaque modèle sur chaque config et retourne
    (results_df, fitted_pipes, test_sets).
    Métriques : Accuracy, F1 (weighted), Precision, Recall, AUC, CV-AUC.
    """
    print("\n[TRAIN] Entraînement et évaluation...")
    results= []
    fitted_pipes= {}
    test_sets= {}

    for csv_path in csv_files:
        cfg = csv_path.replace("matchs_preprocessed_", "").replace(".csv", "")
        df= pd.read_csv(csv_path, index_col=0)
        if "win" not in df.columns:
            continue

        fc = [c for c in df.select_dtypes(include="number").columns if c != "win"]
        X, y = df[fc], df["win"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

        fitted_pipes[cfg] = {}
        test_sets[cfg] = (X_test, y_test)

        for key, model in models.items():
            label = MODEL_LABELS[key]
            pipe= make_pipeline(model)

            cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            acc= accuracy_score(y_test, y_pred)
            f1= f1_score(y_test, y_pred, average="weighted")
            precision= precision_score(y_test, y_pred, average="weighted",zero_division=0)
            recall= recall_score(y_test, y_pred, average="weighted",zero_division=0)
            auc = roc_auc_score(y_test, y_proba)

            fitted_pipes[cfg][label] = pipe
            results.append({
                "config": cfg,
                "model": label,
                "model_key": key,
                "n_features": X.shape[1],
                "cv_auc": round(cv_auc, 4),
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "test_auc": round(auc, 4),
            })
            print(f"  [{cfg:<14}] {label:<22} "
                  f"cv={cv_auc:.3f}  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")

    results_df = pd.DataFrame(results)

    #  Résumé : top 10 par AUC 
    print("\n-- TOP 10 (test AUC) --")
    print(results_df.sort_values("test_auc", ascending=False)
          .head(10)[["config", "model", "accuracy", "f1", "precision",
                      "recall", "test_auc"]].to_string(index=False))

    #  Moyenne par modèle 
    print("\n-- Moyenne par modèle (toutes configs) --")
    print(results_df.groupby("model")[["accuracy", "f1", "precision",
                                        "recall", "test_auc"]]
          .mean().round(4).sort_values("test_auc", ascending=False)
          .to_string())

    #  Classification report + confusion matrix sur config 25 min 
    cfg_25 = "25"
    if cfg_25 in fitted_pipes:
        print(f"\n-- Classification report (config {cfg_25} min) --")
        X_te, y_te = test_sets[cfg_25]
        for key, model in models.items():
            label = MODEL_LABELS[key]
            if label not in fitted_pipes[cfg_25]:
                continue
            pipe= fitted_pipes[cfg_25][label]
            y_pred =pipe.predict(X_te)
            print(f"\n  [{label}]")
            print(classification_report(y_te, y_pred,
                                        target_names=["Defaite bleue", "Victoire bleue"],
                                        digits=3))

        # Confusion matrices visuelles
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]
        for ax, (key, _) in zip(axes, models.items()):
            label = MODEL_LABELS[key]
            if label not in fitted_pipes[cfg_25]:
                continue
            pipe= fitted_pipes[cfg_25][label]
            y_pred = pipe.predict(X_te)
            cm = confusion_matrix(y_te, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Defaite", "Victoire"],
                        yticklabels=["Defaite", "Victoire"])
            auc_val = results_df.query(
                f"config == '{cfg_25}' and model_key == '{key}'"
            )["test_auc"].values
            title_auc = f" AUC={auc_val[0]:.3f}" if len(auc_val) else ""
            ax.set_title(f"{label}\n(config 25 min){title_auc}", fontsize=10)
            ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
        plt.tight_layout()
        save_fig("confusion_matrices")

    # Heatmap AUC 
    pivot = results_df.pivot_table(index="model", columns="config", values="test_auc")
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2),max(6, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_title("AUC ROC — modèles x configurations", fontsize=13)
    ax.set_xlabel("Configuration (minutes)")
    plt.tight_layout()
    save_fig("heatmap_auc")

    # Heatmap F1 
    pivot_f1 = results_df.pivot_table(index="model", columns="config",values="f1")
    fig, ax = plt.subplots(figsize=(max(10, len(pivot_f1.columns) * 1.2),max(6, len(pivot_f1) * 0.6)))
    sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="RdYlGn",vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_title("F1 Score — modèles x configurations", fontsize=13)
    ax.set_xlabel("Configuration (minutes)")
    plt.tight_layout()
    save_fig("heatmap_f1")

    # Sauvegarde CSV 
    out = OUTPUT_DIR / "results_summary.csv"
    results_df.sort_values("test_auc", ascending=False).to_csv(out, index=False)
    print(f"\n  -> {out}")

    return results_df, fitted_pipes, test_sets


# PCA
def task_pca(csv_files: list) -> tuple:
    print("\n[PCA] Analyse en composantes principales (config 25 min)...")
    csv_25 = next(
        (f for f in csv_files
         if f.replace("matchs_preprocessed_", "").replace(".csv", "") == "25"),
        csv_files[-1])

    df= pd.read_csv(csv_25, index_col=0)
    fc= [c for c in df.select_dtypes(include="number").columns if c != "win"]
    X= df[fc].copy()
    y= df["win"].astype(int).reset_index(drop=True)

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imp.fit_transform(X))

    pca_full= PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    thresholds = {80: None, 90: None, 95: None, 99: None}
    for t in thresholds:
        thresholds[t] = int(np.searchsorted(cumulative, t / 100)) + 1
        print(f"{t}% variance -> {thresholds[t]} composantes")
    print(f"Total features : {X_scaled.shape[1]}")

    # Variance cumulée
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_show = 40
    axes[0].bar(range(1, n_show + 1), explained[:n_show] * 100,color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("Composante principale"); axes[0].set_ylabel("Variance (%)")
    axes[0].set_title("Scree plot"); axes[0].grid(axis="y", alpha=0.3)

    colors = {80: "#2ecc71", 90: "#f39c12", 95: "#e74c3c", 99: "#9b59b6"}
    axes[1].plot(range(1, len(cumulative) + 1), cumulative * 100,color="steelblue", linewidth=2)
    for t, n in thresholds.items():
        axes[1].axhline(t, color=colors[t], linestyle="--", alpha=0.7)
        axes[1].axvline(n, color=colors[t], linestyle="--", alpha=0.7)
        axes[1].annotate(f"{t}% -> {n}",xy=(n, t), xytext=(n + 2, t - 6),fontsize=9, color=colors[t])
    axes[1].set_xlabel("Nombre de composantes"); axes[1].set_ylabel("Variance cumulée (%)")
    axes[1].set_title("Variance cumulée"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig("pca_variance")

    # Scatter 2D
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_scaled)
    y_arr= y.values

    fig, ax = plt.subplots(figsize=(9, 6))
    for val, label, color in [(0, "Defaite bleue", "#e74c3c"),(1, "Victoire bleue", "#3498db")]:
        mask = y_arr == val
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=12, alpha=0.35, c=color, label=label)
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Projection PCA 2D — config 25 min"); ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig("pca_scatter2d")

    n_95 = thresholds[95]
    return X_scaled, pca_2d, X_2d, y_arr, n_95


# CLUSTERING
def task_cluster(X_scaled, pca_2d, X_2d, y_arr, n_95) -> None:
    print("\n[CLUSTER] K-Means clustering...")
    pca_95 = PCA(n_components=n_95, random_state=RANDOM_STATE)
    X_95 = pca_95.fit_transform(X_scaled)
    k_range = range(2, 11)
    sil_scores = []

    for k in k_range:
        km= KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lbl =km.fit_predict(X_95)
        sil_scores.append(silhouette_score(X_95, lbl))
        print(f"k={k} silhouette={sil_scores[-1]:.4f}")

    best_k = k_range.start + int(np.argmax(sil_scores))
    print(f"Meilleur k = {best_k}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), sil_scores, "o-", color="steelblue", linewidth=2)
    ax.axvline(best_k, color="#e74c3c", linestyle="--",label=f"Meilleur k={best_k}")
    ax.set_xlabel("Nombre de clusters k"); ax.set_ylabel("Silhouette score")
    ax.set_title("Courbe silhouette K-Means"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig("cluster_silhouette")

    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels= km_best.fit_predict(X_95)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.get_cmap("tab10", best_k)
    for k in range(best_k):
        mask = labels == k
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],s=12, alpha=0.4, color=cmap(k), label=f"Cluster {k}")
    axes[0].set_title(f"Clusters K-Means (k={best_k}) — PCA 2D")
    axes[0].legend(markerscale=2); axes[0].grid(True, alpha=0.3)

    wr_per_cluster = []
    for k in range(best_k):
        mask = labels == k
        wr = y_arr[mask].mean()
        wr_per_cluster.append({"cluster": k, "n": mask.sum(),"win_rate": round(wr * 100, 1)})
    df_wr = pd.DataFrame(wr_per_cluster)
    axes[1].bar(df_wr["cluster"].astype(str), df_wr["win_rate"],color=[cmap(k) for k in df_wr["cluster"]], edgecolor="black")
    axes[1].axhline(50, color="black", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Cluster"); axes[1].set_ylabel("Taux de victoire (%)")
    axes[1].set_title("Taux de victoire par cluster"); axes[1].grid(axis="y", alpha=0.3)
    for _, row in df_wr.iterrows():
        axes[1].text(int(row["cluster"]), row["win_rate"] + 0.5,f"n={row['n']}", ha="center", fontsize=9)
    plt.tight_layout()
    save_fig("cluster_scatter")
    print(df_wr.to_string(index=False))



# LEARNING CURVE
def task_learning_curve(csv_files: list, models: dict) -> None:
    print("\n[LEARNING CURVE] Courbes d'apprentissage (config 25 min)...")
    csv_25 = next(
        (f for f in csv_files if f.replace("matchs_preprocessed_", "").replace(".csv", "") == "25"),csv_files[-1])

    df = pd.read_csv(csv_25, index_col=0)
    fc = [c for c in df.select_dtypes(include="number").columns if c != "win"]
    X, y = df[fc], df["win"].astype(int)

    # Limiter à 3 modèles pour lisibilité
    keys_lc = list(models.keys())[:3]
    fig, axes = plt.subplots(1, len(keys_lc), figsize=(7 * len(keys_lc), 5))
    if len(keys_lc) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys_lc):
        label = MODEL_LABELS[key]
        pipe  = make_pipeline(models[key])
        print(f"{label}...")
        tr_sizes, tr_scores, val_scores = learning_curve(
            pipe, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring="roc_auc", n_jobs=1)
        tm, ts = tr_scores.mean(axis=1),tr_scores.std(axis=1)
        vm, vs = val_scores.mean(axis=1),val_scores.std(axis=1)
        ax.plot(tr_sizes, tm, "o-", color="#3498db", label="Train AUC")
        ax.fill_between(tr_sizes,tm -ts, tm +ts,alpha=0.1, color="#3498db")
        ax.plot(tr_sizes,vm,"o-", color="#e74c3c", label="Val AUC (CV)")
        ax.fill_between(tr_sizes, vm - vs, vm + vs, alpha=0.1, color="#e74c3c")
        ax.set_title(f"Learning Curve — {label}", fontsize=11)
        ax.set_xlabel("Taille du train"); ax.set_ylabel("AUC ROC")
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    save_fig("learning_curves")



# CALIBRATION
def task_calibration(fitted_pipes: dict, test_sets: dict) -> None:
    print("\n[CALIBRATION] Courbes de calibration (config 25 min)...")
    if "25" not in fitted_pipes:
        print("[SKIP] Config 25 min non disponible.")
        return

    X_te, y_te = test_sets["25"]
    available_models = list(fitted_pipes["25"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibration parfaite")

    for label in available_models:
        pipe   = fitted_pipes["25"][label]
        y_prob = pipe.predict_proba(X_te)[:, 1]
        frac, mean_p = calibration_curve(y_te, y_prob, n_bins=10)
        axes[0].plot(mean_p, frac, marker="o", linewidth=2, label=label)
        axes[1].hist(y_prob, bins=30, alpha=0.4, label=label, density=True)

    axes[0].set_xlabel("Probabilité prédite moyenne")
    axes[0].set_ylabel("Fraction de positifs réels")
    axes[0].set_title("Courbes de calibration — config 25 min")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Probabilité prédite")
    axes[1].set_ylabel("Densité")
    axes[1].set_title("Distribution des probabilités prédites")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig("calibration")


##############################################################################
#                               MAIN                                         #
##############################################################################
def main() -> None:
    args   = parse_args()
    tasks  = resolve_tasks(args.tasks)
    models = resolve_models(args.ml_method)

    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"\n[CONFIG] Tâches : {tasks}")
    print(f"[CONFIG] Modèles: {[MODEL_LABELS[k] for k in models]}")
    print(f"[CONFIG] Outputs: {OUTPUT_DIR.resolve()}\n")

    csv_files = load_csv_files()
    print(f"[DATA] {len(csv_files)} configurations CSV trouvées.")

    results_df= None
    fitted_pipes = {}
    test_sets= {}
    pca_state= None

    if "balance" in tasks:
        task_balance(csv_files)

    if "train" in tasks:
        results_df, fitted_pipes, test_sets = task_train(csv_files, models)

    if "pca" in tasks or "cluster" in tasks:
        pca_state = task_pca(csv_files)

    if "cluster" in tasks:
        if pca_state is None:
            pca_state = task_pca(csv_files)
        task_cluster(*pca_state)

    if "learning_curve" in tasks:
        task_learning_curve(csv_files, models)

    if "calibration" in tasks:
        if not fitted_pipes:
            print("[WARN] 'calibration' nécessite 'train'. Lance avec --tasks train calibration")
        else:
            task_calibration(fitted_pipes, test_sets)

    print(f"\n[DONE] Pipeline terminé en {time.time() - t0:.1f}s")
    print(f"[DONE] Fichiers sauvegardés dans : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
