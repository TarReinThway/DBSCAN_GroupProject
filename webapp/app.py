from flask import Flask, render_template, request, flash, redirect, url_for
import smtplib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
import io, base64

app = Flask(__name__)

def fig_to_base64(fig): 
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def make_plot(X, labels, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = plt.cm.tab10.colors

    for i, label in enumerate(sorted(set(labels))):
        mask = labels == label if isinstance(labels, np.ndarray) else [l == label for l in labels]
        mask = np.array(mask)
        if isinstance(label, str):
            is_noise = label == '-1'
            display = 'Noise' if is_noise else f'Cluster {label}'
        else:
            is_noise = label == -1
            display = 'Noise' if is_noise else f'Cluster {label}'

        color = '#e53e3e' if is_noise else palette[i % 10]
        marker = 'x' if is_noise else 'o'

        if len(X.shape) == 2 and X.shape[1] >= 2:
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[color], marker=marker, s=50, alpha=0.8, label=display)
        else:
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[color], marker=marker, s=50, alpha=0.8, label=display)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    return fig_to_base64(fig)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/mall', methods=['GET', 'POST'])
def mall():
    dbscan_img = None
    kmeans_img = None
    stats = None

    eps        = 0.25
    min_samples = 4
    n_clusters  = 5

    if request.method == 'POST':
        eps         = float(request.form.get('eps', 0.25))
        min_samples = int(request.form.get('min_samples', 4))
        n_clusters  = int(request.form.get('n_clusters', 5))

        df = pd.read_csv('Mall_Customers.csv')
        df.drop(columns=['CustomerID', 'Age', 'Gender'], inplace=True)

        scaler    = RobustScaler()
        df_scaled = scaler.fit_transform(df)

        # DBSCAN plot
        cluster = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = cluster.fit_predict(df_scaled)
        dbscan_img = make_plot(
            df_scaled, db_labels,
            f'DBSCAN  (eps={eps}, min_samples={min_samples})',
            'Annual Income (k$)', 'Spending Score (1-100)'
        )

        # K-Means plot
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km_labels = km.fit_predict(df_scaled)

        fig, ax = plt.subplots(figsize=(6, 5))
        palette = plt.cm.tab10.colors
        for i, label in enumerate(sorted(set(km_labels))):
            mask = km_labels == label
            ax.scatter(df_scaled[mask, 0], df_scaled[mask, 1],
                       c=[palette[i % 10]], s=50, alpha=0.8, label=f'Cluster {label}')
        c = km.cluster_centers_
        ax.scatter(c[:, 0], c[:, 1], c='black', marker='X', s=150, zorder=5, label='Centroids')
        ax.set_title(f'K-Means  (k={n_clusters})', fontsize=12, pad=10)
        ax.set_xlabel('Annual Income (scaled)')
        ax.set_ylabel('Spending Score (scaled)')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        kmeans_img = fig_to_base64(fig)

        db_mask = db_labels != -1
        sil_db   = round(silhouette_score(df_scaled[db_mask], db_labels[db_mask]), 3) if len(set(db_labels)) - (1 if -1 in db_labels else 0) >= 2 else 'N/A'
        sil_km   = round(silhouette_score(df_scaled, km_labels), 3)

        stats = {
            'dbscan_clusters':   len(set(db_labels)) - (1 if -1 in db_labels else 0),
            'noise_points':      int((db_labels == -1).sum()),
            'kmeans_clusters':   n_clusters,
            'total_points':      len(df),
            'sil_dbscan':        sil_db,
            'sil_kmeans':        sil_km,
        }


    return render_template('mall.html',
                           dbscan_img=dbscan_img,
                           kmeans_img=kmeans_img,
                           stats=stats,
                           eps=eps,
                           min_samples=min_samples,
                           n_clusters=n_clusters)


@app.route('/moon', methods=['GET', 'POST'])
def moon():
    dbscan_img = None
    kmeans_img = None
    stats = None

    n_samples   = 1000
    noise       = 0.05
    eps         = 0.20
    min_samples = 5
    n_clusters  = 2

    if request.method == 'POST':
        n_samples   = int(request.form.get('n_samples', 1000))
        noise       = float(request.form.get('noise', 0.05))
        eps         = float(request.form.get('eps', 0.20))
        min_samples = int(request.form.get('min_samples', 5))
        n_clusters  = int(request.form.get('n_clusters', 2))

        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

        # DBSCAN plot
        db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        dbscan_img = make_plot(
            X, db_labels,
            f'DBSCAN  (eps={eps}, min_samples={min_samples})',
            'x1', 'x2'
        )

        # K-Means plot
        km_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)

        fig, ax = plt.subplots(figsize=(6, 5))
        palette = plt.cm.tab10.colors
        for i, label in enumerate(sorted(set(km_labels))):
            mask = km_labels == label
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[palette[i % 10]], s=50, alpha=0.8, label=f'Cluster {label}')
        c = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X).cluster_centers_
        ax.scatter(c[:, 0], c[:, 1], c='black', marker='X', s=150, zorder=5, label='Centroids')
        ax.set_title(f'K-Means  (k={n_clusters})', fontsize=12, pad=10)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        kmeans_img = fig_to_base64(fig)

        stats = {
            'dbscan_clusters': len(set(db_labels)) - (1 if -1 in db_labels else 0),
            'noise_points':    int((db_labels == -1).sum()),
            'kmeans_clusters': n_clusters,
            'total_points':    n_samples,
        }

        db_mask = db_labels != -1
        sil_db   = round(silhouette_score(X[db_mask], db_labels[db_mask]), 3) if len(set(db_labels)) - (1 if -1 in db_labels else 0) >= 2 else 'N/A'
        sil_km   = round(silhouette_score(X, km_labels), 3)

        stats = {
            'dbscan_clusters':   len(set(db_labels)) - (1 if -1 in db_labels else 0),
            'noise_points':      int((db_labels == -1).sum()),
            'kmeans_clusters':   n_clusters,
            'total_points':      n_samples,
            'sil_dbscan':        sil_db,
            'sil_kmeans':        sil_km,
        }


    return render_template('moon.html',
                           dbscan_img=dbscan_img,
                           kmeans_img=kmeans_img,
                           stats=stats,
                           n_samples=n_samples,
                           noise=noise,
                           eps=eps,
                           min_samples=min_samples,
                           n_clusters=n_clusters)

if __name__ == "__main__":
    app.run(debug=True)
