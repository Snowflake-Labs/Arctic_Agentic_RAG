# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import defaultdict

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score


class Clustering:
    """Utility class for clustering algorithms."""

    @staticmethod
    def choose_subset_hdb(
        embeddings: np.array, **kwargs
    ) -> tuple[list[int], dict[int, list[int]]]:
        min_cluster_size: int = kwargs.get("min_cluster_size", 2)
        min_samples: int = kwargs.get("min_samples", 2)

        num_embeddings = embeddings.shape[0]
        if num_embeddings < max(min_cluster_size, min_samples):
            return list(range(num_embeddings)), {i: [i] for i in range(num_embeddings)}

        scores = (embeddings * embeddings[:, None, :]).sum(axis=-1)
        cosine_distance = 1 - scores

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
        )
        labels = clusterer.fit_predict(cosine_distance)

        # Choose the representative for each cluster
        unique_labels = np.unique(labels)
        chosen = []
        for cluster_id in unique_labels:
            # Get indices of all points in the current cluster
            cluster_points = np.where(labels == cluster_id)[0]
            if cluster_id == -1:  # -1 corresponds to outliers as identified by hdbscan
                if len(unique_labels) == 1:
                    chosen.extend(cluster_points.tolist())
                continue

            # Find the point closest to the cluster's centroid
            cluster_vectors = embeddings[cluster_points]
            centroid = cluster_vectors.mean(axis=0).reshape(1, -1)  # Compute centroid
            closest, _ = pairwise_distances_argmin_min(centroid, cluster_vectors)
            representative_idx = cluster_points[closest[0]]
            chosen.append(representative_idx)

        chosen = sorted(chosen)
        clusters = defaultdict(list)
        for i, c in enumerate(labels):
            clusters[int(c)].append(i)

        return chosen, clusters

    @staticmethod
    def choose_subset_kmeans(
        embeddings: np.array,
    ) -> tuple[list[int], dict[int, list[int]]]:
        if embeddings.shape[0] == 1:
            return [0], {0: [0]}
        if embeddings.shape[0] == 2:
            return [0, 1], {0: [0], 1: [1]}

        # Perform K-means clustering for each k and calculate silhouette score
        silhouette_scores = []
        k_values = list(range(2, embeddings.shape[0]))
        kmeans_models = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            unique_labels = np.unique(labels)
            kmeans_models.append((kmeans, labels, unique_labels))
            # if any([np.where(labels == cluster_id)[0].shape[0] == 0 for cluster_id in range(k)]):
            if len(unique_labels) != k:
                score = -100
            else:
                score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)

        # Find the best k (with the highest silhouette score)
        idx_and_scores = [
            (b, a)
            for (a, b) in sorted(
                [(k, i) for i, k in enumerate(silhouette_scores)], reverse=True
            )
        ]
        if len(idx_and_scores) == 1:
            best_idx = idx_and_scores[0][0]
        else:
            idx1 = idx_and_scores[0][0]
            idx2 = idx_and_scores[1][0]
            best_idx = max(idx1, idx2)

        _, best_model, labels, unique_labels = (
            k_values[best_idx],
            *kmeans_models[best_idx],
        )
        final_centroids = best_model.cluster_centers_

        # Choose the representative for each cluster
        chosen = []
        clusters = {}
        for cluster_id in unique_labels:
            # Get indices of all points in the current cluster
            cluster_points = np.where(labels == cluster_id)[0]
            clusters[cluster_id.item()] = cluster_points.tolist()

            # Find the point closest to the cluster's centroid
            cluster_vectors = embeddings[cluster_points]
            centroid = final_centroids[cluster_id].reshape(1, -1)
            closest, _ = pairwise_distances_argmin_min(centroid, cluster_vectors)
            representative_idx = cluster_points[closest[0]]
            chosen.append(representative_idx.item())

        return sorted(chosen), clusters
