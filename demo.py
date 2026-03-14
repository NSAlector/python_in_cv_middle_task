from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from encoder import ImageDatasetEncoder
from plugin_loader import create_plugin, load_plugin_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo for encoder + fuzzy c-means + grid search.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Path to dataset directory.")
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="How many dataset samples to use in demo (0 means all).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iter", type=int, default=40, help="Max FCM iterations.")
    args = parser.parse_args()

    encoder = ImageDatasetEncoder(args.dataset, threshold=0.5, noise_percent=20.0, seed=args.seed)
    total_items = len(encoder)
    selected = total_items if args.limit == 0 else min(args.limit, total_items)

    encoded_as_int = [encoder[idx] for idx in range(selected)]
    encoded_as_bits = [list(encoder.get_bits(idx)) for idx in range(selected)]

    print(f"Loaded {selected} samples from: {args.dataset.resolve()}")
    print(f"Example encoded integer: {encoded_as_int[0]}")
    print(f"Bit-vector length: {len(encoded_as_bits[0])}")

    clusterizer_cls = load_plugin_class(
        "plugins.clusterizers.fuzzy_c_means",
        "FuzzyCMeansClusterizer",
    )
    optimizer = create_plugin("plugins.optimizers.grid_search:GridSearchHyperparameterOptimizer")

    best = optimizer.optimize(
        clusterizer_cls=clusterizer_cls,
        data=encoded_as_bits,
        param_ranges={
            "c": (2, 6),
            "m": (1.5, 2.5),
            "eps": (0.001, 0.005),
        },
        deltas={
            "c": 1,
            "m": 0.5,
            "eps": 0.002,
        },
        fixed_params={
            "max_iter": args.max_iter,
            "seed": args.seed,
        },
    )

    result = best["result"]
    cluster_sizes = Counter(result["labels"])

    print("\nBest hyperparameters:")
    for name, value in best["best_params"].items():
        print(f"  {name} = {value}")
    print(f"Best score (cluster spread): {best['best_score']:.6f}")
    print(f"Used clusters in result: {best['clusters_used']}")

    print("\nCluster sizes:")
    for cluster_idx in sorted(cluster_sizes):
        print(f"  cluster {cluster_idx}: {cluster_sizes[cluster_idx]}")

    print("\nCentroid preview (first 8 coordinates each):")
    for idx, centroid in enumerate(result["centroids"]):
        preview = ", ".join(f"{value:.3f}" for value in centroid[:8])
        print(f"  centroid {idx}: [{preview}]")


if __name__ == "__main__":
    main()
