import argparse
import os


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot morphology metrics for a selected project. Reads the metrics "
            "CSVs that morphology.py already extracted (mask_metrics.csv, "
            "light_dark_metrics.csv, well_metrics_summary.csv) -- run morphology.py "
            "first if those are missing or stale."
        )
    )
    parser.add_argument(
        "--remove_outliers",
        action="store_true",
        help=(
            "Remove outliers (per the project's outlier_detection settings in "
            "plotting_metadata.json) from the data feeding every morphology-metrics "
            "plot/statistic, before plotting. Does NOT affect "
            "plot_variability_and_outliers itself (which needs the full, un-stripped "
            "data to report on outliers in the first place), nor detection QC's "
            "considered/not-considered filter (a different, segmentation-quality "
            "concept, unrelated to statistical outliers)."
        ),
    )
    return parser.parse_args()


ARGS = _parse_args()

from utils_plot import *

# Everything plot-specific (which features/metrics to plot, which grouping
# variables to compare, which days to run statistical testing across
# conditions on, outlier-detection settings, batches_complete_info) is
# configured in plotting_metadata.json, not here -- see that file (and
# plotting_metadata.json in data_analysis/ for a fully-commented template)
# for what each key means.
#
# A project can keep more than one plotting_metadata*.json file for
# distinct comparisons it wants analyzed separately (e.g.
# 2026_04_30-Zeynep-CP's plotting_metadata_mtesr_vs_ctrl.json /
# plotting_metadata_e10_vs_e11.json) -- discover_plotting_metadata_files
# finds every one of them (skipping ad hoc "...copy..." backups) and the
# driver loop below runs the full pipeline once per file, each writing
# into its own plots_<suffix>/ directory via configure_plots_dirname so
# they never overwrite each other. A project with just the plain
# plotting_metadata.json (the common case) gets exactly one iteration,
# writing into "plots/" exactly as before this driver loop existed.


def run_pipeline(plotting_metadata, run_label):
    """Run the full morphology plotting/analysis pipeline once, using the
    given (already-normalized) plotting_metadata -- called once per
    plotting_metadata*.json variant discovered for this project (see the
    driver loop below)."""
    features = plotting_metadata["features"]
    features_names = plotting_metadata["feature_labels"]
    groups = plotting_metadata["groups"]
    days = plotting_metadata["days_for_group_comparison"]
    pca_umap_days = plotting_metadata["pca_umap_days"]
    plot_aggregation_levels = plotting_metadata["plot_aggregation_levels"]
    well_metric_plots = plotting_metadata["well_metric_plots"]
    pca_correlation_threshold = plotting_metadata["pca_correlation_threshold"]
    growth_rate_features = plotting_metadata["growth_rate_features"]
    developmental_windows = plotting_metadata["developmental_windows"]
    window_instant_metrics = plotting_metadata["window_instant_metrics"]
    window_expansion_rate_features = plotting_metadata["window_expansion_rate_features"]
    line_plot_error_bar = plotting_metadata["line_plot_error_bar"]
    umap_min_cluster_size = plotting_metadata["umap_min_cluster_size"]
    umap_cluster_selection_method = plotting_metadata["umap_cluster_selection_method"]
    umap_n_neighbors = plotting_metadata["umap_n_neighbors"]
    umap_min_dist = plotting_metadata["umap_min_dist"]
    umap_random_state = plotting_metadata["umap_random_state"]
    transition_days = plotting_metadata["transition_days"]
    min_wells_for_stats = plotting_metadata["min_wells_for_stats"]
    batches_complete_info = plotting_metadata["batches_complete_info"]

    outlier_settings = plotting_metadata["outlier_detection"]
    outlier_method = outlier_settings["method"]
    iqr_multiplier = outlier_settings["iqr_multiplier"]
    # For manual mean +/- N SD thresholding, set outlier_method = "mean_sd"
    # and change std_multiplier or provide manual_std_thresholds.
    std_multiplier = outlier_settings["std_multiplier"]
    manual_std_thresholds = outlier_settings["manual_std_thresholds"]
    outlier_aggregation_level = outlier_settings["aggregation_level"]

    if ARGS.remove_outliers:
        print(
            f"🧹 --remove_outliers enabled: stripping {outlier_method}-based outliers "
            f"(across {features}) from every morphology-metrics plot's data before "
            "plotting/statistics. See plots/quality_control/outliers/ (from "
            "plot_variability_and_outliers, always computed on the full data) for what "
            "gets removed."
        )
    configure_outlier_removal(
        enabled=ARGS.remove_outliers,
        features=features,
        method=outlier_method,
        iqr_multiplier=iqr_multiplier,
        std_multiplier=std_multiplier,
    )

    # plot_feature_grid_scatters expects batches_complete_info as a
    # {batch_id: {...}} dict (every other plotting function here expects the
    # list form) -- a pre-existing inconsistency in utils_plot.py, not
    # something introduced by the plotting_metadata.json split. The dict must
    # carry every batch key (cell-line/condition/isogenic_pair/etc.), not just
    # "days" -- plot_feature_grid_scatters resolves its groups via
    # info.get(group_by, batch_id), so a dict missing the actual group_by key
    # silently fell back to per-batch grouping for any group_by other than
    # "batch" (confirmed via a real end-to-end run against PCH2_CB with
    # group_by="condition": groups printed as "QB09_1"/"QB10_1" instead of
    # "PCH"/"CTRL"). "days" itself is never read from this dict --
    # plot_feature_grid_scatters iterates over its own `days` parameter instead.
    plates_info_dict = {
        batch["batch"]: batch
        for batch in batches_complete_info
    }

    # Feature correlation structure and per-well growth kinetics characterize
    # the morphology feature space / the timecourse itself rather than a
    # between-group comparison, so these run once per project instead of once
    # per grouping variable in the loop below.
    plot_feature_correlation_heatmap(base_dir, project_dir, batches_complete_info, features, features_names)

    for growth_feature in growth_rate_features:
        analyze_growth_trajectories(
            base_dir, project_dir, batches_complete_info,
            feature=growth_feature["feature"],
            metric_label=growth_feature.get("metric_label", growth_feature["feature"]),
            group_by=groups[0] if groups else "batch",
            transition_days=transition_days,
            min_wells_for_stats=min_wells_for_stats,
        )

    # UMAP+HDBSCAN clustering is unsupervised -- the embedding/clustering itself
    # doesn't depend on group_by, only the cluster-vs-group contingency table
    # does -- so, like the two analyses above, this runs once per project
    # (using the first configured group for that table) rather than once per
    # grouping variable in the loop below.
    plot_umap_clusters(
        base_dir, project_dir, days=pca_umap_days, batches_complete_info=batches_complete_info,
        group_by=groups[0] if groups else "batch", segregation_groups=groups,
        features=features, correlation_threshold=pca_correlation_threshold,
        min_cluster_size=umap_min_cluster_size, cluster_selection_method=umap_cluster_selection_method,
        n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=umap_random_state,
    )

    # Well-level dataset + outlier-exclusion audit -- project-wide (not tied
    # to any one grouping variable), so these run once here rather than
    # once per group in the loop below.
    export_well_level_summary(base_dir, project_dir, batches_complete_info, features)
    audit_outlier_exclusion_by_well(
        base_dir, project_dir, batches_complete_info, features,
        group_by=groups[0] if groups else "batch",
    )

    for group in groups:
        for aggregation_level in plot_aggregation_levels:
            plot_feature_over_time(
                base_dir,
                project_dir,
                batches_complete_info,
                features,
                features_names,
                group_by=group,
                aggregation_level=aggregation_level,
                error_bar=line_plot_error_bar,
                transition_days=transition_days,
                min_wells_for_stats=min_wells_for_stats,
            )
        plot_pca_grid_by_group(
            base_dir, project_dir, days=pca_umap_days, batches_complete_info=batches_complete_info,
            group_by=group, features=features, correlation_threshold=pca_correlation_threshold,
        )

        plot_feature_grid_scatters(
            base_dir, project_dir, days=days, batches_complete_info=plates_info_dict,
            features=features, metric_names=features_names, group_by=group,
            min_wells_for_stats=min_wells_for_stats,
        )
        plot_well_organoid_counts_over_time(
            base_dir, project_dir, batches_complete_info, group_by=group,
            error_bar=line_plot_error_bar, transition_days=transition_days,
        )

        # Per-well metrics (e.g. Total_Segmented_Area_um2 -- robust to touching
        # organoids merged into one connected component) -- see
        # plotting_metadata.json's "well_metric_plots" for the full list.
        for well_metric in well_metric_plots:
            plot_well_metric_over_time(
                base_dir, project_dir, batches_complete_info,
                metric_column=well_metric["metric_column"],
                metric_label=well_metric["metric_label"],
                csv_filename=well_metric["csv_filename"],
                group_by=group,
                error_bar=line_plot_error_bar,
                transition_days=transition_days,
                min_wells_for_stats=min_wells_for_stats,
            )
            plot_well_metric_boxplots_by_group(
                base_dir, project_dir, days, batches_complete_info,
                metric_column=well_metric["metric_column"],
                metric_label=well_metric["metric_label"],
                csv_filename=well_metric["csv_filename"],
                group_by=group,
                min_wells_for_stats=min_wells_for_stats,
            )

        plot_variability_and_outliers(
            base_dir,
            project_dir,
            batches_complete_info,
            features=features,
            metric_names=features_names,
            group_by=group,
            outlier_method=outlier_method,
            iqr_multiplier=iqr_multiplier,
            std_multiplier=std_multiplier,
            manual_std_thresholds=manual_std_thresholds,
            aggregation_level=outlier_aggregation_level,
            transition_days=transition_days,
        )

        detection_summary, out_path = create_detection_analysis(base_dir, project_dir, batches_complete_info, batches_info, group_by=group)
        plot_undetected_summary(detection_summary, out_path)

        # Per-object instant metrics (e.g. Light_Fraction/Dark_Fraction from
        # light_dark_metrics.csv, needs track_light_dark_split: true in
        # segmentation_metadata.json) and expansion rate, longitudinally
        # across developmental "windows" and compared between groups within
        # each -- see plotting_metadata.json's "developmental_windows"/
        # "window_instant_metrics"/"window_expansion_rate_features". Skipped
        # entirely (default) for projects that haven't configured windows.
        for window_metric in window_instant_metrics:
            plot_feature_over_time_by_window(
                base_dir, project_dir, batches_complete_info, developmental_windows,
                feature=window_metric["feature"],
                metric_label=window_metric.get("metric_label", window_metric["feature"]),
                group_by=group,
                csv_filename=window_metric.get("csv_filename", "mask_metrics.csv"),
                error_bar=line_plot_error_bar,
                transition_days=transition_days,
            )
            plot_feature_boxplots_by_window(
                base_dir, project_dir, batches_complete_info, developmental_windows,
                feature=window_metric["feature"],
                metric_label=window_metric.get("metric_label", window_metric["feature"]),
                group_by=group,
                csv_filename=window_metric.get("csv_filename", "mask_metrics.csv"),
                min_wells_for_stats=min_wells_for_stats,
            )

        for rate_feature in window_expansion_rate_features:
            analyze_expansion_rate_by_window(
                base_dir, project_dir, batches_complete_info, developmental_windows,
                feature=rate_feature["feature"],
                metric_label=rate_feature.get("metric_label", rate_feature["feature"]),
                group_by=group,
                error_bar=line_plot_error_bar,
                min_wells_for_stats=min_wells_for_stats,
            )

    print(f"✅ All processing and plotting completed ({run_label}).")


for suffix, path in discover_plotting_metadata_files(project_dir):
    variant_metadata = load_plotting_metadata_file(path)
    plots_dirname = f"plots_{suffix}" if suffix else "plots"
    configure_plots_dirname(plots_dirname)
    run_label = f"{os.path.basename(path)} -> {plots_dirname}/"
    print(f"\n{'=' * 70}\n▶ Running plotting pipeline using {run_label}\n{'=' * 70}\n")
    run_pipeline(variant_metadata, run_label)
    configure_plots_dirname(None)
