import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import cycle
from statannotations.Annotator import Annotator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import chi2_contingency, ttest_ind
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests
import umap
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines

from experiment_info import *

plt.rcParams.update({
    "axes.titlesize": 12,           # Title font size
    "axes.labelsize": 12,           # X and Y label font size
    "xtick.labelsize": 10,          # X-axis tick labels font size
    "ytick.labelsize": 10,          # Y-axis tick labels font size
    "legend.fontsize": 10,          # Legend font size
    "axes.spines.top": False,       # Remove top spine
    "axes.spines.right": False,      # Remove right spine
    "axes.spines.bottom": False,       # Remove top spine
    "axes.spines.left": False      # Remove right spine
})

# Output directory name every plotting function writes into (under
# base_dir/project_dir/), set once per run via configure_plots_dirname.
# "plots" by default -- unchanged for every project with a single
# plotting_metadata.json. plot.py's driver loop overrides this to
# "plots_<suffix>" once per discovered plotting_metadata_<suffix>.json
# variant (see experiment_info.py's discover_plotting_metadata_files), so
# e.g. 2026_04_30-Zeynep-CP's two comparisons land in
# plots_mtesr_vs_ctrl/ and plots_e10_vs_e11/ instead of overwriting a
# single shared plots/. Defined here, before this module's own first use
# of it below, rather than alongside _OUTLIER_REMOVAL further down.
_PLOTS_DIRNAME = {"name": "plots"}


def configure_plots_dirname(name=None):
    """Set the output directory name for every subsequent plotting call in
    this process. None (or "plots") resets to the default -- call this
    between variants in plot.py's driver loop so one variant's
    override never leaks into the next."""
    _PLOTS_DIRNAME["name"] = name or "plots"


def _plots_root(base_dir, project_dir):
    """base_dir/project_dir/<configured plots dirname> -- the one place
    every plotting function should build its own plot_dir from, instead of
    hardcoding the 'plots' literal directly (see _PLOTS_DIRNAME above)."""
    return os.path.join(base_dir, project_dir, _PLOTS_DIRNAME["name"])


plot_dir = _plots_root(base_dir, project_dir)
os.makedirs(plot_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Standard figure sizing, calibrated to
# AnnaRaic/Area_um2_expansion_rate_boxplots_by_window.png's actual rendered
# size (~6.2in wide x ~3.7in tall, a 2-window/one-row figure with a bottom
# legend). STANDARD_FIG_WIDTH_IN is that figure's total width, shared
# across up to MAX_PANEL_COLS panels in a row rather than multiplied per
# column -- that's what "grid scatterplots have a maximum of two subplots
# side by side" means concretely: two panels split one standard figure's
# width between them, exactly as in the calibration figure. Additional
# panels beyond MAX_PANEL_COLS wrap into more rows instead of growing
# width. STANDARD_FIG_HEIGHT_IN is the axes-only height for ONE row of
# panels; a figure's legend (drawn below the axes, not inline) adds
# LEGEND_RESERVE_IN on top of that separately at each call site -- the
# calibration figure's own ~3.7in total already IS one row (2.5in) plus
# its own ~1.2in legend margin, not the "2.5" restated as one lump sum.
# Single-panel figures with an inline legend (growth trajectories, CV) use
# STANDARD_FIG_HEIGHT_IN/STANDARD_FIG_WIDTH_IN directly. Growth rate by
# group is the one single-panel exception: it uses a below-figure legend
# (LEGEND_RESERVE_IN, like the multi-panel figures) and, per explicit
# instruction, only half of STANDARD_FIG_WIDTH_IN -- matching one
# plot_feature_grid_scatters column's width rather than the full standard
# width the multi-day/window figures share.
STANDARD_FIG_WIDTH_IN = 5
STANDARD_FIG_HEIGHT_IN = 3
MAX_PANEL_COLS = 3
stats_fontsize = 6

def _panel_grid_dims(n_panels):
    """(n_cols, n_rows) for laying out ``n_panels`` subplots within one
    standard figure's total width -- at most MAX_PANEL_COLS columns
    sharing STANDARD_FIG_WIDTH_IN between them, wrapping into additional
    rows (each STANDARD_FIG_HEIGHT_IN tall) rather than adding columns, so
    the figure never exceeds MAX_PANEL_COLS panels per row regardless of
    how many panels (days, windows, ...) there are."""
    n_cols = min(MAX_PANEL_COLS, max(n_panels, 1))
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_cols, n_rows


# Human-readable names for the raw feature columns, used wherever a feature
# name would otherwise appear verbatim on a figure (e.g. the UMAP cluster
# effect-size heatmap's x-axis) -- falls back to the raw name for any
# feature not listed here rather than raising, so new features don't need
# an entry before they can be plotted.
FEATURE_DISPLAY_NAMES = {
    "Area_um2": "Area",
    "Diameter_um": "Diameter",
    "Perimeter_um": "Perimeter",
    "Feret_Diameter_Max_um": "Feret Diameter",
    "Aspect_Ratio": "Aspect Ratio",
    "Circularity": "Circularity",
    "Roundness": "Roundness",
    "Solidity": "Solidity",
    "Eccentricity": "Eccentricity",
    "Skeleton_Length_um": "Skeleton Length",
}


def _display_feature_name(feature):
    return FEATURE_DISPLAY_NAMES.get(feature, feature)


# Area_um2 is plotted in mm^2 wherever its raw value appears on a figure's
# axis (growth trajectories, grid scatterplot) -- growth-rate/expansion-rate
# figures are exempt since their plotted value is an exponential-fit slope
# (log(Area_um2) ~ Day), which is mathematically unchanged by rescaling the
# underlying feature by a constant factor.
AREA_UM2_TO_MM2 = 1e-6


def _um2_label_to_mm2(label):
    return label.replace("um²", "mm²").replace("um^2", "mm^2").replace("um2", "mm2")


# Global outlier-removal state, set once per run via configure_outlier_removal
# (plot.py, from --remove_outliers). Off by default -- every existing
# caller of filter_by_diameter (nearly every morphology-metrics plot; see
# that function's docstring) is unaffected unless explicitly enabled.
_OUTLIER_REMOVAL = {
    "enabled": False,
    "features": [],
    "method": "iqr",
    "iqr_multiplier": 1.5,
    "std_multiplier": 3.0,
    "percentile_bounds": (0.05, 0.95),
}


def configure_outlier_removal(
    enabled, features=None, method="iqr", iqr_multiplier=1.5,
    std_multiplier=3.0, percentile_bounds=(0.05, 0.95),
):
    """Enable/configure removing outliers from every morphology-metrics
    plot's underlying data (via filter_by_diameter), using the exact same
    calculate_outlier_thresholds method plot_variability_and_outliers
    already uses to *report* outliers -- called once from plot.py,
    driven by --remove_outliers, with the project's own outlier_detection
    settings so removal and reporting stay consistent with each other.
    """
    _OUTLIER_REMOVAL.update({
        "enabled": bool(enabled),
        "features": list(features) if features else [],
        "method": method,
        "iqr_multiplier": iqr_multiplier,
        "std_multiplier": std_multiplier,
        "percentile_bounds": percentile_bounds,
    })


def _remove_outlier_rows(df):
    """Drop rows that are an outlier (per _OUTLIER_REMOVAL's configured
    method/multiplier) in ANY of its configured features.

    Thresholds are computed fresh on whatever ``df`` was passed in -- in
    practice, since filter_by_diameter is normally called on one batch's
    one day at a time, that means "unusual relative to this same batch on
    this same day", not pooled across batches/groups (a different, looser
    granularity than plot_variability_and_outliers's own per-Group-per-Day
    report, which pools across every batch in a group). NaN values in a
    feature are never treated as outliers (missing, not extreme).
    """
    settings = _OUTLIER_REMOVAL
    features = [f for f in settings["features"] if f in df.columns]
    if not features or df.empty:
        return df

    keep_mask = pd.Series(True, index=df.index)
    for feature in features:
        thresholds = calculate_outlier_thresholds(
            df[feature], method=settings["method"],
            iqr_multiplier=settings["iqr_multiplier"],
            std_multiplier=settings["std_multiplier"],
            percentile_bounds=settings["percentile_bounds"],
            log_transform=feature in SIZE_LIKE_FEATURES,
        )
        lower, upper = thresholds["lower_threshold"], thresholds["upper_threshold"]
        if pd.isna(lower) or pd.isna(upper):
            continue
        values = pd.to_numeric(df[feature], errors='coerce')
        keep_mask &= values.between(lower, upper) | values.isna()

    return df[keep_mask]


def filter_by_diameter(df, day):
    """No size/diameter plausibility filtering -- every segmented object's
    own metrics are kept exactly as originally extracted. This used to
    apply a day-dependent min/max diameter plausibility gate; that's been
    removed entirely (per explicit instruction) since silently dropping
    organoids in a way that isn't visible in the output data is worse than
    keeping the full, originally-segmented population and letting
    --remove_outliers (IQR or another configured method, optionally
    log-scale for size-like features -- see calculate_outlier_thresholds)
    exclude statistical outliers downstream, on demand, when actually
    wanted. ``day`` is kept as a parameter purely for call-site
    compatibility -- every existing caller still passes it, even though
    this function no longer uses it itself.
    """
    if 'Diameter_um' not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    if _OUTLIER_REMOVAL["enabled"]:
        df = _remove_outlier_rows(df)

    return df


def _image_id_from_metric_id(organoid_id):
    return str(organoid_id).split("_object", 1)[0]


def _ensure_metric_identity_columns(df):
    """Keep legacy metrics readable while exposing image/well/object identities."""
    df = df.copy()

    if 'Organoid_ID' not in df.columns:
        df['Organoid_ID'] = [f"object{idx:04d}" for idx in range(1, len(df) + 1)]

    if 'Image_ID' not in df.columns:
        df['Image_ID'] = df['Organoid_ID'].map(_image_id_from_metric_id)

    if 'Well_ID' not in df.columns:
        df['Well_ID'] = df['Image_ID']

    if 'Object_Label' not in df.columns:
        object_label = df['Organoid_ID'].astype(str).str.extract(r"_object(\d+)$")[0]
        df['Object_Label'] = pd.to_numeric(object_label, errors='coerce').fillna(1).astype(int)

    if 'Object_Count_In_Image' not in df.columns:
        df['Object_Count_In_Image'] = df.groupby('Image_ID')['Organoid_ID'].transform('count')

    return df


def prepare_morphology_records(df, day, features, aggregation_level="object"):
    """
    Return object-level rows or one row per image/well.

    Object level preserves individual organoids. Well/image level converts each
    image into a per-well mean for each requested feature and keeps the number of
    detected organoids in ``Object_Count_In_Image``.
    """
    features = [features] if isinstance(features, str) else list(features)
    df = _ensure_metric_identity_columns(df)
    df = filter_by_diameter(df, day).copy()
    if df.empty:
        return df

    aggregation_level = str(aggregation_level).lower()
    if aggregation_level in {"object", "organoid", "individual"}:
        df['Aggregation_Level'] = "object"
        return df

    if aggregation_level not in {"well", "image", "well_mean", "image_mean"}:
        raise ValueError("aggregation_level must be 'object' or 'well'.")

    group_cols = [
        column for column in [
            'Batch', 'Day', 'Run', 'Image_ID', 'Well_ID',
            'Plate_Index', 'Well_Index', 'Organoid_Mode'
        ]
        if column in df.columns
    ]
    if not group_cols:
        group_cols = ['Image_ID']

    available_features = [feature for feature in features if feature in df.columns]
    if not available_features:
        return pd.DataFrame()

    well_df = df.groupby(group_cols, dropna=False)[available_features].mean().reset_index()
    counts = (
        df
        .groupby(group_cols, dropna=False)
        .agg(Object_Count_In_Image=('Organoid_ID', 'count'))
        .reset_index()
    )
    well_df = well_df.merge(counts, on=group_cols, how='left')
    well_df['Organoid_ID'] = well_df['Image_ID'] if 'Image_ID' in well_df.columns else well_df.index.astype(str)
    well_df['Aggregation_Level'] = "well"

    return well_df


def _hedges_g(vals1, vals2):
    """Bias-corrected standardized mean difference (Hedges' g) between two
    independent samples -- reported alongside every Welch's t-test p-value
    in this module, since significance alone doesn't convey effect
    magnitude. Uses the pooled-variance convention (as in Cohen's d) with
    the small-sample correction factor J = 1 - 3/(4*df - 1)
    (Hedges, 1981; Borenstein et al. 2009), df = n1 + n2 - 2. Positive means
    group 1 > group 2. Returns NaN if either sample has fewer than 2 values
    or the pooled SD is zero.
    """
    vals1 = pd.to_numeric(pd.Series(vals1), errors='coerce').dropna()
    vals2 = pd.to_numeric(pd.Series(vals2), errors='coerce').dropna()
    n1, n2 = len(vals1), len(vals2)
    if n1 < 2 or n2 < 2:
        return float('nan')

    var1, var2 = vals1.var(ddof=1), vals2.var(ddof=1)
    df = n1 + n2 - 2
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / df)
    if pooled_sd == 0 or pd.isna(pooled_sd):
        return float('nan')

    cohen_d = (vals1.mean() - vals2.mean()) / pooled_sd
    correction = 1 - 3 / (4 * df - 1)
    return cohen_d * correction


def _stat_annotation_text(s):
    """Compact 'g=<Hedges' g> <stars>' label for a pairwise-comparison
    bracket annotation. Effect size is shown directly on every group-
    comparison figure (grid scatterplots, box/strip-plot grids) rather
    than only in each figure's companion stats CSV, since the manuscript
    build only carries forward outlier/detection tables and expects the
    effect sizes behind a figure's significance stars to be readable from
    the figure itself."""
    g = s.get('hedges_g', float('nan'))
    g_text = f"g={g:.2f}" if pd.notna(g) else "g=n/a"
    return f"{g_text} {s['stars']}"


def _resolve_error_bar_values(summary, error_bar="sem"):
    """Return the y-error values for a mean+/-error line plot, given a
    groupby summary that already has 'std' and 'count' columns.

    ``error_bar`` (set via plotting_metadata.json's "line_plot_error_bar",
    applied consistently across every mean+/-error line plot in this
    module) is one of:
      - "sem" (default): standard error of the mean, std / sqrt(n).
      - "sd": standard deviation.
      - "ci": a parametric 95% confidence interval, t_crit * sem, using
        each group/day's own sample size for the t critical value (NaN
        where n < 2, since a CI is undefined for a single observation).
    """
    error_bar = str(error_bar).lower()
    sem = summary['std'] / np.sqrt(summary['count'])

    if error_bar == "sd":
        return summary['std']

    if error_bar == "ci":
        t_crit = summary['count'].apply(
            lambda n: t_dist.ppf(0.975, n - 1) if n > 1 else np.nan
        )
        return sem * t_crit

    return sem


def _annotate_transitions(ax, transition_days):
    """Draw a vertical marker + rotated label on ``ax`` for each {day: label}
    in ``transition_days`` (plotting_metadata.json's "transition_days") --
    e.g. a day where the imaging/plating regime itself changed (96-well/
    single-organoid -> 6-well/multiple-organoids-per-well), so a jump or
    shift in a longitudinal metric at that day isn't left unexplained. A
    no-op for the empty dict every project gets by default. Call this
    AFTER all data-driven artists are on the axes (so ylim reflects the
    real data, not just the marker) and before saving."""
    if not transition_days:
        return

    # Read the current (already data-driven) y-limit directly -- do NOT
    # call ax.set_autoscale_on(False) here: before the axes has ever been
    # drawn, get_ylim()/get_xlim() can still report matplotlib's raw
    # (0, 1) default rather than the real autoscaled range, and freezing
    # autoscale at that point stuck later artists (this exact bug blew a
    # plot_well_metric_over_time figure up to 100,000+ px wide once
    # transition_days actually had an entry to draw). axvline/text don't
    # perturb autoscaling on their own, so nothing here needs it disabled.
    y_top = ax.get_ylim()[1]
    for day, label in sorted(transition_days.items()):
        ax.axvline(day, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=0)
        ax.text(
            day, y_top, f" {label}", rotation=90, ha='left', va='top',
            fontsize=7, color='gray', alpha=0.8,
        )


def plot_feature_over_time(base_dir, project_dir, batches_complete_info, features, metric_names, group_by="batch", aggregation_level="object", error_bar="sem", transition_days=None, min_wells_for_stats=2):
    """
    Plots morphological features over time (mean +/- error_bar -- "sem" by
    default, or "sd"/"ci"; see _resolve_error_bar_values -- set via
    plotting_metadata.json's "line_plot_error_bar").
    Performs Welch's T-test with FDR (Benjamini-Hochberg) correction across tested days.
    """
    if isinstance(features, str):
        features = [features]

    aggregation_level = str(aggregation_level).lower()
    if aggregation_level in {"organoid", "individual"}:
        aggregation_level = "object"
    if aggregation_level in {"image", "well_mean", "image_mean"}:
        aggregation_level = "well"

    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    for i, feature in enumerate(features):
        all_records = []
        stats_records = []
        metric_label = metric_names[i] if i < len(metric_names) else feature

        # --- 1. DATA COLLECTION ---
        for entry in batches_complete_info:
            batch, label = entry['batch'], entry.get(group_by)
            for day in entry['days']:
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df_filtered = prepare_morphology_records(
                        df,
                        day,
                        [feature],
                        aggregation_level=aggregation_level,
                    )

                    if feature in df_filtered.columns and not df_filtered.empty:
                        df_filtered['Batch'] = batch
                        df_filtered['Group'] = label
                        df_filtered['Day'] = day
                        if aggregation_level == "object":
                            df_filtered['Analysis_Unit_ID'] = df_filtered['Organoid_ID'].astype(str)
                        else:
                            df_filtered['Analysis_Unit_ID'] = df_filtered['Image_ID'].astype(str)

                        keep_columns = [
                            'Batch', 'Group', 'Day', 'Image_ID', 'Well_ID',
                            'Organoid_ID', 'Analysis_Unit_ID',
                            'Object_Count_In_Image', 'Aggregation_Level', feature
                        ]
                        keep_columns = [column for column in keep_columns if column in df_filtered.columns]
                        all_records.append(df_filtered[keep_columns])

                    # Statistical testing always runs on well-level means,
                    # independent of whatever aggregation_level was requested
                    # for the visualization. This avoids pseudoreplication in
                    # organoid_mode == "multiple" data, where many
                    # non-independent organoids from the same well would
                    # otherwise each count as a separate sample in the
                    # t-test, inflating apparent significance. For
                    # organoid_mode == "single" this is a no-op: each well
                    # already has exactly one organoid, so its well-level
                    # mean equals the object-level value.
                    df_well = prepare_morphology_records(df, day, [feature], aggregation_level="well")
                    if feature in df_well.columns and not df_well.empty:
                        df_well = df_well.copy()
                        df_well['Group'] = label
                        df_well['Day'] = day
                        stats_records.append(df_well[['Group', 'Day', feature]])

        if not all_records:
            continue

        df_all = pd.concat(all_records, ignore_index=True)
        df_stats_all = pd.concat(stats_records, ignore_index=True) if stats_records else pd.DataFrame()
        unique_groups = sorted(df_all['Group'].unique())
        df_all.to_csv(
            os.path.join(plot_data_dir, f"{feature}_{aggregation_level}_{group_by}_over_time_records.csv"),
            index=False,
        )
        
        # --- 2. SUMMARY STATISTICS ---
        # Two summaries, deliberately kept separate: `visualization_summary`
        # at whatever aggregation_level was requested (purely descriptive --
        # exported under its original filename, unchanged), and `summary`
        # -- ALWAYS well-level, from df_stats_all -- which is what actually
        # drives the plotted mean+/-error-bar line, its N, and the
        # significance annotation's position. Before this fix, the plotted
        # line/N came from `visualization_summary` even when
        # aggregation_level == "object", so a project with many organoids
        # per well (Organoid_Mode == "multiple") got a deceptively tight
        # error band (N inflated by pooling every organoid as if
        # independent) sitting right next to a significance star honestly
        # tested on well-level N -- the two numbers on the same figure
        # didn't agree. This mirrors the identical decoupling the p-value
        # itself already used (see the data-collection comment above) and
        # is also the statistically correct estimate of "the group's mean"
        # when wells hold very different organoid counts -- a mean of
        # well-means gives every well equal weight, rather than letting a
        # 13-organoid well outweigh a 9-organoid one in a pooled average.
        visualization_summary = df_all.groupby(['Group', 'Day'])[feature].agg(['mean', 'std', 'count']).reset_index()
        visualization_summary['sem'] = visualization_summary['std'] / np.sqrt(visualization_summary['count'])
        visualization_summary['error_value'] = _resolve_error_bar_values(visualization_summary, error_bar)
        visualization_summary['Aggregation_Level'] = aggregation_level
        count_column = 'N_Organoids' if aggregation_level == "object" else 'N_Wells'
        visualization_summary.rename(columns={'count': count_column}, inplace=True)
        visualization_summary.to_csv(
            os.path.join(plot_data_dir, f"{feature}_{aggregation_level}_{group_by}_over_time_summary.csv"),
            index=False,
        )

        if not df_stats_all.empty:
            summary = df_stats_all.groupby(['Group', 'Day'])[feature].agg(['mean', 'std', 'count']).reset_index()
        else:
            # No resolvable well-level data at all (shouldn't normally
            # happen when all_records is non-empty) -- fall back to the
            # visualization-level summary rather than plotting nothing.
            summary = visualization_summary.rename(columns={count_column: 'count'})
        summary['sem'] = summary['std'] / np.sqrt(summary['count'])
        summary['error_value'] = _resolve_error_bar_values(summary, error_bar)
        summary.rename(columns={'count': 'N_Wells'}, inplace=True)
        summary.to_csv(
            os.path.join(plot_data_dir, f"{feature}_{aggregation_level}_{group_by}_over_time_well_summary.csv"),
            index=False,
        )
        print(
            f"{feature} ({aggregation_level}): plotted {len(df_all)} records "
            f"({visualization_summary[count_column].sum()} {count_column}, "
            f"{summary['N_Wells'].sum()} wells); summary saved in {plot_data_dir}"
        )
        
        # --- 3. STATISTICAL TESTING WITH FDR CORRECTION ---
        # Tested on df_stats_all (well-level means), never on df_all
        # (whatever aggregation_level is being visualized) -- see the
        # data-collection comment above.
        stats_data = []
        stats_rows = []
        skipped_rows = []
        if len(unique_groups) == 2 and not df_stats_all.empty:
            g1, g2 = unique_groups
            all_days = sorted(df_all['Day'].unique())
            # Optionally test all days or just the last 3. Testing all days is better for FDR context.
            days_to_test = all_days[-3:]

            raw_pvals = []
            effect_sizes = []
            sample_sizes = []
            valid_days = []

            for day in days_to_test:
                vals_g1 = df_stats_all[(df_stats_all['Day'] == day) & (df_stats_all['Group'] == g1)][feature].dropna()
                vals_g2 = df_stats_all[(df_stats_all['Day'] == day) & (df_stats_all['Group'] == g2)][feature].dropna()
                n1, n2 = len(vals_g1), len(vals_g2)

                if n1 >= min_wells_for_stats and n2 >= min_wells_for_stats:
                    _, p_val = ttest_ind(vals_g1, vals_g2, equal_var=False)
                    raw_pvals.append(p_val)
                    effect_sizes.append(_hedges_g(vals_g1, vals_g2))
                    sample_sizes.append((n1, n2))
                    valid_days.append(day)
                elif n1 >= 1 and n2 >= 1:
                    print(
                        f"Skipped {g1} vs {g2} on Day {day}: insufficient wells "
                        f"(n={n1}, {n2}) below min_wells_for_stats={min_wells_for_stats}."
                    )
                    skipped_rows.append({
                        'Day': day, 'Group_1': g1, 'Group_2': g2,
                        'N_Wells_1': n1, 'N_Wells_2': n2,
                        'p_raw': np.nan, 'p_adj': np.nan, 'hedges_g': np.nan,
                        'significance': 'skipped', 'Skipped_Insufficient_N': True,
                    })

            if raw_pvals:
                # Apply Benjamini-Hochberg Correction
                reject, p_corrected, _, _ = multipletests(raw_pvals, method='fdr_bh')

                print(f"\n--- FDR Corrected Statistics for {metric_label} ({g1} vs {g2}), tested on well-level means (n = independent wells) ---")
                for d_idx, day in enumerate(valid_days):
                    p_adj = p_corrected[d_idx]
                    sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"

                    # Position logic
                    day_sum = summary[summary['Day'] == day]
                    y_pos = (day_sum['mean'] + day_sum['error_value']).max()
                    n1, n2 = sample_sizes[d_idx]

                    stats_data.append({
                        'Day': day, 'label': sig, 'y_pos': y_pos,
                        'hedges_g': effect_sizes[d_idx],
                    })
                    stats_rows.append({
                        'Day': day, 'Group_1': g1, 'Group_2': g2,
                        'N_Wells_1': n1, 'N_Wells_2': n2,
                        'p_raw': raw_pvals[d_idx], 'p_adj': p_adj,
                        'hedges_g': effect_sizes[d_idx], 'significance': sig,
                        'Skipped_Insufficient_N': False,
                    })
                    print(
                        f"Day {day}: p_raw = {raw_pvals[d_idx]:.5f}, p_adj = {p_adj:.5f} ({sig}), "
                        f"Hedges' g = {effect_sizes[d_idx]:.3f} (n={n1} vs n={n2} wells)"
                    )

        stats_rows = stats_rows + skipped_rows
        if stats_rows:
            pd.DataFrame(stats_rows).to_csv(
                os.path.join(plot_data_dir, f"{feature}_{aggregation_level}_{group_by}_over_time_stats.csv"),
                index=False,
            )

        # --- 4. PLOTTING ---
        palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

        def scatter_measurement_points(ax):
            rng = np.random.default_rng(7)
            group_offsets = dict(zip(unique_groups, np.linspace(-0.18, 0.18, max(len(unique_groups), 1))))
            point_size = 8 if aggregation_level == "object" else 22
            alpha = 0.12 if aggregation_level == "object" else 0.40
            jitter_width = 0.035 if aggregation_level == "object" else 0.015

            for group in unique_groups:
                group_data = df_all[df_all['Group'] == group]
                for day, day_data in group_data.groupby('Day'):
                    y_values = pd.to_numeric(day_data[feature], errors='coerce').dropna()
                    if y_values.empty:
                        continue
                    x_values = (
                        np.full(len(y_values), float(day))
                        + group_offsets[group]
                        + rng.normal(0, jitter_width, len(y_values))
                    )
                    ax.scatter(
                        x_values,
                        y_values,
                        s=point_size,
                        alpha=alpha,
                        color=palette[group],
                        linewidths=0,
                        zorder=1,
                    )

        def add_stats_and_save(ax, filename):
            for res in stats_data:
                #if res['label'] != "ns": # Only plot if significant after correction
                text = _stat_annotation_text({'stars': res['label'], 'hedges_g': res['hedges_g']})
                ax.text(res['Day'], res['y_pos'] * 1.05, text,
                            ha='center', va='bottom', color='black', fontsize=8, fontweight='bold')
            ax.set_xlabel('Day')
            ax.set_ylabel(f"{metric_label}")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False, loc='upper left')
            _annotate_transitions(ax, transition_days)
            plt.tight_layout()
            save_path = os.path.join(plot_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        # Plot mean +/- error_bar (default "sem"; see _resolve_error_bar_values)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        scatter_measurement_points(ax)
        for group in unique_groups:
            data = summary[summary['Group'] == group]
            ax.errorbar(data['Day'], data['mean'], yerr=data['error_value'],
                        label=group, marker='o', capsize=5, color=palette[group], linewidth=2, zorder=3)

        suffix = str(error_bar).lower()
        add_stats_and_save(ax, f"{feature}_{aggregation_level}_{group_by}_over_time_{suffix}.png")
            

def _select_non_redundant_features(df, candidate_features, correlation_threshold=0.9, method="spearman"):
    """
    Greedily select a non-redundant feature subset from ``candidate_features``,
    computed on this ``df``'s own real, pooled data -- not a fixed list tuned
    on one project's correlation structure, since that would not generalize
    to a different project's organoid morphology.

    Iterates features in the order given, keeping a feature unless it
    correlates with an already-kept feature above ``correlation_threshold``
    (absolute correlation). Spearman (the default) rather than Pearson,
    because several of these features are related by a deterministic but
    nonlinear monotonic transform -- e.g. Aspect_Ratio and Eccentricity are
    both pure functions of the same major/minor-axis ratio, and Pearson can
    substantially underestimate that redundancy (confirmed empirically:
    Pearson r=0.71 but Spearman r=1.00 for that pair on real data) while
    Spearman correctly reports it as r=1.00.

    Because earlier features in the list are preferred over later, more
    redundant ones, simply ordering ``candidate_features`` with a preferred
    representative first (e.g. Area_um2 before Diameter_um/Perimeter_um/
    Feret_Diameter_Max_um; Aspect_Ratio before Eccentricity) is enough to
    express that preference -- no separate priority list is needed.

    Returns (selected_features, dropped_log), where dropped_log is a list of
    dicts recording each excluded feature, which retained feature it
    correlated with, and the correlation value -- for transparency about why
    a feature was left out of the PCA.
    """
    available = [f for f in candidate_features if f in df.columns]
    numeric_df = df[available].apply(pd.to_numeric, errors='coerce')
    corr = numeric_df.corr(method=method).abs()

    selected = []
    dropped_log = []
    for feature in available:
        if numeric_df[feature].dropna().empty:
            dropped_log.append({
                'feature': feature, 'kept_instead_of': None,
                'correlation': np.nan, 'reason': 'no valid data',
            })
            continue

        conflict = None
        conflict_corr = 0.0
        for kept in selected:
            r = corr.loc[feature, kept]
            if pd.notna(r) and r > conflict_corr:
                conflict_corr = r
                conflict = kept

        if conflict is not None and conflict_corr > correlation_threshold:
            dropped_log.append({
                'feature': feature,
                'kept_instead_of': conflict,
                'correlation': conflict_corr,
                'reason': f"|{method} r| = {conflict_corr:.3f} with {conflict} (> {correlation_threshold})",
            })
        else:
            selected.append(feature)

    return selected, dropped_log


def plot_pca_grid_by_group(base_dir, project_dir, days, batches_complete_info, group_by='batch', features=None, correlation_threshold=0.9):
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    # 1. Identify all unique groups across the whole dataset for consistent coloring
    all_possible_groups = list(set(batch[group_by] for batch in batches_complete_info))

    # 2. Generate the HUSL palette using Seaborn
    # This creates a list of RGB tuples
    husl_palette = sns.color_palette("husl", len(all_possible_groups))
    color_map = dict(zip(all_possible_groups, husl_palette))

    candidate_features = list(features) if features else [
        'Area_um2', 'Diameter_um', 'Perimeter_um',
        'Feret_Diameter_Max_um', 'Aspect_Ratio', 'Circularity',
        'Roundness', 'Solidity', 'Eccentricity', 'Skeleton_Length_um',
    ]

    plate_group_map = {batch['batch']: batch[group_by] for batch in batches_complete_info}
    valid_plates = plate_group_map.keys()
    markers = ["o", "v", "s", "*", "D", "P", "X", "^"]

    # --- Pool every day/batch's data once, both to select a non-redundant
    # feature subset from this project's own real correlation structure, and
    # to avoid re-reading each CSV a second time inside the day loop below. ---
    pooled = []
    for day in days:
        for batch in valid_plates:
            metrics_csv = os.path.join(
                base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv'
            )
            if os.path.exists(metrics_csv):
                pooled_df = pd.read_csv(metrics_csv)
                pooled_df = filter_by_diameter(pooled_df, day)
                if pooled_df.empty:
                    continue
                pooled_df = pooled_df.copy()
                pooled_df['Day'] = day
                pooled_df['Group'] = plate_group_map[batch]
                pooled.append(pooled_df)

    if not pooled:
        print(f"No data available for PCA grid ({group_by}).")
        return

    df_pool = pd.concat(pooled, ignore_index=True)

    features, dropped_log = _select_non_redundant_features(
        df_pool, candidate_features, correlation_threshold=correlation_threshold, method="spearman",
    )
    if len(features) < 2:
        print(f"PCA grid ({group_by}) skipped: fewer than 2 non-redundant features survived selection ({features}).")
        return

    print(f"PCA grid ({group_by}): using non-redundant features {features}")
    if dropped_log:
        pd.DataFrame(dropped_log).to_csv(
            os.path.join(plot_data_dir, f"pca_{group_by}_feature_selection.csv"), index=False
        )
        for entry in dropped_log:
            if entry['kept_instead_of']:
                print(f"  dropped {entry['feature']}: {entry['reason']}")

    n_cols = 3
    n_rows = int(np.ceil(len(days) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    sns.set_style("whitegrid")
    
    axes = axes.flatten()

    group_handles = {} # To store legend info

    for idx, day in enumerate(days):
        df_all = df_pool[df_pool['Day'] == day].dropna(subset=features)

        if df_all.empty:
            axes[idx].axis('off')
            continue

        X_scaled = StandardScaler().fit_transform(df_all[features].values)
        pcs = PCA(n_components=2).fit_transform(X_scaled)

        df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
        df_pca['Group'] = df_all['Group'].values

        unique_groups_this_day = sorted(df_pca['Group'].unique())
        ax = axes[idx]

        for i, group in enumerate(all_possible_groups):
            if group not in unique_groups_this_day:
                continue
                
            subset = df_pca[df_pca['Group'] == group]
            
            # Use the pre-calculated HUSL color
            group_color = color_map[group]
            group_marker = markers[i % len(markers)]

            ax.scatter(
                subset['PC1'], subset['PC2'],
                color=group_color,
                marker=group_marker,
                label=group,
                alpha=0.6,
                s=20 # Reduced size for better visibility in density
            )
            
            # Store handle for global legend if not already there
            if group not in group_handles:
                group_handles[group] = mlines.Line2D(
                    [], [], color=group_color,
                    marker=group_marker,
                    linestyle='None', markersize=8, label=group
                )

        ax.set_title(f'Day {day}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
        
    # Remove unused subplots
    for j in range(len(days), len(axes)):
        fig.delaxes(axes[j])

    # Global legend using the handles we collected
    # Sorting handles to match the group order
    sorted_labels = sorted(group_handles.keys())
    handles = [group_handles[lbl] for lbl in sorted_labels]
    
    fig.legend(
        handles=handles, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05), 
        ncol=min(len(handles), 5), 
        title=None,
        frameon=False
    )

    # 3. Use rect to make room at the bottom for the legend (left, bottom, right, top)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    #fig.tight_layout()

    # Save plot
    filename = os.path.join(plot_dir, f'PCA_grid_by_{group_by}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def _cluster_purity(cluster_labels, condition_labels):
    """Weighted majority-vote purity: for each real cluster (excluding
    HDBSCAN's -1 noise label), the fraction of its members belonging to
    that cluster's single most common condition, averaged across clusters
    weighted by cluster size. 1.0 means every cluster is made up of a
    single condition; the chance-level baseline is roughly the largest
    condition's own share of the population, not 0 -- compare against that,
    not against zero."""
    cluster_labels = np.asarray(cluster_labels)
    condition_labels = np.asarray(condition_labels)
    mask = cluster_labels != -1
    if mask.sum() == 0:
        return float('nan')
    df = pd.DataFrame({'cluster': cluster_labels[mask], 'condition': condition_labels[mask]})
    majority_counts = df.groupby('cluster')['condition'].agg(lambda s: s.value_counts().iloc[0])
    return float(majority_counts.sum() / mask.sum())


def plot_umap_clusters(
    base_dir, project_dir, days, batches_complete_info, group_by='batch',
    segregation_groups=None,
    features=None, correlation_threshold=0.9, min_cluster_size=5,
    cluster_selection_method='eom',
    n_neighbors=15, min_dist=0.1, random_state=42,
):
    """
    UMAP + HDBSCAN clustering of organoid morphology, one embedding per day,
    to surface morphological subtypes that a linear method (PCA, see
    plot_pca_grid_by_group) can blur together into a single elongated
    cloud.

    Feature set: the same non-redundant selection as plot_pca_grid_by_group
    (_select_non_redundant_features, computed once on this project's own
    data pooled across every day in ``days``), so both dimensionality-
    reduction views use an identical, redundancy-checked feature basis and
    stay directly comparable to each other.

    Clustering runs directly on each day's own 2D UMAP embedding (not a
    high-dimensional graph, unlike typical single-cell pipelines) -- the
    simplest choice, and the one where what you see in the scatter plot is
    exactly what was clustered, at the cost of inheriting UMAP's own
    well-known distortion of relative distances. Treat cluster boundaries
    as a hypothesis to follow up on, not a statistically self-justifying
    partition. HDBSCAN's noise label (-1, "not confidently assigned to any
    cluster") is plotted in light gray and excluded from the downstream
    distinguishing-feature stats.

    ``cluster_selection_method`` defaults to "eom" (HDBSCAN's own default,
    excess of mass): picks whichever clusters are most *stable* across the
    condensed cluster tree, adopted here as this pipeline's standard per
    explicit instruction. The alternative, "leaf", selects the tree's
    leaves directly instead, trading more noise points and smaller clusters
    for finer sub-structure -- confirmed empirically on real data
    (6_WNT_treatment/AB115-AB117, Day 2) to recover 42 clusters where "eom"
    collapses everything but a couple of clearly-separate outlier groups
    into one undifferentiated mass (2 clusters total: sizes 539 and 11, out
    of n=550). Pass "leaf" in for a project/day where "eom" is visibly
    over-collapsing a population with real sub-structure.

    Each day is embedded and clustered independently -- cluster numbering
    carries no meaning across days (Day 10's cluster 0 has no relationship
    to Day 20's cluster 0).

    For every (day, cluster) with at least 2 members (and at least 2
    non-members), computes Hedges' g and a Welch's t-test (that cluster
    vs. every other organoid that day, pooled) per feature, BH-FDR
    corrected across every (day, cluster, feature) test -- the same
    convention as every other statistical test in this module -- to answer
    "which features distinguish this cluster from the rest". Also cross-
    tabulates cluster membership against ``group_by``'s existing groups
    (does this cluster correspond to a known condition, or cut across
    them?).

    Saves cluster assignments, the distinguishing-feature stats table, the
    cluster x group contingency table, and -- for every column in
    ``segregation_groups`` (default: just ``group_by``) -- a segregation
    summary (Adjusted Rand Index, Normalized Mutual Information, weighted
    cluster purity, and a chi-square test of independence between cluster
    and that grouping variable, all computed with noise (-1) excluded) as
    CSVs under plots/plot_data/, plus the UMAP scatter grid, one cluster x
    feature effect-size heatmap per day, and one cluster-composition
    stacked-bar figure per (segregation_groups column, day) pair (skipped
    for days with fewer than 1 real cluster) under plots/ -- the
    composition figure for a given day is sized with the exact same
    width/height formula as that day's effect-size heatmap (see
    BAR_SLOT_HEIGHT_IN below), and both sort clusters the same way
    (ascending, cluster 0 at the top), so the two figure types are
    directly comparable side by side for the same day.

    ARI/NMI answer "does cluster membership statistically align with this
    known grouping" (0 = no better than chance, 1 = perfect agreement;
    ARI is chance-corrected, NMI is not) -- the direct, quantitative answer
    to "can this clustering segregate experimental conditions". A high
    value means morphology-based clusters recover the known groups (the
    conditions are morphologically distinguishable); a value near 0 means
    the clusters cut across conditions (morphology doesn't separate them
    at this developmental stage, at least not via these features).

    Returns (assignments_df, stats_df).
    """
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    segregation_groups = list(segregation_groups) if segregation_groups else [group_by]

    candidate_features = list(features) if features else [
        'Area_um2', 'Diameter_um', 'Perimeter_um',
        'Feret_Diameter_Max_um', 'Aspect_Ratio', 'Circularity',
        'Roundness', 'Solidity', 'Eccentricity', 'Skeleton_Length_um',
    ]

    batch_metadata_map = {batch['batch']: batch for batch in batches_complete_info}
    valid_plates = batch_metadata_map.keys()

    pooled = []
    for day in days:
        for batch in valid_plates:
            metrics_csv = os.path.join(
                base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv'
            )
            if os.path.exists(metrics_csv):
                pooled_df = pd.read_csv(metrics_csv)
                pooled_df = filter_by_diameter(pooled_df, day)
                if pooled_df.empty:
                    continue
                pooled_df = _ensure_metric_identity_columns(pooled_df).copy()
                pooled_df['Day'] = day
                pooled_df['Group'] = batch_metadata_map[batch].get(group_by, batch)
                for col in segregation_groups:
                    pooled_df[col] = batch_metadata_map[batch].get(col, batch)
                pooled.append(pooled_df)

    if not pooled:
        print(f"No data available for UMAP clustering ({group_by}).")
        return pd.DataFrame(), pd.DataFrame()

    df_pool = pd.concat(pooled, ignore_index=True)

    selected_features, dropped_log = _select_non_redundant_features(
        df_pool, candidate_features, correlation_threshold=correlation_threshold, method="spearman",
    )
    if len(selected_features) < 2:
        print(f"UMAP clustering skipped: fewer than 2 non-redundant features survived selection ({selected_features}).")
        return pd.DataFrame(), pd.DataFrame()

    print(f"UMAP clustering ({group_by}): using non-redundant features {selected_features}")

    sns.set_style("whitegrid")

    all_assignments = []
    all_stats = []
    all_contingency = []
    all_segregation = []
    id_columns = [c for c in ['Organoid_ID', 'Batch', 'Group', 'Day'] + segregation_groups if c in df_pool.columns]

    for day in days:
        df_day = df_pool[df_pool['Day'] == day].dropna(subset=selected_features).reset_index(drop=True)

        min_required = max(min_cluster_size, 3)
        if len(df_day) < min_required:
            print(f"Day {day}: only {len(df_day)} organoids, skipping (need >= {min_required}).")
            continue

        # One standalone figure per day, at the standard figure size -- not
        # a combined subplot grid -- so each day's embedding is its own
        # plot rather than a panel sharing space with every other day.
        fig, ax = plt.subplots(figsize=(STANDARD_FIG_WIDTH_IN, STANDARD_FIG_HEIGHT_IN))

        X_scaled = StandardScaler().fit_transform(df_day[selected_features].values)
        embedding = umap.UMAP(
            n_neighbors=min(n_neighbors, len(df_day) - 1),
            min_dist=min_dist,
            random_state=random_state,
        ).fit_transform(X_scaled)

        cluster_labels = HDBSCAN(
            min_cluster_size=min_cluster_size, cluster_selection_method=cluster_selection_method
        ).fit_predict(embedding)

        df_day['UMAP1'] = embedding[:, 0]
        df_day['UMAP2'] = embedding[:, 1]
        df_day['Cluster'] = cluster_labels
        all_assignments.append(df_day[id_columns + ['UMAP1', 'UMAP2', 'Cluster'] + selected_features])

        unique_clusters = sorted(c for c in set(cluster_labels) if c != -1)
        noise_mask = cluster_labels == -1
        palette = dict(zip(unique_clusters, sns.color_palette("husl", max(len(unique_clusters), 1))))

        if noise_mask.any():
            ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1], color='lightgray', s=10, alpha=0.5, zorder=1)
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            ax.scatter(embedding[mask, 0], embedding[mask, 1], color=palette[cluster_id], s=15, alpha=0.8, zorder=2)
            # Cluster label placed directly on the cluster (at its own
            # centroid) instead of in a side/corner legend -- with 10+
            # clusters a legend key becomes a long list the reader has to
            # cross-reference by color; an in-place label reads directly off
            # the point cloud.
            centroid = embedding[mask].mean(axis=0)
            ax.annotate(
                f'C{cluster_id}', xy=centroid, ha='center', va='center',
                fontsize=8, fontweight='bold', color='black', zorder=3,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7),
            )

        ax.set_title(f'Day {day} ({len(unique_clusters)} clusters, n={len(df_day)})')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.grid(True, alpha=0.3)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if unique_clusters:
            contingency = df_day.groupby(['Cluster', 'Group']).size().reset_index(name='Count')
            contingency['Day'] = day
            all_contingency.append(contingency)

            non_noise_mask = (cluster_labels != -1)
            for col in segregation_groups:
                if col not in df_day.columns:
                    continue
                condition_values = df_day[col].values
                cross_tab = pd.crosstab(cluster_labels[non_noise_mask], condition_values[non_noise_mask])
                if cross_tab.shape[0] >= 2 and cross_tab.shape[1] >= 2:
                    try:
                        chi2, chi2_p, _, _ = chi2_contingency(cross_tab)
                    except ValueError:
                        chi2, chi2_p = float('nan'), float('nan')
                else:
                    chi2, chi2_p = float('nan'), float('nan')

                all_segregation.append({
                    'Day': day,
                    'Segregation_Group': col,
                    'N_Clusters': len(unique_clusters),
                    'N_Points_Used': int(non_noise_mask.sum()),
                    'N_Noise': int((~non_noise_mask).sum()),
                    'N_Group_Categories': df_day.loc[non_noise_mask, col].nunique(),
                    'ARI': adjusted_rand_score(condition_values[non_noise_mask], cluster_labels[non_noise_mask]),
                    'NMI': normalized_mutual_info_score(condition_values[non_noise_mask], cluster_labels[non_noise_mask]),
                    'Weighted_Purity': _cluster_purity(cluster_labels, condition_values),
                    'Largest_Group_Share': df_day.loc[non_noise_mask, col].value_counts(normalize=True).iloc[0],
                    'Chi2': chi2,
                    'Chi2_p': chi2_p,
                })

        for cluster_id in unique_clusters:
            cluster_mask = df_day['Cluster'] == cluster_id
            if cluster_mask.sum() < 2 or (~cluster_mask).sum() < 2:
                continue
            for feature in selected_features:
                vals_cluster = df_day.loc[cluster_mask, feature].dropna()
                vals_rest = df_day.loc[~cluster_mask, feature].dropna()
                if len(vals_cluster) > 1 and len(vals_rest) > 1:
                    _, p_val = ttest_ind(vals_cluster, vals_rest, equal_var=False)
                    all_stats.append({
                        'Day': day, 'Cluster': cluster_id,
                        'N_Cluster': len(vals_cluster), 'N_Rest': len(vals_rest),
                        'Feature': feature,
                        'Cluster_Mean': vals_cluster.mean(), 'Rest_Mean': vals_rest.mean(),
                        'hedges_g': _hedges_g(vals_cluster, vals_rest),
                        'p_raw': p_val,
                    })

        plt.tight_layout()
        save_path = os.path.join(plot_dir, f'umap_clusters_by_{group_by}_day{day}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved: {save_path}")

    assignments_df = pd.concat(all_assignments, ignore_index=True) if all_assignments else pd.DataFrame()
    if not assignments_df.empty:
        assignments_df.to_csv(
            os.path.join(plot_data_dir, f"umap_cluster_assignments_{group_by}.csv"), index=False
        )

    contingency_df = pd.concat(all_contingency, ignore_index=True) if all_contingency else pd.DataFrame()
    if not contingency_df.empty:
        contingency_df.to_csv(
            os.path.join(plot_data_dir, f"umap_cluster_vs_{group_by}_contingency.csv"), index=False
        )

    segregation_df = pd.DataFrame(all_segregation)
    if not segregation_df.empty:
        segregation_df.to_csv(
            os.path.join(plot_data_dir, "umap_cluster_segregation_summary.csv"), index=False
        )

        print("\n--- Does clustering segregate experimental conditions? (ARI/NMI, noise excluded) ---")
        for _, row in segregation_df.iterrows():
            chi2_str = f"chi2 p={row['Chi2_p']:.4f}" if pd.notna(row['Chi2_p']) else "chi2 n/a"
            print(
                f"Day {row['Day']}, '{row['Segregation_Group']}' ({row['N_Group_Categories']} categories, "
                f"{row['N_Clusters']} clusters, n={row['N_Points_Used']}): "
                f"ARI={row['ARI']:.3f}, NMI={row['NMI']:.3f}, "
                f"purity={row['Weighted_Purity']:.3f} (vs {row['Largest_Group_Share']:.3f} baseline), {chi2_str}"
            )

        # --- Cluster composition stacked bar, one figure per (segregation
        # group, day) -- matching umap_cluster_effect_size_heatmap's own
        # one-figure-per-day structure (rather than the previous single
        # figure combining every day, whose height summed cluster counts
        # across ALL days and so could never match a single day's heatmap)
        # so the two figure types are directly comparable for the same day.
        BAR_SLOT_HEIGHT_IN = 0.3
        for col in segregation_groups:
            if assignments_df.empty or col not in assignments_df.columns:
                continue
            col_data = assignments_df[assignments_df['Cluster'] != -1]
            if col_data.empty:
                continue

            all_categories = sorted(col_data[col].dropna().unique())
            cat_palette = dict(zip(all_categories, sns.color_palette("husl", len(all_categories))))

            for day in sorted(col_data['Day'].unique()):
                day_data = col_data[col_data['Day'] == day]
                n_clusters = day_data['Cluster'].nunique()
                if n_clusters < 1:
                    continue

                # Same width (STANDARD_FIG_WIDTH_IN) and the same
                # 0.3-in-per-cluster/floor-at-STANDARD_FIG_HEIGHT_IN height
                # formula as umap_cluster_effect_size_heatmap's figsize for
                # this same day, so the two figure types share the same
                # size for the same cluster count.
                bar_height = max(n_clusters * BAR_SLOT_HEIGHT_IN, STANDARD_FIG_HEIGHT_IN)
                fig_bar, ax = plt.subplots(figsize=(STANDARD_FIG_WIDTH_IN, bar_height))

                proportions = day_data.groupby(['Cluster', col]).size().unstack(fill_value=0)
                proportions = proportions.div(proportions.sum(axis=1), axis=0)
                proportions = proportions.reindex(columns=all_categories, fill_value=0)
                proportions.plot(
                    kind='barh', stacked=True, ax=ax,
                    color=[cat_palette[c] for c in all_categories],
                    legend=False, width=0.8,
                )
                ax.invert_yaxis()  # cluster 0 at the top, matching the heatmap's row order
                ax.set_ylabel("Cluster")
                ax.set_xlabel("Proportion")
                ax.tick_params(axis='y', rotation=0)
                ax.grid(True, alpha=0.3)

                handles = [
                    mlines.Line2D([], [], color=cat_palette[c], marker='s', linestyle='None', markersize=10, label=str(c))
                    for c in all_categories
                ]
                fig_bar.legend(
                    handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                    ncol=min(len(handles), 2), frameon=False, title="",
                )
                composition_path = os.path.join(plot_dir, f"umap_cluster_composition_by_{col}_day{day}.png")
                plt.tight_layout()
                plt.savefig(composition_path, dpi=300, bbox_inches='tight')
                plt.close(fig_bar)
                print(f"Plot saved: {composition_path}")

    stats_df = pd.DataFrame(all_stats)
    if not stats_df.empty:
        _, corrected, _, _ = multipletests(stats_df['p_raw'].values, method='fdr_bh')
        stats_df['p_adj'] = corrected
        stats_df['stars'] = stats_df['p_adj'].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        )
        stats_df.to_csv(
            os.path.join(plot_data_dir, f"umap_cluster_distinguishing_features_{group_by}.csv"), index=False
        )

        print("\n--- Top distinguishing features per cluster (Hedges' g, BH-FDR corrected) ---")
        for (day, cluster_id), cluster_group in stats_df.groupby(['Day', 'Cluster']):
            top = cluster_group.reindex(cluster_group['hedges_g'].abs().sort_values(ascending=False).index).head(3)
            summary = ", ".join(f"{r['Feature']} (g={r['hedges_g']:.2f}, {r['stars']})" for _, r in top.iterrows())
            n_members = int(cluster_group['N_Cluster'].iloc[0])
            print(f"Day {day}, Cluster {cluster_id} (n={n_members}): {summary}")

        for day in sorted(stats_df['Day'].unique()):
            day_stats = stats_df[stats_df['Day'] == day]
            pivot = day_stats.pivot(index='Cluster', columns='Feature', values='hedges_g')
            if pivot.empty:
                continue
            # Width fixed (STANDARD_FIG_WIDTH_IN); height depends only on
            # cluster count (0.8/cluster + 1.5 base, floored at
            # STANDARD_FIG_HEIGHT_IN) -- identical formula to
            # umap_cluster_composition_by_{group}'s bars_height, so the two
            # figure types share the same height for the same cluster count.
            heatmap_height = max(0.3 * pivot.shape[0], STANDARD_FIG_HEIGHT_IN)
            fig, ax = plt.subplots(figsize=(STANDARD_FIG_WIDTH_IN, heatmap_height))
            # annot_kws fontsize set explicitly, independent of the general
            # (13pt) rcParams font size -- at this figure's fixed, narrow
            # width, cell annotations for every feature column (as many as
            # 10) would otherwise run into each other.
            sns.heatmap(
                pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                annot_kws={"size": 7},
                cbar_kws={'label': "Hedges' g (cluster vs rest)"},
                xticklabels=[_display_feature_name(f) for f in pivot.columns],
            )
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            plt.tight_layout()
            heatmap_path = os.path.join(plot_dir, f"umap_cluster_effect_size_heatmap_day{day}.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()

    return assignments_df, stats_df


def plot_feature_grid_scatters(base_dir, project_dir, days, batches_complete_info, features, metric_names, group_by='batch', min_wells_for_stats=2):
    """
    Creates a grid of scatter plots with Welch's T-test and FDR correction.
    Vertical axis labels and ticks are hidden for internal panels.

    Called once per grouping variable from plot.py's `for group in
    groups:` loop, so both outputs are named
    "{feature}_grid_scatterplot_by_{group_by}.png"/
    "{feature}_grid_scatterplot_stats_by_{group_by}.csv" (matching
    plot_pca_grid_by_group's PCA_grid_by_{group_by}.png convention) -- a
    group_by-agnostic filename here would mean every grouping variable's
    figure/stats land on the same path and only the last group in
    `plotting_metadata.json`'s "groups" list survives on disk.
    """
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    if isinstance(features, str):
        features = [features]
        metric_names = [metric_names]

    # 1. Setup metadata mapping
    plate_group_map = {
        batch_id: info.get(group_by, batch_id)
        for batch_id, info in batches_complete_info.items()
    }
    all_possible_groups = sorted(list(set(plate_group_map.values())))

    # 2. Setup Plotting Aesthetics
    husl_palette = sns.color_palette("husl", len(all_possible_groups))
    color_dict = dict(zip(all_possible_groups, husl_palette))

    for f_idx, feature in enumerate(features):
        metric_label = metric_names[f_idx]
        if feature == "Area_um2":
            metric_label = _um2_label_to_mm2(metric_label)

        # --- PHASE A: DATA GATHERING & STATS CALCULATION ---
        # We need to calculate ALL p-values first to apply FDR correction across the whole feature
        stats_list = []
        skipped_stats_list = []
        feature_data_by_day = {}

        for day in days:
            day_data = []
            day_data_well = []
            for batch_id in plate_group_map.keys():
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch_id, 'inference_metrics', 'mask_metrics.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)

                    df_filtered = prepare_morphology_records(df, day, [feature], aggregation_level="object")
                    if not df_filtered.empty and feature in df_filtered.columns:
                        df_filtered = df_filtered.copy()
                        df_filtered['Group'] = plate_group_map[batch_id]
                        day_data.append(df_filtered[['Group', feature]])

                    # Statistics run on well-level means (one value per
                    # Image_ID/Well_ID), never on the raw per-object rows
                    # used for the scatter/violin visualization above -- see
                    # plot_feature_over_time's identical convention. Without
                    # this, organoid_mode == "multiple" data would treat
                    # every non-independent organoid from the same well as
                    # its own sample, inflating apparent significance. A
                    # no-op for organoid_mode == "single" (one organoid per
                    # well already).
                    df_well = prepare_morphology_records(df, day, [feature], aggregation_level="well")
                    if not df_well.empty and feature in df_well.columns:
                        df_well = df_well.copy()
                        df_well['Group'] = plate_group_map[batch_id]
                        day_data_well.append(df_well[['Group', feature]])

            if day_data:
                df_day = pd.concat(day_data, ignore_index=True)
                if feature == "Area_um2":
                    df_day[feature] = df_day[feature] * AREA_UM2_TO_MM2
                feature_data_by_day[day] = df_day

            if day_data_well:
                df_day_well = pd.concat(day_data_well, ignore_index=True)

                # Perform Welch's T-test for all pairs present this day
                u_groups = sorted(df_day_well['Group'].unique())
                if len(u_groups) > 1:
                    for i in range(len(u_groups)):
                        for j in range(i + 1, len(u_groups)):
                            g1, g2 = u_groups[i], u_groups[j]
                            vals1 = df_day_well[df_day_well['Group'] == g1][feature].dropna()
                            vals2 = df_day_well[df_day_well['Group'] == g2][feature].dropna()
                            n1, n2 = len(vals1), len(vals2)

                            if n1 >= min_wells_for_stats and n2 >= min_wells_for_stats:
                                _, p_val = ttest_ind(vals1, vals2, equal_var=False)
                                stats_list.append({
                                    'day': day, 'pair': (g1, g2), 'p_raw': p_val,
                                    'hedges_g': _hedges_g(vals1, vals2),
                                    'n1': n1, 'n2': n2, 'Skipped_Insufficient_N': False,
                                })
                            elif n1 >= 1 and n2 >= 1:
                                print(
                                    f"Skipped {g1} vs {g2} on Day {day} ({feature}): insufficient "
                                    f"wells (n={n1}, {n2}) below min_wells_for_stats={min_wells_for_stats}."
                                )
                                skipped_stats_list.append({
                                    'day': day, 'pair': (g1, g2), 'p_raw': np.nan, 'p_corr': np.nan,
                                    'stars': 'skipped', 'hedges_g': np.nan,
                                    'n1': n1, 'n2': n2, 'Skipped_Insufficient_N': True,
                                })

        # Apply FDR correction (Benjamini-Hochberg) across all tests for this feature
        if stats_list:
            raw_pvals = [s['p_raw'] for s in stats_list]
            _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')
            print(f"\n--- FDR Corrected Statistics for {metric_label} by {group_by}, tested on well-level means (n = independent wells) ---")
            for i, s in enumerate(stats_list):
                s['p_corr'] = corrected_pvals[i]
                # Convert p-value to stars
                p = s['p_corr']
                s['stars'] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(
                    f"Day {s['day']} {s['pair'][0]} vs {s['pair'][1]}: p_raw={s['p_raw']:.5f} "
                    f"p_adj={p:.5f} ({s['stars']}), Hedges' g = {s['hedges_g']:.3f} (n={s['n1']} vs {s['n2']} wells)"
                )
            # group_by is embedded in the filename (matching plot_pca_grid_by_group's
            # PCA_grid_by_{group_by}.png convention) because this function is called
            # once per grouping variable from plot.py's `for group in groups:`
            # loop -- without it, every group's stats/figure landed on the same path
            # and only the last group in the configured `groups` list survived on disk.
            pd.DataFrame(stats_list + skipped_stats_list).to_csv(
                os.path.join(plot_data_dir, f"{feature}_grid_scatterplot_stats_by_{group_by}.csv"), index=False
            )
        elif skipped_stats_list:
            pd.DataFrame(skipped_stats_list).to_csv(
                os.path.join(plot_data_dir, f"{feature}_grid_scatterplot_stats_by_{group_by}.csv"), index=False
            )

        # --- PHASE B: PLOTTING ---
        n_cols, n_rows = _panel_grid_dims(len(days))
        # No rotated/visible x-tick labels on these panels (see
        # ax.tick_params(labelbottom=False) below) and no per-panel x-label
        # text either, so both the inter-row gap and the legend margin can
        # be tighter than a figure that reserves room for label text there.
        LEGEND_RESERVE_IN = 0.6
        fig = plt.figure(figsize=(STANDARD_FIG_WIDTH_IN/3*n_cols, STANDARD_FIG_HEIGHT_IN * n_rows))
        bottom_frac = LEGEND_RESERVE_IN / (STANDARD_FIG_HEIGHT_IN * n_rows + LEGEND_RESERVE_IN)
        gs = fig.add_gridspec(n_rows, n_cols, top=0.95, bottom=bottom_frac, hspace=0.15, wspace=0.3)
        sns.set_style("whitegrid")
        axes = [fig.add_subplot(gs[i // n_cols, i % n_cols]) for i in range(len(days))]
        for ax in axes[1:]:
            ax.sharey(axes[0])

        # All day-panels share one y-axis (via sharey above), so the
        # vertical spacing between stacked bracket rows must be computed
        # from ONE shared reference rather than each panel's own y_max --
        # otherwise an earlier day with a much smaller y_max (organoids
        # are smaller early in development) gets a proportionally tiny
        # y_offset that renders as illegibly cramped brackets once drawn
        # against a later, larger day's shared axis range. The GLOBAL max
        # across every day (rather than, say, the last day specifically)
        # is used as that reference -- a later day isn't guaranteed to
        # have the largest max (e.g. fewer wells surviving by then) even
        # when the feature grows on average -- each panel still starts its
        # own stack at its own y_max, only the step size between rows is
        # shared.
        all_day_maxes = [df[feature].max() for df in feature_data_by_day.values()]
        shared_y_offset = max(all_day_maxes) * 0.3 if all_day_maxes else None

        group_handles = {}

        for d_idx, day in enumerate(days):
            ax = axes[d_idx]
            if day not in feature_data_by_day:
                ax.axis('off')
                continue

            df_day = feature_data_by_day[day]
            u_groups = sorted(df_day['Group'].unique())

            # 1. Violin (distribution shape) + Scatter/Stripplot (individual points)
            sns.violinplot(
                data=df_day, x='Group', y=feature, order=u_groups,
                palette=color_dict, hue='Group', legend=False, ax=ax,
                inner=None, cut=0, alpha=0.35, linewidth=0.8, zorder=1,
            )
            sns.stripplot(
                data=df_day, x='Group', y=feature, order=u_groups,
                palette=color_dict, hue='Group', jitter=True, alpha=0.8,
                s=2, ax=ax, legend=False, zorder=2,
            )

            # 2. Mean Lines
            means = df_day.groupby('Group')[feature].mean().reindex(u_groups)
            for x_pos, mean_val in enumerate(means):
                ax.hlines(y=mean_val, xmin=x_pos - 0.25, xmax=x_pos + 0.25, color="black", lw=1, zorder=5)

            # 3. Add statistical stars
            day_stats = [s for s in stats_list if s['day'] == day]
            y_max = df_day[feature].max()
            y_offset = shared_y_offset if shared_y_offset else y_max * 0.3
            
            for s_idx, s in enumerate(day_stats):
                #if s['stars'] == "ns": continue
                g1, g2 = s['pair']
                x1, x2 = u_groups.index(g1), u_groups.index(g2)
                
                # Draw brackets/stars manually
                h = y_max + (s_idx * y_offset)
                ax.plot([x1, x1, x2, x2], [h, h + y_offset*0.2, h + y_offset*0.2, h], lw=0.5, color='black')
                ax.text((x1 + x2) * 0.5, h + y_offset*0.2, _stat_annotation_text(s), ha='center', va='bottom', color='black', fontsize=stats_fontsize)

            # 4. Vertical Axis Cleanup
            if d_idx % n_cols > 0:
                ax.set_ylabel("") # Remove axis title
                ax.tick_params(labelleft=False) # Remove numerical labels
                # ax.spines['left'].set_visible(False) Optional: remove left spine for "floating" look
            else:
                ax.set_ylabel(metric_label)
            
            ax.tick_params(labelbottom=False)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("")
            ax.set_title(f"Day {day}", fontsize=10)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False) 

        # 5. Global Legend
        for i, grp in enumerate(all_possible_groups):
            group_handles[grp] = mlines.Line2D(
                [], [], color=color_dict[grp], marker='o', 
                linestyle='None', markersize=10, label=grp
            )
        
        handles = [group_handles[k] for k in sorted(group_handles.keys())]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                   ncol=min(len(handles), 2), frameon=False, title=None)

        #plt.suptitle(f"{metric_label} Growth Trajectory", fontsize=16, fontweight='bold', y=1.02)
        # 6. Save
        # group_by embedded here too, for the same reason as the stats CSV above.
        save_path = os.path.join(plot_dir, f"{feature}_grid_scatterplot_by_{group_by}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def _to_numeric_series(values):
    return pd.to_numeric(pd.Series(values), errors='coerce').dropna()


def _safe_pct(numerator, denominator):
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return (numerator / denominator) * 100


def _resolve_std_multiplier(manual_std_thresholds, feature=None, group=None, day=None, default=3.0):
    """Resolve a manual mean +/- N SD threshold from a scalar or lookup dict."""
    if manual_std_thresholds is None:
        return default

    if isinstance(manual_std_thresholds, (int, float)):
        return float(manual_std_thresholds)

    if not isinstance(manual_std_thresholds, dict):
        return default

    lookup_keys = [
        (feature, group, day),
        (feature, day),
        (group, day),
        feature,
        str(day),
        day,
        "default",
    ]
    for key in lookup_keys:
        if key in manual_std_thresholds:
            value = manual_std_thresholds[key]
            if isinstance(value, dict):
                value = value.get(day, value.get(str(day), value.get("default", default)))
            return float(value)

    if feature in manual_std_thresholds and isinstance(manual_std_thresholds[feature], dict):
        feature_thresholds = manual_std_thresholds[feature]
        return float(feature_thresholds.get(day, feature_thresholds.get(str(day), feature_thresholds.get("default", default))))

    return default


# Physical size measures -- strictly positive and typically right-skewed
# (log-normal-ish), so a symmetric fence on the raw scale over-flags the
# upper tail as "outliers" when it's really just ordinary skew. Shape
# descriptors (Aspect_Ratio, Circularity, Roundness, Solidity,
# Eccentricity) are dimensionless/bounded ratios, not size-like, and are
# deliberately excluded here -- they keep the raw-scale fence.
SIZE_LIKE_FEATURES = {
    "Area_um2", "Diameter_um", "Perimeter_um", "Feret_Diameter_Max_um", "Skeleton_Length_um",
}


def calculate_outlier_thresholds(
    values,
    method="iqr",
    iqr_multiplier=1.5,
    std_multiplier=3.0,
    percentile_bounds=(0.05, 0.95),
    log_transform=False,
):
    """
    Calculate lower/upper outlier thresholds and robust descriptive statistics.

    The default is Tukey's IQR rule: Q1 - 1.5*IQR and Q3 + 1.5*IQR.
    The 5th/95th percentiles are reported for context, not used as default
    cutoffs unless method="percentile".

    ``log_transform=True`` computes the lower/upper fence on log-transformed
    values, then converts the two resulting bounds back to the original
    scale before returning them -- appropriate for right-skewed,
    strictly-positive size-like features (see SIZE_LIKE_FEATURES), where a
    symmetric fence on the raw scale over-flags the upper tail. The
    descriptive stats (mean/std/median/quantiles/iqr) are always computed
    on the original, untransformed scale regardless of this setting --
    only the fence computation itself changes. Non-positive values (where
    log is undefined) are dropped before the transform; if that empties
    the sample, silently falls back to the untransformed (raw-scale) fence.
    """
    vals = _to_numeric_series(values)
    if vals.empty:
        return {
            "method": method,
            "lower_threshold": np.nan,
            "upper_threshold": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "iqr": np.nan,
            "n": 0,
        }

    q05 = vals.quantile(0.05)
    q25 = vals.quantile(0.25)
    median = vals.quantile(0.50)
    q75 = vals.quantile(0.75)
    q95 = vals.quantile(0.95)
    iqr = q75 - q25
    mean = vals.mean()
    std = vals.std(ddof=1)
    method_key = str(method).lower()

    fence_vals = vals
    log_scale = False
    if log_transform:
        positive_vals = vals[vals > 0]
        if not positive_vals.empty:
            fence_vals = np.log(positive_vals)
            log_scale = True

    fence_q25 = fence_vals.quantile(0.25)
    fence_q75 = fence_vals.quantile(0.75)
    fence_iqr = fence_q75 - fence_q25
    fence_mean = fence_vals.mean()
    fence_std = fence_vals.std(ddof=1)

    if method_key in {"iqr", "tukey", "tukey_iqr"}:
        lower_threshold = fence_q25 - iqr_multiplier * fence_iqr
        upper_threshold = fence_q75 + iqr_multiplier * fence_iqr
        method_label = f"Tukey IQR ({iqr_multiplier}x)"
    elif method_key in {"mean_sd", "sd", "std", "standard_deviation"}:
        lower_threshold = fence_mean - std_multiplier * fence_std
        upper_threshold = fence_mean + std_multiplier * fence_std
        method_label = f"Mean +/- {std_multiplier} SD"
    elif method_key in {"percentile", "quantile"}:
        lower_q, upper_q = percentile_bounds
        lower_threshold = fence_vals.quantile(lower_q)
        upper_threshold = fence_vals.quantile(upper_q)
        method_label = f"Percentile ({lower_q:.0%}-{upper_q:.0%})"
    else:
        raise ValueError("Unknown outlier method. Use 'iqr', 'mean_sd', or 'percentile'.")

    if log_scale:
        lower_threshold = np.exp(lower_threshold)
        upper_threshold = np.exp(upper_threshold)
        method_label += ", log-scale"

    return {
        "method": method_label,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "mean": mean,
        "std": std,
        "median": median,
        "q05": q05,
        "q25": q25,
        "q75": q75,
        "q95": q95,
        "iqr": iqr,
        "n": len(vals),
    }


def _classify_outliers(df, feature, thresholds):
    values = pd.to_numeric(df[feature], errors='coerce')
    lower = thresholds["lower_threshold"]
    upper = thresholds["upper_threshold"]

    if pd.isna(lower) or pd.isna(upper):
        outlier_mask = pd.Series(False, index=df.index)
    else:
        outlier_mask = (values < lower) | (values > upper)

    directions = pd.Series("", index=df.index, dtype=object)
    directions[values < lower] = "low"
    directions[values > upper] = "high"

    return values, outlier_mask, directions


def plot_variability_and_outliers(
    base_dir,
    project_dir,
    batches_complete_info,
    features,
    metric_names,
    group_by="batch",
    outlier_method="iqr",
    iqr_multiplier=1.5,
    std_multiplier=3.0,
    manual_std_thresholds=None,
    percentile_bounds=(0.05, 0.95),
    mad_threshold=None,
    aggregation_level="object",
    transition_days=None,
):
    """
    Plot variability and quantify morphology outliers.

    Default outlier rule: Tukey IQR fences, Q1 - 1.5*IQR and Q3 + 1.5*IQR.
    For manual mean/SD thresholding, call with outlier_method="mean_sd" and set
    std_multiplier or manual_std_thresholds.

    ``aggregation_level`` here (plotting_metadata.json's
    outlier_detection.aggregation_level) controls ONLY what this function
    reports outliers AMONG -- individual organoids ("object", the default)
    or well-means ("well"). It is intentionally independent of what
    --remove_outliers actually REMOVES elsewhere in the pipeline: real
    removal (configure_outlier_removal/filter_by_diameter, used by every
    other plotting function via prepare_morphology_records) always runs on
    individual organoid rows before any well-mean is computed, regardless
    of this setting -- see audit_outlier_exclusion_by_well for a report of
    what that removal actually excludes, per well. Setting this to "well"
    changes what this function's own CV/outlier-percentage plots and
    outlier_summary_all_features.csv describe (outliers among well-means,
    a different population than individual organoids), not what
    --remove_outliers strips from the data other figures are drawn from.
    """
    if mad_threshold is not None and outlier_method == "mad":
        print("MAD outlier mode is no longer used here; using Tukey IQR instead.")
        outlier_method = "iqr"

    # This function's whole purpose is reporting what the outliers ARE, so
    # it must always see the full, un-stripped data regardless of whether
    # --remove_outliers is active elsewhere -- otherwise it would just
    # report "0 outliers found" after they'd already been removed upstream.
    _outlier_removal_was_enabled = _OUTLIER_REMOVAL["enabled"]
    _OUTLIER_REMOVAL["enabled"] = False

    plot_dir = _plots_root(base_dir, project_dir)
    outlier_dir = os.path.join(plot_dir, 'quality_control', 'outliers')
    os.makedirs(outlier_dir, exist_ok=True)

    all_summary_rows = []
    all_outlier_rows = []

    for i, feature in enumerate(features):
        all_records = []
        metric_label = metric_names[i] if i < len(metric_names) else feature

        for entry in batches_complete_info:
            batch = entry['batch']
            label = entry.get(group_by, batch)
            for day in entry['days']:
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
                if not os.path.exists(path):
                    continue

                df = pd.read_csv(path)
                df_f = prepare_morphology_records(
                    df,
                    day,
                    [feature],
                    aggregation_level=aggregation_level,
                )
                if df_f.empty or feature not in df_f.columns:
                    continue

                df_f = df_f.copy()
                if 'Organoid_ID' not in df_f.columns:
                    df_f['Organoid_ID'] = [f"{batch}_day{day}_object{idx:04d}" for idx in range(1, len(df_f) + 1)]

                df_f['Batch'] = batch
                df_f['Group'] = label
                df_f['Day'] = day
                all_records.append(df_f[['Batch', 'Group', 'Day', 'Organoid_ID', feature]])

        if not all_records:
            continue

        df_all = pd.concat(all_records, ignore_index=True)
        unique_groups = sorted(df_all['Group'].dropna().unique())
        palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

        # Coefficient of variation (SD/mean, as a %), not raw SD -- raw SD
        # on its own conflates "more variable" with "just bigger" (organoids
        # that have grown 10x by a later day trivially have ~10x the
        # absolute SD even at constant relative spread). Normalizing by the
        # mean isolates whether relative variability itself is changing.
        # NaN wherever the mean is ~0 (division by a near-zero mean would
        # otherwise blow up into an uninterpretably large ratio).
        cv_summary = df_all.groupby(['Group', 'Day'])[feature].agg(['mean', 'std']).reset_index()
        cv_summary['cv_pct'] = np.where(
            cv_summary['mean'].abs() > 1e-9,
            100 * cv_summary['std'] / cv_summary['mean'],
            np.nan,
        )
        fig, ax = plt.subplots(figsize=(STANDARD_FIG_WIDTH_IN, STANDARD_FIG_HEIGHT_IN))
        sns.set_style("whitegrid")
        sns.lineplot(data=cv_summary, x='Day', y='cv_pct', hue='Group', palette=palette, ax=ax, linewidth=1, marker='o', zorder=2, markersize=5)
        ax.set_ylabel(f'CV (%) - {metric_label}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        _annotate_transitions(ax, transition_days)
        plt.savefig(os.path.join(plot_dir, f"{feature}_{aggregation_level}_variability_cv.png"), bbox_inches='tight', dpi=300)
        plt.close()

        feature_summary_rows = []
        feature_outlier_rows = []
        for (group, day), group_df in df_all.groupby(['Group', 'Day']):
            resolved_std_multiplier = _resolve_std_multiplier(
                manual_std_thresholds,
                feature=feature,
                group=group,
                day=day,
                default=std_multiplier,
            )
            thresholds = calculate_outlier_thresholds(
                group_df[feature],
                method=outlier_method,
                iqr_multiplier=iqr_multiplier,
                std_multiplier=resolved_std_multiplier,
                percentile_bounds=percentile_bounds,
                log_transform=feature in SIZE_LIKE_FEATURES,
            )
            values, outlier_mask, directions = _classify_outliers(group_df, feature, thresholds)
            total_count = len(group_df)
            outlier_count = int(outlier_mask.sum())
            low_count = int((directions == "low").sum())
            high_count = int((directions == "high").sum())

            summary_row = {
                'Feature': feature,
                'Aggregation_Level': aggregation_level,
                'Group': group,
                'Day': day,
                'Outlier_Method': thresholds['method'],
                'Lower_Threshold': thresholds['lower_threshold'],
                'Upper_Threshold': thresholds['upper_threshold'],
                'Mean': thresholds['mean'],
                'Std_Dev': thresholds['std'],
                'Median': thresholds['median'],
                'Q05': thresholds['q05'],
                'Q25': thresholds['q25'],
                'Q75': thresholds['q75'],
                'Q95': thresholds['q95'],
                'IQR': thresholds['iqr'],
                'Total_Objects': total_count,
                'Outlier_Count': outlier_count,
                'Low_Outlier_Count': low_count,
                'High_Outlier_Count': high_count,
                'Outlier_Percent': _safe_pct(outlier_count, total_count),
            }
            feature_summary_rows.append(summary_row)
            all_summary_rows.append(summary_row)

            outliers_df = group_df.loc[outlier_mask].copy()
            if not outliers_df.empty:
                outliers_df['Feature'] = feature
                outliers_df['Aggregation_Level'] = aggregation_level
                outliers_df['Value'] = values.loc[outlier_mask].values
                outliers_df['Outlier_Direction'] = directions.loc[outlier_mask].values
                outliers_df['Outlier_Method'] = thresholds['method']
                outliers_df['Lower_Threshold'] = thresholds['lower_threshold']
                outliers_df['Upper_Threshold'] = thresholds['upper_threshold']
                outliers_df['Distance_From_Threshold'] = np.where(
                    outliers_df['Outlier_Direction'] == "low",
                    thresholds['lower_threshold'] - outliers_df['Value'],
                    outliers_df['Value'] - thresholds['upper_threshold'],
                )
                outliers_df = outliers_df[
                    [
                        'Feature', 'Batch', 'Group', 'Day', 'Organoid_ID',
                        'Aggregation_Level',
                        'Value', 'Outlier_Direction', 'Distance_From_Threshold',
                        'Outlier_Method', 'Lower_Threshold', 'Upper_Threshold',
                    ]
                ]
                feature_outlier_rows.extend(outliers_df.to_dict('records'))
                all_outlier_rows.extend(outliers_df.to_dict('records'))

        outlier_summary = pd.DataFrame(feature_summary_rows)
        outlier_summary.to_csv(os.path.join(outlier_dir, f"{feature}_outlier_summary.csv"), index=False)

        outlier_objects = pd.DataFrame(feature_outlier_rows)
        outlier_objects.to_csv(os.path.join(outlier_dir, f"{feature}_outlier_objects.csv"), index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        sns.lineplot(
            data=outlier_summary,
            x='Day',
            y='Outlier_Percent',
            hue='Group',
            marker='s',
            palette=palette,
            ax=ax,
        )
        #ax.set_title(f'Percentage of Outliers in {metric_label} ({outlier_method})')
        ax.set_ylabel('Outliers (%)')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        _annotate_transitions(ax, transition_days)
        plt.savefig(os.path.join(plot_dir, f"{feature}_{aggregation_level}_outlier_percentage.png"), bbox_inches='tight', dpi=300)
        plt.close()

    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_csv(
            os.path.join(outlier_dir, "outlier_summary_all_features.csv"),
            index=False,
        )
    if all_outlier_rows:
        pd.DataFrame(all_outlier_rows).to_csv(
            os.path.join(outlier_dir, "outlier_objects_all_features.csv"),
            index=False,
        )

    _OUTLIER_REMOVAL["enabled"] = _outlier_removal_was_enabled

    print(f"Variability and outlier QC saved to {plot_dir}")


def plot_well_organoid_counts_over_time(base_dir, project_dir, batches_complete_info, group_by="batch", error_bar="sem", transition_days=None):
    """Plot the number of detected organoids per image/well over time
    (mean +/- error_bar -- "sem" by default, or "sd"/"ci"; see
    _resolve_error_bar_values -- set via plotting_metadata.json's
    "line_plot_error_bar")."""
    qc_dir = os.path.join(_plots_root(base_dir, project_dir), 'quality_control')
    os.makedirs(qc_dir, exist_ok=True)

    records = []
    for entry in batches_complete_info:
        batch = entry['batch']
        label = entry.get(group_by, batch)
        for day in entry['days']:
            metrics_dir = os.path.join(
                base_dir,
                project_dir,
                'processed_data',
                f'Day {day}',
                batch,
                'inference_metrics',
            )
            well_summary_path = os.path.join(metrics_dir, 'well_metrics_summary.csv')
            metrics_path = os.path.join(metrics_dir, 'mask_metrics.csv')

            if os.path.exists(well_summary_path):
                well_df = pd.read_csv(well_summary_path)
                count_col = (
                    'Detected_Organoid_Count'
                    if 'Detected_Organoid_Count' in well_df.columns
                    else 'Object_Count_In_Image'
                )
                if count_col not in well_df.columns:
                    continue
                for _, row in well_df.iterrows():
                    records.append({
                        'Batch': batch,
                        'Group': label,
                        'Day': day,
                        'Image_ID': row.get('Image_ID', row.get('Well_ID')),
                        'Well_ID': row.get('Well_ID', row.get('Image_ID')),
                        'Plate_Index': row.get('Plate_Index', np.nan),
                        'Well_Index': row.get('Well_Index', np.nan),
                        'Organoid_Count': row[count_col],
                    })
                continue

            if not os.path.exists(metrics_path):
                continue

            metrics_df = _ensure_metric_identity_columns(pd.read_csv(metrics_path))
            count_df = (
                metrics_df
                .groupby(['Image_ID', 'Well_ID'], dropna=False)
                .agg(Organoid_Count=('Organoid_ID', 'count'))
                .reset_index()
            )
            for _, row in count_df.iterrows():
                records.append({
                    'Batch': batch,
                    'Group': label,
                    'Day': day,
                    'Image_ID': row['Image_ID'],
                    'Well_ID': row['Well_ID'],
                    'Plate_Index': np.nan,
                    'Well_Index': np.nan,
                    'Organoid_Count': row['Organoid_Count'],
                })

    if not records:
        print("No per-well organoid count data found.")
        return pd.DataFrame()

    count_df = pd.DataFrame(records)
    count_df.to_csv(os.path.join(qc_dir, "organoid_counts_by_well.csv"), index=False)

    summary = (
        count_df
        .groupby(['Group', 'Day'], dropna=False)['Organoid_Count']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'count': 'N_Wells'})
    )
    # _resolve_error_bar_values reads 'count' -- keep both names rather than
    # renaming in place, so the CSV is self-explanatory (N_Wells) without
    # touching that shared helper's expected column name.
    summary['count'] = summary['N_Wells']
    summary['sem'] = summary['std'] / np.sqrt(summary['N_Wells'])
    summary['error_value'] = _resolve_error_bar_values(summary, error_bar)
    summary.drop(columns=['count'], inplace=True)
    summary.to_csv(os.path.join(qc_dir, "organoid_counts_by_group_day.csv"), index=False)

    unique_groups = sorted(count_df['Group'].dropna().unique())
    palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

    fig, ax = plt.subplots(figsize=(STANDARD_FIG_WIDTH_IN, STANDARD_FIG_HEIGHT_IN))
    sns.set_style("whitegrid")
    for group in unique_groups:
        group_df = summary[summary['Group'] == group]
        ax.errorbar(
            group_df['Day'],
            group_df['mean'],
            yerr=group_df['error_value'],
            label=group,
            marker='o',
            markersize=3,
            capsize=4,
            color=palette[group],
            linewidth=1,
        )
    ax.spines['top'].set_visible(False)    
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel('Day')
    ax.set_ylabel('Detected organoids per well')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _annotate_transitions(ax, transition_days)
    plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "organoid_counts_per_well_over_time.png"), bbox_inches='tight', dpi=300)
    plt.close()

    return count_df


# Below this, a post-outlier-removal well organoid count is flagged as a
# QC warning in audit_outlier_exclusion_by_well -- not a hard cutoff on
# anything downstream, just a visible signal that a well's mean now rests
# on very little (or no) surviving data.
LOW_ORGANOID_COUNT_WARNING_THRESHOLD = 3


def export_well_level_summary(base_dir, project_dir, batches_complete_info, features):
    """One row per Batch/Well_ID/Day: every requested feature's well-level
    mean (post outlier-removal, respecting whatever --remove_outliers
    configured -- the same data every other well-level plot/statistic in
    this module already uses) plus Object_Count_In_Image, alongside every
    batch-level metadata field from batches_complete_info (cell-line,
    condition, batch_no_plate, ... whatever a project defines) merged in
    directly -- rather than picking just one group_by's label, so this one
    dataset can be re-sliced by ANY of a project's grouping variables later
    without rerunning anything. Individual plotting functions already
    produce their own one-feature-at-a-time well-level CSVs as a side
    effect of the figure they draw; this is the directly reusable
    equivalent that isn't tied to any single plot.

    Saves plots/plot_data/well_level_summary.csv (or
    plots_<variant>/plot_data/... -- see configure_plots_dirname)."""
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    records = []
    for entry in batches_complete_info:
        batch = entry['batch']
        batch_metadata = {k: v for k, v in entry.items() if k not in ('batch', 'days')}
        for day in entry['days']:
            path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            well_df = prepare_morphology_records(df, day, features, aggregation_level="well")
            if well_df.empty:
                continue

            well_df = well_df.copy()
            well_df['Batch'] = batch
            well_df['Day'] = day
            for key, value in batch_metadata.items():
                well_df[key] = value

            keep_columns = (
                ['Batch', 'Day', 'Image_ID', 'Well_ID', 'Object_Count_In_Image']
                + list(batch_metadata.keys())
                + [f for f in features if f in well_df.columns]
            )
            keep_columns = [c for c in keep_columns if c in well_df.columns]
            records.append(well_df[keep_columns])

    if not records:
        print("No well-level data found for export_well_level_summary.")
        return pd.DataFrame()

    summary_df = pd.concat(records, ignore_index=True)
    out_path = os.path.join(plot_data_dir, "well_level_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"Well-level summary saved: {out_path} ({len(summary_df)} well/day rows)")
    return summary_df


def audit_outlier_exclusion_by_well(base_dir, project_dir, batches_complete_info, features, group_by="batch"):
    """Per well/day, how many organoids were detected vs. how many would
    survive outlier removal per the currently configured outlier_detection
    settings -- regardless of whether --remove_outliers is actually
    enabled for this run, the same "report what WOULD be flagged"
    convention plot_variability_and_outliers already uses. Confirms
    outlier removal targets individual organoids within a well, never the
    well itself outright: a well is never dropped as a unit, but its
    post-removal organoid count can quietly become very low or zero.

    Prints (and flags a row) whenever a well's post-removal count falls
    below LOW_ORGANOID_COUNT_WARNING_THRESHOLD -- such a well would
    otherwise silently vanish (if 0) or contribute a barely-supported mean
    (if just a few) to every downstream well-level aggregation with no
    visible trace. Organoid_Mode-aware: for Organoid_Mode == "single" wells
    (exactly 1 organoid by design), only N_After_Outlier_Removal == 0 is
    ever flagged -- N_Detected == 1 is normal there, not a low count.

    Saves plots/quality_control/outlier_exclusion_by_well.csv."""
    qc_dir = os.path.join(_plots_root(base_dir, project_dir), 'quality_control')
    os.makedirs(qc_dir, exist_ok=True)

    saved_outlier_removal = dict(_OUTLIER_REMOVAL)
    _OUTLIER_REMOVAL["enabled"] = True
    if not _OUTLIER_REMOVAL["features"]:
        _OUTLIER_REMOVAL["features"] = list(features)

    try:
        rows = []
        for entry in batches_complete_info:
            batch, label = entry['batch'], entry.get(group_by, entry['batch'])
            for day in entry['days']:
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
                if not os.path.exists(path):
                    continue
                raw_df = _ensure_metric_identity_columns(pd.read_csv(path))
                if raw_df.empty or 'Well_ID' not in raw_df.columns:
                    continue
                filtered_df = filter_by_diameter(raw_df.copy(), day)

                detected_counts = raw_df.groupby('Well_ID', dropna=False).size()
                after_counts = (
                    filtered_df.groupby('Well_ID', dropna=False).size()
                    if not filtered_df.empty else pd.Series(dtype=int)
                )
                modes_by_well = (
                    raw_df.groupby('Well_ID', dropna=False)['Organoid_Mode'].first()
                    if 'Organoid_Mode' in raw_df.columns else pd.Series(dtype=object)
                )
                for well_id, n_detected in detected_counts.items():
                    n_detected = int(n_detected)
                    n_after = int(after_counts.get(well_id, 0))
                    organoid_mode = modes_by_well.get(well_id, "single")
                    # Organoid_Mode == "single" wells have exactly 1 organoid
                    # by design -- N_Detected == 1 is normal there, so only
                    # losing that one organoid entirely (n_after == 0) is
                    # worth flagging. The LOW_ORGANOID_COUNT_WARNING_THRESHOLD
                    # floor only makes sense for "multiple" wells, where a
                    # much larger starting count dropping to a handful is a
                    # real signal.
                    low_count = (
                        n_after == 0 if organoid_mode == "single"
                        else n_after < LOW_ORGANOID_COUNT_WARNING_THRESHOLD
                    )
                    rows.append({
                        'Batch': batch, 'Group': label, 'Day': day, 'Well_ID': well_id,
                        'Organoid_Mode': organoid_mode,
                        'N_Detected': n_detected, 'N_After_Outlier_Removal': n_after,
                        'N_Excluded': n_detected - n_after,
                        'Pct_Excluded': _safe_pct(n_detected - n_after, n_detected),
                        'Low_Count_Warning': low_count,
                    })
                    if low_count:
                        print(
                            f"⚠️  {batch} Day {day} {well_id} ({organoid_mode}): only {n_after}/{n_detected} "
                            f"organoids would survive outlier removal."
                        )
    finally:
        _OUTLIER_REMOVAL.clear()
        _OUTLIER_REMOVAL.update(saved_outlier_removal)

    if not rows:
        print("No data found for outlier exclusion audit.")
        return pd.DataFrame()

    audit_df = pd.DataFrame(rows)
    out_path = os.path.join(qc_dir, "outlier_exclusion_by_well.csv")
    audit_df.to_csv(out_path, index=False)
    print(f"Outlier exclusion audit saved: {out_path}")
    return audit_df


def _collect_well_metric_records(base_dir, project_dir, batches_complete_info, metric_column, csv_filename, group_by="batch"):
    """Gather one row per well/image for ``metric_column`` out of
    ``csv_filename`` (a per-well CSV under each day/batch's
    ``inference_metrics/`` folder -- e.g. ``well_metrics_summary.csv``'s
    ``Total_Segmented_Area_um2``), tagged with ``Batch``/``Group``/``Day``
    for plotting.

    Unlike ``prepare_morphology_records``/``mask_metrics.csv``, these CSVs
    are already one row per well/image (never per-object), so there is no
    aggregation step here -- just collection across days/batches.
    """
    records = []
    for entry in batches_complete_info:
        batch = entry['batch']
        label = entry.get(group_by, batch)
        for day in entry['days']:
            path = os.path.join(
                base_dir, project_dir, 'processed_data', f'Day {day}', batch,
                'inference_metrics', csv_filename,
            )
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if metric_column not in df.columns:
                continue
            df = df[[metric_column]].copy()
            df['Batch'] = batch
            df['Group'] = label
            df['Day'] = day
            records.append(df)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _pairwise_group_tests_by_day(df_all, metric_column, min_wells_for_stats=2):
    """Welch's t-test for every pair of groups present on each day, with
    Benjamini-Hochberg FDR correction applied across ALL of those tests
    together (not per day) -- consistent with ``plot_feature_grid_scatters``'s
    existing convention elsewhere in this module.

    A pair is skipped (not tested) when either group has fewer than
    ``min_wells_for_stats`` samples that day (default 2, the bare minimum
    Welch's t-test/Hedges' g can even compute) -- recorded as its own
    entry with ``Skipped_Insufficient_N=True`` and p/g left NaN, and
    printed, rather than just vanishing from the output with no trace.
    This is what lets a reader tell "not significant" apart from "never
    tested" by looking at the output alone. Callers that draw significance
    brackets should filter these out first (``[s for s in stats_list if
    not s['Skipped_Insufficient_N']]``); callers writing a stats CSV should
    keep them so the skip is visible there too.

    Returns a list of dicts with ``Day``, ``pair``, ``p_raw``, ``p_corr``,
    ``stars``, ``n1``, ``n2``, ``Skipped_Insufficient_N``."""
    stats_list = []
    skipped_list = []
    for day in sorted(df_all['Day'].dropna().unique()):
        day_df = df_all[df_all['Day'] == day]
        present_groups = sorted(day_df['Group'].dropna().unique())
        for i in range(len(present_groups)):
            for j in range(i + 1, len(present_groups)):
                g1, g2 = present_groups[i], present_groups[j]
                vals1 = day_df[day_df['Group'] == g1][metric_column].dropna()
                vals2 = day_df[day_df['Group'] == g2][metric_column].dropna()
                n1, n2 = len(vals1), len(vals2)
                if n1 >= min_wells_for_stats and n2 >= min_wells_for_stats:
                    _, p_val = ttest_ind(vals1, vals2, equal_var=False)
                    stats_list.append({
                        'Day': day, 'pair': (g1, g2), 'p_raw': p_val,
                        'hedges_g': _hedges_g(vals1, vals2),
                        'n1': n1, 'n2': n2, 'Skipped_Insufficient_N': False,
                    })
                elif n1 >= 1 and n2 >= 1:
                    print(
                        f"Skipped {g1} vs {g2} on Day {day}: insufficient wells "
                        f"(n={n1}, {n2}) below min_wells_for_stats={min_wells_for_stats}."
                    )
                    skipped_list.append({
                        'Day': day, 'pair': (g1, g2), 'p_raw': np.nan, 'p_corr': np.nan,
                        'stars': 'skipped', 'hedges_g': np.nan,
                        'n1': n1, 'n2': n2, 'Skipped_Insufficient_N': True,
                    })

    if stats_list:
        # ttest_ind returns p_raw=nan when both groups have zero variance
        # (e.g. every well has exactly 1 organoid on an early single-
        # organoid-per-image day) -- statsmodels' multipletests propagates
        # a single nan through its whole sort-and-correct procedure,
        # silently turning EVERY OTHER day's p_corr to nan too (confirmed:
        # this previously wiped out a p_raw=1e-12 result down to "ns").
        # Excluding nan p-values from the correction array (they carry no
        # real evidence either way -- identical, zero-spread data isn't a
        # significant difference by definition) keeps every other test's
        # correction correct regardless of how many nan entries exist.
        for s in stats_list:
            s['p_corr'] = np.nan
        testable_idx = [idx for idx, s in enumerate(stats_list) if not np.isnan(s['p_raw'])]
        if testable_idx:
            raw_pvals = [stats_list[idx]['p_raw'] for idx in testable_idx]
            _, corrected, _, _ = multipletests(raw_pvals, method='fdr_bh')
            for idx, p_corr in zip(testable_idx, corrected):
                stats_list[idx]['p_corr'] = p_corr
        for s in stats_list:
            p = s['p_corr']
            s['stars'] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    return stats_list + skipped_list


def plot_well_metric_over_time(
    base_dir, project_dir, batches_complete_info, metric_column, metric_label,
    csv_filename="well_metrics_summary.csv", group_by="batch", output_prefix=None, error_bar="sem",
    transition_days=None, min_wells_for_stats=2,
):
    """Line plot (mean +/- error_bar -- "sem" by default, or "sd"/"ci"; see
    _resolve_error_bar_values -- set via plotting_metadata.json's
    "line_plot_error_bar") of a per-well/image metric over time, with
    pairwise Welch's t-tests (Benjamini-Hochberg FDR corrected) between
    every pair of groups present on each day.

    Works for any per-well CSV under ``inference_metrics/`` (one row per
    ``Image_ID``/well) -- in particular ``"Total_Segmented_Area_um2"`` from
    ``well_metrics_summary.csv`` (the well's total segmented area, robust to
    touching organoids being merged into one connected component, since it
    doesn't depend on individual object identity at all -- see
    ``_summarize_metrics_by_image``). Per-object CSVs like
    ``light_dark_metrics.csv`` are NOT valid here (one row per organoid, not
    per well) -- see ``plot_feature_over_time_by_window`` instead.

    Saves the per-well records, the per-group/day summary, and (if any
    pairwise test could be run) the FDR-corrected test results as CSVs
    under ``plots/plot_data/``, and the figure under ``plots/``.
    """
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    df_all = _collect_well_metric_records(base_dir, project_dir, batches_complete_info, metric_column, csv_filename, group_by)
    if df_all.empty:
        print(f"No data found for {metric_column} in {csv_filename}.")
        return pd.DataFrame()

    prefix = output_prefix or metric_column
    df_all.to_csv(os.path.join(plot_data_dir, f"{prefix}_{group_by}_over_time_records.csv"), index=False)

    summary = df_all.groupby(['Group', 'Day'])[metric_column].agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['error_value'] = _resolve_error_bar_values(summary, error_bar)
    summary.to_csv(os.path.join(plot_data_dir, f"{prefix}_{group_by}_over_time_summary.csv"), index=False)

    unique_groups = sorted(df_all['Group'].dropna().unique())
    stats_list = (
        _pairwise_group_tests_by_day(df_all, metric_column, min_wells_for_stats=min_wells_for_stats)
        if len(unique_groups) > 1 else []
    )
    if stats_list:
        pd.DataFrame(stats_list).to_csv(
            os.path.join(plot_data_dir, f"{prefix}_{group_by}_over_time_stats.csv"), index=False
        )
        print(f"\n--- FDR-corrected pairwise tests for {metric_label} ---")
        for s in stats_list:
            print(
                f"Day {s['Day']} {s['pair'][0]} vs {s['pair'][1]}: p_raw={s['p_raw']:.5f} "
                f"p_adj={s['p_corr']:.5f} ({s['stars']}), Hedges' g = {s['hedges_g']:.3f} (n={s['n1']} vs {s['n2']})"
            )

    # Plot only the most significant pair's stars per day, to keep a line
    # plot (unlike the box plots below, which have room for every pair)
    # readable. Skipped (insufficient-N) entries are excluded here -- they
    # have no p_corr to compare and would otherwise render as "g=nan
    # skipped" on the figure; they're still in the CSV above.
    day_best_stats = {}
    for s in stats_list:
        if s['Skipped_Insufficient_N']:
            continue
        if s['Day'] not in day_best_stats or s['p_corr'] < day_best_stats[s['Day']]['p_corr']:
            day_best_stats[s['Day']] = s

    palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    for group in unique_groups:
        group_df = summary[summary['Group'] == group]
        ax.errorbar(
            group_df['Day'], group_df['mean'], yerr=group_df['error_value'],
            label=group, marker='o', capsize=4, color=palette[group], linewidth=2,
        )
    for day, s in day_best_stats.items():
        day_sum = summary[summary['Day'] == day]
        if day_sum.empty:
            continue
        y_pos = (day_sum['mean'] + day_sum['error_value']).max()
        ax.text(day, y_pos * 1.05, _stat_annotation_text(s), ha='center', va='bottom', color='black', fontsize=8, fontweight='bold')

    ax.set_xlabel('Day')
    ax.set_ylabel(metric_label)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _annotate_transitions(ax, transition_days)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"{prefix}_{group_by}_over_time.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")

    return df_all


def plot_well_metric_boxplots_by_group(
    base_dir, project_dir, days, batches_complete_info, metric_column, metric_label,
    csv_filename="well_metrics_summary.csv", group_by="batch", output_prefix=None,
    min_wells_for_stats=2,
):
    """Per-day box plots comparing groups for a per-well/image metric, with
    pairwise Welch's t-test significance brackets (Benjamini-Hochberg FDR
    corrected across all day x pair tests) via ``statannotations``.

    See ``plot_well_metric_over_time`` for the companion line-plot version
    and valid ``csv_filename``/``metric_column`` combinations (e.g.
    ``"Total_Segmented_Area_um2"`` from ``well_metrics_summary.csv``).
    """
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    df_all = _collect_well_metric_records(base_dir, project_dir, batches_complete_info, metric_column, csv_filename, group_by)
    if df_all.empty:
        print(f"No data found for {metric_column} in {csv_filename}.")
        return pd.DataFrame()

    prefix = output_prefix or metric_column
    plot_days = [day for day in days if day in set(df_all['Day'])]
    if not plot_days:
        print(f"None of the requested days have {metric_column} data.")
        return pd.DataFrame()

    unique_groups = sorted(df_all['Group'].dropna().unique())
    palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

    stats_list = _pairwise_group_tests_by_day(
        df_all[df_all['Day'].isin(plot_days)], metric_column, min_wells_for_stats=min_wells_for_stats
    )
    if stats_list:
        pd.DataFrame(stats_list).to_csv(
            os.path.join(plot_data_dir, f"{prefix}_{group_by}_boxplot_stats.csv"), index=False
        )

    n_cols = len(plot_days)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 7), sharey=True)
    sns.set_style("whitegrid")
    if n_cols == 1:
        axes = [axes]

    for d_idx, day in enumerate(plot_days):
        ax = axes[d_idx]
        day_df = df_all[df_all['Day'] == day]
        present_groups = [group for group in unique_groups if group in set(day_df['Group'])]
        if day_df.empty or not present_groups:
            ax.axis('off')
            continue

        sns.boxplot(
            data=day_df, x='Group', y=metric_column, order=present_groups,
            palette=palette, hue='Group', legend=False, ax=ax, showfliers=False,
        )
        sns.stripplot(
            data=day_df, x='Group', y=metric_column, order=present_groups,
            color='black', alpha=0.4, size=4, ax=ax,
        )

        day_pairs = [
            (s['pair'], s['p_corr']) for s in stats_list
            if s['Day'] == day and not s['Skipped_Insufficient_N']
            and s['pair'][0] in present_groups and s['pair'][1] in present_groups
        ]
        if day_pairs:
            annotator = Annotator(
                ax, [pair for pair, _ in day_pairs], data=day_df,
                x='Group', y=metric_column, order=present_groups,
            )
            annotator.configure(
                test=None, text_format='star', loc='inside', verbose=False,
                comparisons_correction=None,
            )
            annotator.set_pvalues_and_annotate([p for _, p in day_pairs])

        ax.set_title(f"Day {day}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(metric_label if d_idx == 0 else "")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"{prefix}_{group_by}_boxplots_by_day.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")

    return df_all


def _get_detection_runs(batch_metadata):
    if "get_segmentation_runs" in globals():
        return get_segmentation_runs(batch_metadata)

    runs = []
    if batch_metadata.get("days_96"):
        runs.append({
            "name": "days_96",
            "method": "dl",
            "days": batch_metadata["days_96"],
            "organoid_mode": "single",
            "num_plates": batch_metadata.get("num_plates", 1),
            "stitched_images_per_batch": 24,
            "images_per_plate": 24,
        })
    if batch_metadata.get("days_24"):
        runs.append({
            "name": "days_24",
            "method": "cl",
            "days": batch_metadata["days_24"],
            "organoid_mode": "single",
            "num_plates": batch_metadata.get("num_plates", 1),
            "stitched_images_per_batch": 24,
            "images_per_plate": 24,
        })
    return runs


def _read_metrics(base_dir, project_dir, day, batch):
    metrics_csv = os.path.join(
        base_dir,
        project_dir,
        'processed_data',
        f'Day {day}',
        batch,
        'inference_metrics',
        'mask_metrics.csv',
    )
    if not os.path.exists(metrics_csv):
        return pd.DataFrame(), metrics_csv
    return pd.read_csv(metrics_csv), metrics_csv


def _list_file_ids(folder, extensions):
    if not os.path.isdir(folder):
        return []
    extensions = tuple(ext.lower() for ext in extensions)
    ids = []
    for filename in os.listdir(folder):
        if filename.startswith("."):
            continue
        if filename.lower().endswith(extensions):
            ids.append(os.path.splitext(filename)[0])
    return sorted(set(ids))


def _format_expected_tile_ids(count):
    return [f"Tile{idx:06d}" for idx in range(1, int(count) + 1)]


def _expected_image_ids(base_dir, project_dir, day, batch, run):
    raw_dir = os.path.join(base_dir, project_dir, 'raw_data', f'Day {day}', batch)
    image_ids = _list_file_ids(raw_dir, ('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    if image_ids:
        return image_ids, "raw_data"

    mask_dir = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_npy')
    image_ids = _list_file_ids(mask_dir, ('.npy',))
    if image_ids:
        return image_ids, "inference_npy"

    if run.get("organoid_mode") == "single" and run.get("method") == "cl":
        images_per_plate = run.get("images_per_plate", run.get("stitched_images_per_batch", 24))
        expected_count = int(run.get("num_plates", 0)) * int(images_per_plate)
        if expected_count > 0:
            return _format_expected_tile_ids(expected_count), "metadata"

    return [], "unknown"


def _image_id_from_organoid_id(organoid_id):
    return str(organoid_id).split("_object", 1)[0]


def _metric_image_ids(metrics_df):
    if metrics_df.empty:
        return set()
    if 'Image_ID' in metrics_df.columns:
        return set(metrics_df['Image_ID'].dropna().astype(str))
    return set(_image_id_from_organoid_id(organoid_id) for organoid_id in metrics_df['Organoid_ID'].astype(str))


def create_detection_analysis(base_dir, project_dir, batches_complete_info, batches_info, group_by="batch"):
    """
    Report detected, undetected, and detected-but-filtered organoids.

    For single-organoid images, expected objects are inferred from raw image IDs,
    saved masks, or classical 24-well metadata. For multiple-organoid images, the
    expected object count is not knowable from metadata, so the report focuses on
    detected and not-considered objects and separately tracks expected images.
    """
    qc_dir = os.path.join(_plots_root(base_dir, project_dir), 'quality_control')
    batch_matrix_dir = os.path.join(qc_dir, 'batch_matrices')
    os.makedirs(batch_matrix_dir, exist_ok=True)

    info_lookup = {batch['batch']: batch for batch in batches_info}
    all_batch_summaries = []

    for entry in batches_complete_info:
        batch = entry['batch']
        label = entry.get(group_by, batch)
        batch_metadata = info_lookup.get(batch)

        if not batch_metadata:
            print(f"Metadata for {batch} not found in batches_info. Skipping.")
            continue

        for run in _get_detection_runs(batch_metadata):
            run_name = run.get("name", run.get("method", "run"))
            run_days = sorted(run.get("days", []))
            organoid_mode = run.get("organoid_mode", "single")
            status_by_day = {}
            expected_id_union = set()

            for day in run_days:
                metrics_df, metrics_csv = _read_metrics(base_dir, project_dir, day, batch)
                expected_ids, expected_source = _expected_image_ids(base_dir, project_dir, day, batch, run)
                expected_id_set = set(expected_ids)
                expected_id_union.update(expected_id_set)

                if metrics_df.empty:
                    metrics_df = pd.DataFrame(columns=['Organoid_ID'])
                elif 'Organoid_ID' not in metrics_df.columns:
                    metrics_df = metrics_df.copy()
                    metrics_df['Organoid_ID'] = [f"{batch}_day{day}_object{idx:04d}" for idx in range(1, len(metrics_df) + 1)]

                # "Considered" = has a valid computed Diameter_um metric --
                # an organoid whose metrics row is missing that column (e.g.
                # a malformed/older mask_metrics.csv) is detected but not
                # considered.
                if metrics_df.empty or 'Diameter_um' not in metrics_df.columns:
                    considered_df = pd.DataFrame()
                else:
                    considered_df = metrics_df.copy()

                if organoid_mode == "single":
                    detected_ids = set(metrics_df['Organoid_ID'].astype(str))
                    considered_ids = set(considered_df['Organoid_ID'].astype(str)) if not considered_df.empty else set()
                    if expected_id_set:
                        detected_ids = detected_ids & expected_id_set
                        considered_ids = considered_ids & expected_id_set

                    expected_objects = len(expected_id_set) if expected_id_set else np.nan
                    detected_count = len(detected_ids)
                    considered_count = len(considered_ids)
                    not_considered_count = max(0, detected_count - considered_count)
                    undetected_count = max(0, expected_objects - detected_count) if not pd.isna(expected_objects) else np.nan

                    if expected_id_set:
                        status = pd.Series("undetected", index=sorted(expected_id_set), dtype=object)
                        detected_not_considered = detected_ids - considered_ids
                        status.loc[sorted(detected_not_considered)] = "detected_not_considered"
                        status.loc[sorted(considered_ids)] = "considered"
                        status_by_day[day] = status
                else:
                    detected_count = len(metrics_df)
                    considered_count = len(considered_df)
                    not_considered_count = max(0, detected_count - considered_count)
                    expected_objects = np.nan
                    undetected_count = np.nan

                detected_image_ids = _metric_image_ids(metrics_df)
                considered_image_ids = _metric_image_ids(considered_df)
                expected_images = len(expected_id_set) if expected_id_set else np.nan
                undetected_images = (
                    max(0, expected_images - len(detected_image_ids & expected_id_set))
                    if expected_id_set else np.nan
                )

                excluded_count = not_considered_count
                if not pd.isna(undetected_count):
                    excluded_count += undetected_count

                all_batch_summaries.append({
                    'Batch': batch,
                    'Group': label,
                    'Day': day,
                    'Run': run_name,
                    'Method': run.get("method"),
                    'Organoid_Mode': organoid_mode,
                    'Expected_Source': expected_source,
                    'Expected_Images': expected_images,
                    'Detected_Images': len(detected_image_ids),
                    'Considered_Images': len(considered_image_ids),
                    'Undetected_Images': undetected_images,
                    'Expected_Objects': expected_objects,
                    'Detected': detected_count,
                    'Considered': considered_count,
                    'Not_Considered': not_considered_count,
                    'Undetected': undetected_count,
                    'Excluded_From_Analysis': excluded_count,
                    'Metrics_Path': metrics_csv,
                })

            if status_by_day:
                status_matrix = pd.DataFrame(status_by_day).reindex(sorted(expected_id_union))
                status_matrix.to_csv(os.path.join(batch_matrix_dir, f"{batch}_{run_name}_status_matrix.csv"))
                numeric_matrix = status_matrix.replace({
                    "undetected": 0,
                    "detected_not_considered": -1,
                    "considered": 1,
                })
                numeric_matrix.to_csv(os.path.join(batch_matrix_dir, f"{batch}_{run_name}_numeric_matrix.csv"))

    summary_df = pd.DataFrame(all_batch_summaries)
    if summary_df.empty:
        print("No detection QC data processed.")
        return None, qc_dir

    summary_df['Detection_Rate_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Detected'], row['Expected_Objects']),
        axis=1,
    )
    summary_df['Considered_Rate_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Considered'], row['Expected_Objects']),
        axis=1,
    )
    summary_df['Undetected_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Undetected'], row['Expected_Objects']),
        axis=1,
    )
    summary_df['Not_Considered_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Not_Considered'], row['Expected_Objects']),
        axis=1,
    )
    summary_df['Not_Considered_Of_Detected_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Not_Considered'], row['Detected']),
        axis=1,
    )
    summary_df['Excluded_From_Analysis_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Excluded_From_Analysis'], row['Expected_Objects']),
        axis=1,
    )

    summary_df.to_csv(os.path.join(qc_dir, "detection_qc_by_batch_day.csv"), index=False)

    def sum_with_nan(series):
        return series.sum(min_count=1)

    global_summary = summary_df.groupby(['Group', 'Day'], dropna=False).agg({
        'Expected_Images': sum_with_nan,
        'Detected_Images': sum_with_nan,
        'Considered_Images': sum_with_nan,
        'Undetected_Images': sum_with_nan,
        'Expected_Objects': sum_with_nan,
        'Detected': 'sum',
        'Considered': 'sum',
        'Not_Considered': 'sum',
        'Undetected': sum_with_nan,
        'Excluded_From_Analysis': sum_with_nan,
    }).reset_index()

    global_summary['Total_Expected'] = global_summary['Expected_Objects']
    global_summary['Detection_Rate_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Detected'], row['Expected_Objects']),
        axis=1,
    )
    global_summary['Considered_Rate_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Considered'], row['Expected_Objects']),
        axis=1,
    )
    global_summary['Undetected_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Undetected'], row['Expected_Objects']),
        axis=1,
    )
    global_summary['Not_Considered_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Not_Considered'], row['Expected_Objects']),
        axis=1,
    )
    global_summary['Not_Considered_Of_Detected_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Not_Considered'], row['Detected']),
        axis=1,
    )
    global_summary['Excluded_From_Analysis_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Excluded_From_Analysis'], row['Expected_Objects']),
        axis=1,
    )

    global_summary.to_csv(os.path.join(qc_dir, "detection_qc_global_summary.csv"), index=False)
    global_summary.to_csv(os.path.join(qc_dir, "undetected_tiles_global_summary.csv"), index=False)

    return global_summary, qc_dir


def plot_undetected_summary(global_summary, output_dir):
    """
    Plot detection QC percentages from create_detection_analysis.

    The plot separates true undetected objects from detected objects that were
    later excluded by morphology filters.
    """
    if global_summary is None or global_summary.empty:
        print("No detection QC summary available for plotting.")
        return

    percentage_columns = {
        'Undetected_Pct': 'Undetected (% expected)',
        'Not_Considered_Of_Detected_Pct': 'Not considered (% detected)',
        'Excluded_From_Analysis_Pct': 'Excluded from analysis (% expected)',
    }
    available_columns = [
        column for column in percentage_columns
        if column in global_summary.columns and not global_summary[column].isna().all()
    ]

    if not available_columns:
        print("No percentage columns available for detection QC plotting.")
        return

    plot_df = global_summary.melt(
        id_vars=['Group', 'Day'],
        value_vars=available_columns,
        var_name='QC_Metric',
        value_name='Percent',
    ).dropna(subset=['Percent'])
    plot_df['QC_Metric'] = plot_df['QC_Metric'].map(percentage_columns)

    plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")
    ax = sns.lineplot(
        data=plot_df,
        x='Day',
        y='Percent',
        hue='Group',
        style='QC_Metric',
        markers=True,
        dashes=False,
        palette="husl",
        linewidth=2,
    )

    plt.ylabel('Organoids (%)')
    plt.xlabel('Day')
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "detection_qc_percentage_plot.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    undetected_only = global_summary.dropna(subset=['Undetected_Pct'])
    if not undetected_only.empty:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        ax = sns.lineplot(
            data=undetected_only,
            x='Day',
            y='Undetected_Pct',
            hue='Group',
            palette="husl",
            marker='o',
            linewidth=2,
        )
        plt.ylabel('Undetected Organoids (%)')
        plt.xlabel('Day')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "undetected_tiles_percentage_plot.png"), bbox_inches='tight', dpi=300)
        plt.close()


def _fit_well_growth_rates(base_dir, project_dir, batches_complete_info, feature, group_by, min_points):
    """Shared by analyze_growth_trajectories and
    analyze_growth_rate_by_two_factors -- loads each well's full timecourse
    for ``feature`` and fits a per-well exponential growth model
    area(t) = area_0 * exp(k*t) (see analyze_growth_trajectories's own
    docstring for the full Well_ID/Organoid_Mode-averaging rationale, which
    applies identically here since this is the same fit, just factored out
    so a second caller can reuse it without duplicating ~50 lines).

    Returns (df_all, growth_df); writes no CSVs and prints nothing --
    callers own their own output naming/side effects."""
    records = []
    for entry in batches_complete_info:
        batch, label = entry['batch'], entry.get(group_by, entry['batch'])
        for day in entry['days']:
            path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            df = _ensure_metric_identity_columns(df)
            if df.empty or feature not in df.columns:
                continue

            df = filter_by_diameter(df, day)
            if df.empty:
                continue

            # One row per well already for organoid_mode == "single" (mean
            # is then a no-op safety net). For organoid_mode == "multiple",
            # this is a genuine per-day mean across every organoid detected
            # in that well's image -- see analyze_growth_trajectories's own
            # docstring for why the well identity stays trackable even
            # though individual organoids within it are not. Object_Count_
            # In_Image and Organoid_Mode are carried through per well/day
            # so the two cases remain distinguishable in the saved records.
            day_df = df.groupby('Well_ID', dropna=False)[feature].mean().reset_index()
            counts = df.groupby('Well_ID', dropna=False).size().reset_index(name='Object_Count_In_Image')
            day_df = day_df.merge(counts, on='Well_ID', how='left')
            if 'Organoid_Mode' in df.columns:
                modes = df.groupby('Well_ID', dropna=False)['Organoid_Mode'].first().reset_index()
                day_df = day_df.merge(modes, on='Well_ID', how='left')
            day_df['Batch'] = batch
            day_df['Group'] = label
            day_df['Day'] = day
            records.append(day_df)

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.concat(records, ignore_index=True)

    growth_rows = []
    for (batch, well_id, group), well_df in df_all.groupby(['Batch', 'Well_ID', 'Group']):
        well_df = well_df.dropna(subset=[feature]).sort_values('Day')
        well_df = well_df[well_df[feature] > 0]
        if len(well_df) < min_points:
            continue

        day_vals = well_df['Day'].values.astype(float)
        log_vals = np.log(well_df[feature].values)
        slope, intercept = np.polyfit(day_vals, log_vals, 1)
        predicted = slope * day_vals + intercept
        ss_res = np.sum((log_vals - predicted) ** 2)
        ss_tot = np.sum((log_vals - log_vals.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        growth_rows.append({
            'Batch': batch, 'Well_ID': well_id, 'Group': group,
            'N_Days': len(well_df),
            'First_Day': day_vals.min(), 'Last_Day': day_vals.max(),
            'Growth_Rate_Per_Day': slope,
            'Doubling_Time_Days': np.log(2) / slope if slope > 0 else np.nan,
            'R_Squared': r_squared,
            f'{feature}_Start': well_df[feature].iloc[0],
            f'{feature}_End': well_df[feature].iloc[-1],
        })

    return df_all, pd.DataFrame(growth_rows)


def analyze_growth_trajectories(
    base_dir, project_dir, batches_complete_info, feature="Area_um2",
    metric_label="Area (um^2)", group_by="batch", min_points=3,
    transition_days=None, min_wells_for_stats=2, show_pairwise_stats=True,
):
    """
    Track each physical well across its full timecourse and estimate a
    per-well growth rate.

    Well_ID (derived from fixed tile-position arithmetic in
    _get_image_measurement_context -- Plate_Index/Well_Index come from the
    tile number modulo images_per_plate, not from a sequential counter over
    however many images happened to be detected that day) reliably ties the
    same physical well to itself across days, for both Organoid_Mode
    values. What differs between the two modes is what "the well's value"
    on a given day means: for Organoid_Mode == "single" it is that one
    organoid's own measurement; for Organoid_Mode == "multiple" (crowded
    multi-organoid wells, where individual organoids have no stable
    identity from one day to the next) it is the *mean* of that day's
    per-organoid measurements across every organoid detected in that well's
    image. Averaging over the well's organoids on each day sidesteps the
    lack of individual-organoid identity -- the well itself is still a
    trackable unit even when the organoids inside it are not -- so a
    per-well growth rate can be fit for "multiple" wells too, just on the
    well's mean-organoid trajectory rather than a single organoid's. Both
    modes are pooled into the same fit below; the per-well output rows
    carry Organoid_Mode and Object_Count_In_Image (organoids averaged that
    day; always 1 for single mode) so the two cases stay distinguishable
    downstream.

    For each Batch x Well_ID with at least ``min_points`` days of data, fits
    an exponential growth model area(t) = area_0 * exp(k*t) via a linear
    regression of log(feature) on Day -- the standard approach for
    organoid/spheroid growth curves -- and reports the growth rate,
    doubling time (ln(2)/k), and fit quality (R^2).

    Saves the per-well timecourse and growth-rate summary as CSVs under
    plots/plot_data/, plus two figures: individual well trajectories
    ("spaghetti plot") with the group mean overlaid, and a per-well growth
    rate box/strip plot by group (with a Welch's t-test + Hedges' g if
    exactly two groups are present).

    ``show_pairwise_stats=False`` still computes and prints every pairwise
    Welch's t-test/Hedges' g (BH-FDR corrected) to the console, but skips
    drawing them as brackets on the violin plot -- for group_by columns
    with enough categories (e.g. a synthetic two-factor "condition x lab"
    label) that an every-pair bracket for each one would stack far past the
    top of the plot and become unreadable rather than informative.
    """
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    df_all, growth_df = _fit_well_growth_rates(base_dir, project_dir, batches_complete_info, feature, group_by, min_points)
    if df_all.empty:
        print(f"No data found for growth trajectory analysis of {feature}.")
        return pd.DataFrame(), pd.DataFrame()

    df_all.to_csv(os.path.join(plot_data_dir, f"{feature}_growth_trajectories_records.csv"), index=False)

    if not growth_df.empty:
        growth_df.to_csv(os.path.join(plot_data_dir, f"{feature}_growth_rate_summary.csv"), index=False)
        print(f"Growth rate summary ({feature}): {len(growth_df)} wells with >= {min_points} timepoints.")
    else:
        print(f"No wells had >= {min_points} timepoints for growth rate fitting ({feature}).")

    unique_groups = sorted(df_all['Group'].dropna().unique())
    palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

    # --- Plot 1: individual trajectories ("spaghetti plot") + group mean ---
    plot_unit_scale = AREA_UM2_TO_MM2 if feature == "Area_um2" else 1.0
    plot_metric_label = _um2_label_to_mm2(metric_label) if feature == "Area_um2" else metric_label
    fig, ax = plt.subplots(figsize=(STANDARD_FIG_WIDTH_IN, STANDARD_FIG_HEIGHT_IN))
    sns.set_style("whitegrid")
    for group in unique_groups:
        group_df = df_all[df_all['Group'] == group]
        for _, well_df in group_df.groupby(['Batch', 'Well_ID']):
            well_df = well_df.sort_values('Day')
            ax.plot(well_df['Day'], well_df[feature] * plot_unit_scale, color=palette[group], alpha=0.1, linewidth=0.5, zorder=0.5)

        group_summary = group_df.groupby('Day')[feature].mean().reset_index()
        ax.plot(group_summary['Day'], group_summary[feature] * plot_unit_scale, color=palette[group], linewidth=1, marker='o', label=group, zorder=2, markersize=2)

    ax.set_xlabel('Day')
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(plot_metric_label)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _annotate_transitions(ax, transition_days)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"{feature}_growth_trajectories_{group_by}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")

    # --- Plot 2: per-well growth rate by group ---
    if not growth_df.empty:
        # Same dimensions/legend approach as analyze_expansion_rate_by_window's
        # boxplot grid: a bottom-reserved margin for a color legend instead of
        # an inline one, and x-tick labels hidden entirely (the legend is
        # enough to identify each group -- see the identical choice there).
        # Width is half of plot_feature_grid_scatters's (STANDARD_FIG_WIDTH_IN
        # split across MAX_PANEL_COLS panels), matching one grid-scatterplot
        # column's width per explicit instruction -- this figure has only one
        # "panel" (all groups in a single axes), so it doesn't need the full
        # standard width the multi-day/window figures share. LEGEND_RESERVE_IN
        # matches plot_feature_grid_scatters's own value exactly (rather than
        # analyze_expansion_rate_by_window's larger one) for full height
        # consistency with one of that figure's own panels.
        LEGEND_RESERVE_IN = 0.6
        fig = plt.figure(figsize=(STANDARD_FIG_WIDTH_IN*3 / MAX_PANEL_COLS, STANDARD_FIG_HEIGHT_IN))
        bottom_frac = LEGEND_RESERVE_IN / (STANDARD_FIG_HEIGHT_IN + LEGEND_RESERVE_IN)
        gs = fig.add_gridspec(1, 1, top=0.95, bottom=bottom_frac, hspace=0.15, wspace=0.3)
        sns.set_style("whitegrid")
        ax = fig.add_subplot(gs[0, 0])
        sns.violinplot(
            data=growth_df, x='Group', y='Growth_Rate_Per_Day', order=unique_groups,
            palette=palette, hue='Group', legend=False, ax=ax,
            inner=None, cut=0, alpha=0.35, linewidth=0.8, zorder=1,
        )
        # Colored by group (not a fixed black) and jittered, matching
        # plot_feature_grid_scatters's own stripplot exactly.
        sns.stripplot(
            data=growth_df, x='Group', y='Growth_Rate_Per_Day', order=unique_groups,
            palette=palette, hue='Group', jitter=True, alpha=0.8, s=2, ax=ax,
            legend=False, zorder=2,
        )

        # Mean lines -- same style as plot_feature_grid_scatters's.
        means = growth_df.groupby('Group')['Growth_Rate_Per_Day'].mean().reindex(unique_groups)
        for x_pos, mean_val in enumerate(means):
            ax.hlines(y=mean_val, xmin=x_pos - 0.25, xmax=x_pos + 0.25, color="black", lw=1, zorder=5)

        if len(unique_groups) >= 2:
            # Reuses _pairwise_group_tests_by_day (all-pairs Welch's t-test +
            # Hedges' g, one shared FDR correction) against a constant fake
            # "Day" column -- growth_df has no day dimension of its own (one
            # rate per well already), but the all-pairs-plus-FDR logic is
            # otherwise identical to every other group-comparison figure in
            # this module, so it's reused rather than duplicated. Handles
            # any number of groups, not just exactly two (see boxplots_by_window
            # for the same stacked-bracket convention with 3+ groups).
            pair_stats = _pairwise_group_tests_by_day(
                growth_df.assign(Day=0), 'Growth_Rate_Per_Day', min_wells_for_stats=min_wells_for_stats
            )
            # Skipped (insufficient-N) entries are already reported by
            # _pairwise_group_tests_by_day itself -- excluded here so they
            # don't also get a NaN-valued bracket drawn on the figure below.
            pair_stats = [s for s in pair_stats if not s['Skipped_Insufficient_N']]
            for s in pair_stats:
                print(
                    f"Growth rate ({feature}), {s['pair'][0]} vs {s['pair'][1]}: "
                    f"p_corr = {s['p_corr']:.5f} ({s['stars']}), "
                    f"Hedges' g = {s['hedges_g']:.3f} (n={s['n1']} vs {s['n2']} wells)"
                )

            y_max = growth_df['Growth_Rate_Per_Day'].max()
            y_offset = abs(y_max) * 0.15 if y_max else 0.01
            for s_idx, s in enumerate(pair_stats if show_pairwise_stats else []):
                g1, g2 = s['pair']
                x1, x2 = unique_groups.index(g1), unique_groups.index(g2)
                h = y_max + (s_idx * y_offset)
                ax.plot([x1, x1, x2, x2], [h, h + y_offset * 0.2, h + y_offset * 0.2, h], lw=0.5, color='black')
                ax.text((x1 + x2) * 0.5, h + y_offset * 0.2, _stat_annotation_text(s),
                        ha='center', va='bottom', fontsize=stats_fontsize)

        ax.set_xlabel("")
        ax.set_ylabel(f"Growth rate ({metric_label}/day)")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelbottom=False)
        # No frame -- same spine-hiding as plot_feature_grid_scatters's
        # panels (only the left spine stays, for the y-axis).
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        group_handles = {
            group: mlines.Line2D(
                [], [], color=palette[group], marker='o',
                linestyle='None', markersize=10, label=group
            )
            for group in unique_groups
        }
        handles = [group_handles[k] for k in sorted(group_handles.keys())]
        # Wrapped to at most 2 columns (unlike the multi-panel figures'
        # single-row legend) -- this figure is only half their width, so a
        # single row of every group's full name would force savefig's
        # bbox_inches='tight' to widen the saved image well past the
        # intended half-width to fit the legend.
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                   ncol=min(len(handles), 4), frameon=False, title=None)

        save_path = os.path.join(plot_dir, f"{feature}_growth_rate_by_{group_by}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_path}")

    return df_all, growth_df


def analyze_growth_rate_by_two_factors(
    base_dir, project_dir, batches_complete_info, feature="Area_um2",
    metric_label="Area (um^2)", primary_factor="clone", secondary_factor="lab",
    min_points=3, min_wells_for_stats=2,
):
    """
    Per-well growth-rate violin comparing two grouping variables at once,
    as a dodge-style grouped violin: ``primary_factor`` along the x-axis
    (e.g. clone), ``secondary_factor`` as a hue-dodged sub-violin within
    each x-axis category (e.g. LabA/LabB side by side within each clone).

    Reuses _fit_well_growth_rates -- the same per-well exponential growth
    fit analyze_growth_trajectories itself uses -- so growth rates are
    identical either way; this function only differs in how the result is
    grouped/plotted (two independent factors at once, rather than one
    factor, or two factors concatenated into a single flat axis).

    A flat "all pairs" comparison across every (primary, secondary)
    combination grows combinatorially and stops being readable past a
    handful of categories (e.g. 4 clones x 2 labs = 28 pairs -- see
    analyze_growth_trajectories's own single-axis violin, which draws
    exactly that). Statistical testing here is restricted instead to the
    one question a 2-factor design like this actually asks directly: does
    secondary_factor (e.g. lab) shift the growth rate within each
    primary_factor category (e.g. clone)? That is exactly
    _pairwise_group_tests_by_day's own "every pair of groups present on
    each day" logic with the two factors swapped in for "group"/"day" --
    primary_factor's categories standing in for "day" and
    secondary_factor's two levels standing in for "group" -- reused as-is
    rather than reimplemented, which is also what gives BH-FDR correction
    across exactly those within-category comparisons (not against every
    other figure's tests, and not against the combinatorial full pair set
    this function deliberately avoids drawing). secondary_factor must have
    exactly 2 levels for this to be well-defined; skipped (insufficient-N)
    categories are reported by _pairwise_group_tests_by_day itself.

    Saves the per-well growth-rate summary (with both factor columns, one
    row per well) as a CSV under plots/plot_data/, plus the grouped violin
    figure.
    """
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    _, growth_df = _fit_well_growth_rates(
        base_dir, project_dir, batches_complete_info, feature, primary_factor, min_points,
    )
    if growth_df.empty:
        print(f"No wells had >= {min_points} timepoints for growth rate fitting ({feature}).")
        return growth_df

    growth_df = growth_df.rename(columns={'Group': primary_factor})
    batch_secondary = {b['batch']: b.get(secondary_factor, b['batch']) for b in batches_complete_info}
    growth_df[secondary_factor] = growth_df['Batch'].map(batch_secondary)

    secondary_levels = sorted(growth_df[secondary_factor].dropna().unique())
    if len(secondary_levels) != 2:
        print(
            f"analyze_growth_rate_by_two_factors requires exactly 2 {secondary_factor} levels, "
            f"found {len(secondary_levels)} ({secondary_levels}) -- skipping."
        )
        return growth_df

    primary_levels = sorted(growth_df[primary_factor].dropna().unique())
    growth_df.to_csv(
        os.path.join(plot_data_dir, f"{feature}_growth_rate_by_{primary_factor}_{secondary_factor}_summary.csv"),
        index=False,
    )
    print(f"Growth rate summary ({feature}, {primary_factor} x {secondary_factor}): {len(growth_df)} wells.")

    palette = dict(zip(secondary_levels, sns.color_palette("husl", len(secondary_levels))))

    LEGEND_RESERVE_IN = 0.6
    fig = plt.figure(figsize=(STANDARD_FIG_WIDTH_IN, STANDARD_FIG_HEIGHT_IN))
    bottom_frac = LEGEND_RESERVE_IN / (STANDARD_FIG_HEIGHT_IN + LEGEND_RESERVE_IN)
    gs = fig.add_gridspec(1, 1, top=0.95, bottom=bottom_frac, hspace=0.15, wspace=0.3)
    sns.set_style("whitegrid")
    ax = fig.add_subplot(gs[0, 0])

    VIOLIN_WIDTH = 0.8
    sns.violinplot(
        data=growth_df, x=primary_factor, y='Growth_Rate_Per_Day', order=primary_levels,
        hue=secondary_factor, hue_order=secondary_levels, palette=palette, dodge=True,
        width=VIOLIN_WIDTH, legend=False, ax=ax,
        inner=None, cut=0, alpha=0.35, linewidth=0.8, zorder=1,
    )
    # Colored by secondary_factor (not a fixed black) and jittered, matching
    # analyze_growth_trajectories's own stripplot exactly.
    sns.stripplot(
        data=growth_df, x=primary_factor, y='Growth_Rate_Per_Day', order=primary_levels,
        hue=secondary_factor, hue_order=secondary_levels, palette=palette, dodge=True,
        jitter=True, alpha=0.8, s=2, ax=ax, legend=False, zorder=2,
    )

    # Seaborn's own dodge offset for a 2-level hue at width=VIOLIN_WIDTH --
    # computed explicitly (the standard categorical-dodge formula: level i
    # of n sits at width * ((i - (n-1)/2) / n) from the category center)
    # rather than read back from the Axes, so mean lines/brackets land
    # exactly on each dodged sub-violin regardless of draw order.
    n_hues = len(secondary_levels)
    hue_offsets = {
        level: VIOLIN_WIDTH * ((i - (n_hues - 1) / 2) / n_hues)
        for i, level in enumerate(secondary_levels)
    }

    for p_idx, p_level in enumerate(primary_levels):
        for h_level, h_offset in hue_offsets.items():
            cell = growth_df[(growth_df[primary_factor] == p_level) & (growth_df[secondary_factor] == h_level)]
            if cell.empty:
                continue
            x_pos = p_idx + h_offset
            ax.hlines(
                y=cell['Growth_Rate_Per_Day'].mean(), xmin=x_pos - VIOLIN_WIDTH / n_hues * 0.4,
                xmax=x_pos + VIOLIN_WIDTH / n_hues * 0.4, color="black", lw=1, zorder=5,
            )

    # --- Within-primary_factor-category comparison of secondary_factor's
    # two levels only (e.g. LabA vs LabB within each clone) -- see
    # docstring above for why this isn't an all-pairs comparison.
    day_stub = growth_df.rename(columns={primary_factor: 'Day', secondary_factor: 'Group'})
    all_pair_stats = _pairwise_group_tests_by_day(day_stub, 'Growth_Rate_Per_Day', min_wells_for_stats=min_wells_for_stats)
    # Saved (skipped rows included, matching _pairwise_group_tests_by_day's
    # own convention elsewhere) before filtering, so this comparison has a
    # citable table -- previously computed for printing only.
    pd.DataFrame(all_pair_stats).rename(columns={'Day': primary_factor}).to_csv(
        os.path.join(plot_data_dir, f"{feature}_growth_rate_by_{primary_factor}_{secondary_factor}_stats.csv"),
        index=False,
    )
    pair_stats = [s for s in all_pair_stats if not s['Skipped_Insufficient_N']]
    for s in pair_stats:
        print(
            f"Growth rate ({feature}), {primary_factor}={s['Day']}, "
            f"{s['pair'][0]} vs {s['pair'][1]}: p_corr = {s['p_corr']:.5f} ({s['stars']}), "
            f"Hedges' g = {s['hedges_g']:.3f} (n={s['n1']} vs {s['n2']} wells)"
        )

    # Each bracket is placed above its own primary_factor category's own
    # max value (not a single shared/stacked height across every category
    # the way analyze_growth_trajectories's flat all-pairs version does) --
    # there is at most one bracket per category here, so nothing to stack.
    for s in pair_stats:
        p_level = s['Day']
        p_idx = primary_levels.index(p_level)
        x1, x2 = p_idx + hue_offsets[s['pair'][0]], p_idx + hue_offsets[s['pair'][1]]
        cell_vals = growth_df.loc[growth_df[primary_factor] == p_level, 'Growth_Rate_Per_Day']
        y_offset = abs(cell_vals.max()) * 0.15 if len(cell_vals) and cell_vals.max() else 0.01
        h = cell_vals.max() + y_offset
        ax.plot([x1, x1, x2, x2], [h, h + y_offset * 0.3, h + y_offset * 0.3, h], lw=0.5, color='black')
        ax.text((x1 + x2) * 0.5, h + y_offset * 0.3, _stat_annotation_text(s),
                ha='center', va='bottom', fontsize=stats_fontsize)

    ax.set_xlabel("")
    ax.set_ylabel(f"Growth rate ({metric_label}/day)")
    ax.grid(True, alpha=0.3)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend covers secondary_factor only (2 entries) -- primary_factor is
    # already directly readable off the x-axis tick labels, unlike
    # analyze_growth_trajectories's flat version which hides tick labels
    # entirely and relies on the legend for every category.
    handles = [
        mlines.Line2D([], [], color=palette[level], marker='o', linestyle='None', markersize=10, label=level)
        for level in secondary_levels
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=len(handles), frameon=False, title=None)

    save_path = os.path.join(plot_dir, f"{feature}_growth_rate_by_{primary_factor}_{secondary_factor}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")

    return growth_df


def plot_feature_correlation_heatmap(
    base_dir, project_dir, batches_complete_info, features, feature_labels=None,
    day=None, method="spearman",
):
    """
    Pairwise correlation heatmap across every requested morphology feature,
    pooling every batch/day's object-level mask_metrics.csv rows together
    (or a single day, if ``day`` is given).

    Spearman (the default) rather than Pearson, since several of these
    features are related by a deterministic but nonlinear monotonic
    transform (e.g. Aspect_Ratio and Eccentricity, both pure functions of
    the same major/minor-axis ratio), which Pearson can substantially
    underestimate -- the same convention used by
    _select_non_redundant_features to choose the PCA plot's feature subset,
    so this heatmap doubles as a visual explanation of that selection.
    """
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    feature_labels = feature_labels or features
    label_map = dict(zip(features, feature_labels))

    records = []
    for entry in batches_complete_info:
        batch = entry['batch']
        for d in entry['days']:
            if day is not None and d != day:
                continue
            path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {d}', batch, 'inference_metrics', 'mask_metrics.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                df = filter_by_diameter(df, d)
                if not df.empty:
                    records.append(df)

    if not records:
        print("No data available for feature correlation heatmap.")
        return pd.DataFrame()

    df_all = pd.concat(records, ignore_index=True)
    available = [f for f in features if f in df_all.columns]
    numeric_df = df_all[available].apply(pd.to_numeric, errors='coerce')
    corr = numeric_df.corr(method=method)

    suffix = f"_day{day}" if day is not None else "_all_days"
    corr.to_csv(os.path.join(plot_data_dir, f"feature_correlation_{method}{suffix}.csv"))

    display_labels = [label_map.get(f, f) for f in available]

    fig, ax = plt.subplots(figsize=(1.0 * len(available) + 2, 0.9 * len(available) + 1.5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
        xticklabels=display_labels, yticklabels=display_labels,
        square=True, cbar_kws={'label': f"{method.capitalize()} r"}, ax=ax,
    )
    n_organoids = len(numeric_df.dropna())
    title = f"Feature correlation ({method}, n={n_organoids} organoids)"
    if day is not None:
        title += f", Day {day}"
    ax.set_title(title)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"feature_correlation_heatmap_{method}{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")

    return corr


def _window_label(window):
    """Human-readable label for one developmental window (a list of days)."""
    if len(window) > 1:
        return f"Day {window[0]}-{window[-1]}"
    return f"Day {window[0]}"


def plot_feature_over_time_by_window(
    base_dir, project_dir, batches_complete_info, windows, feature, metric_label,
    group_by="batch", csv_filename="mask_metrics.csv", aggregation_level="object", error_bar="sem",
    transition_days=None,
):
    """
    Longitudinal line plot (mean +/- error_bar -- "sem" by default, or
    "sd"/"ci"; see _resolve_error_bar_values -- set via
    plotting_metadata.json's "line_plot_error_bar") of ``feature`` across
    every day in ``windows`` (a list of day-lists, e.g. developmental
    stages), with vertical dashed lines marking each window's start day --
    one continuous trajectory rather than one panel per window, so the
    whole developmental arc stays visible while still showing where each
    window begins.

    Reads ``csv_filename`` under each day/batch's ``inference_metrics/``
    folder -- ``mask_metrics.csv`` by default, but also works for
    ``optical_density_metrics.csv`` (Optical_Density/Core_Darkening_Index),
    since both share the same Batch/Day/Image_ID/Well_ID/Organoid_ID
    identity columns (and Area_um2/Diameter_um, for filter_by_diameter)
    that ``prepare_morphology_records`` expects.

    This only visualizes the trend -- see ``plot_feature_boxplots_by_window``
    for the statistical comparison between groups within each window.
    """
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    aggregation_level = str(aggregation_level).lower()
    if aggregation_level in {"organoid", "individual"}:
        aggregation_level = "object"
    if aggregation_level in {"image", "well_mean", "image_mean"}:
        aggregation_level = "well"

    all_days = sorted({day for window in windows for day in window})

    all_records = []
    for entry in batches_complete_info:
        batch, label = entry['batch'], entry.get(group_by)
        for day in entry['days']:
            if day not in all_days:
                continue
            path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', csv_filename)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df_filtered = prepare_morphology_records(df, day, [feature], aggregation_level=aggregation_level)
            if feature in df_filtered.columns and not df_filtered.empty:
                df_filtered = df_filtered.copy()
                df_filtered['Batch'] = batch
                df_filtered['Group'] = label
                df_filtered['Day'] = day
                all_records.append(df_filtered[['Batch', 'Group', 'Day', feature]])

    if not all_records:
        print(f"No data found for {feature} in {csv_filename} across the configured developmental windows.")
        return pd.DataFrame()

    df_all = pd.concat(all_records, ignore_index=True)
    df_all.to_csv(
        os.path.join(plot_data_dir, f"{feature}_{aggregation_level}_{group_by}_by_window_over_time_records.csv"),
        index=False,
    )

    summary = df_all.groupby(['Group', 'Day'])[feature].agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['error_value'] = _resolve_error_bar_values(summary, error_bar)
    summary.to_csv(
        os.path.join(plot_data_dir, f"{feature}_{aggregation_level}_{group_by}_by_window_over_time_summary.csv"),
        index=False,
    )

    unique_groups = sorted(df_all['Group'].dropna().unique())
    palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

    fig, ax = plt.subplots(figsize=(STANDARD_FIG_WIDTH_IN, STANDARD_FIG_HEIGHT_IN))
    sns.set_style("whitegrid")
    for group in unique_groups:
        data = summary[summary['Group'] == group]
        ax.errorbar(
            data['Day'], data['mean'], yerr=data['error_value'],
            label=group, marker='o', capsize=4, color=palette[group], linewidth=1, markersize=3,
        )

    for window in windows[1:]:
        ax.axvline(window[0], color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=0)

    # Same small-annotation convention as _annotate_transitions's own
    # rotated labels (fontsize=7, alpha=0.8) -- both are vertical-line +
    # text markers on this same axes, so they should read as one visual
    # "meta-annotation" style rather than two different sizes/weights.
    y_top = ax.get_ylim()[1]
    for window in windows:
        ax.text(
            window[0], y_top, _window_label(window),
            rotation=90, va='top', ha='right', fontsize=7, color='gray', alpha=0.8,
        )

    ax.set_xlabel('Day')
    ax.set_ylabel(metric_label)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _annotate_transitions(ax, transition_days)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"{feature}_{aggregation_level}_{group_by}_by_window_over_time.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")

    return df_all


def plot_feature_boxplots_by_window(
    base_dir, project_dir, batches_complete_info, windows, feature, metric_label,
    group_by="batch", csv_filename="mask_metrics.csv", min_wells_for_stats=2,
):
    """
    Grid of violin+strip plots comparing groups for ``feature``: one ROW per
    developmental window, one COLUMN per day within that window. Days are
    NOT pooled together -- each day is its own independent comparison (own
    stats, own panel); the window only decides which days share a row for
    side-by-side visual comparison (and, since ``sharey='row'``, a common
    y-axis scale within that row -- rows are not forced onto one shared
    scale, since absolute values can shift a lot between windows spanning
    different developmental stages). Windows with fewer days than the
    widest one get blank trailing panels.

    Pairwise Welch's t-tests (Benjamini-Hochberg FDR corrected across every
    day x pair test in the whole figure) and Hedges' g effect sizes -- the
    statistical counterpart to ``plot_feature_over_time_by_window``.

    Tests run on well-level means (``prepare_morphology_records(...,
    aggregation_level="well")``), independent of the object-level points
    shown -- the same pseudoreplication-safe convention as
    ``plot_feature_grid_scatters``/``plot_feature_over_time``. A no-op for
    Organoid_Mode == "single" data, where well-level already equals
    object-level.
    """
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    day_data = {}       # (w_idx, day) -> object-level df ['Group', feature]
    day_data_well = {}  # (w_idx, day) -> well-level df ['Group', feature]

    for w_idx, window in enumerate(windows):
        for day in window:
            obj_records = []
            well_records = []
            for entry in batches_complete_info:
                batch, label = entry['batch'], entry.get(group_by)
                if day not in entry['days']:
                    continue
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', csv_filename)
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)

                df_obj = prepare_morphology_records(df, day, [feature], aggregation_level="object")
                if not df_obj.empty and feature in df_obj.columns:
                    df_obj = df_obj.copy()
                    df_obj['Group'] = label
                    obj_records.append(df_obj[['Group', feature]])

                df_well = prepare_morphology_records(df, day, [feature], aggregation_level="well")
                if not df_well.empty and feature in df_well.columns:
                    df_well = df_well.copy()
                    df_well['Group'] = label
                    well_records.append(df_well[['Group', feature]])

            if obj_records:
                day_data[(w_idx, day)] = pd.concat(obj_records, ignore_index=True)
            if well_records:
                day_data_well[(w_idx, day)] = pd.concat(well_records, ignore_index=True)

    if not day_data:
        print(f"No data found for {feature} in {csv_filename} across the configured developmental windows.")
        return pd.DataFrame()

    stats_list = []
    skipped_stats_list = []
    for (w_idx, day), df_well in day_data_well.items():
        u_groups = sorted(df_well['Group'].dropna().unique())
        if len(u_groups) < 2:
            continue
        for i in range(len(u_groups)):
            for j in range(i + 1, len(u_groups)):
                g1, g2 = u_groups[i], u_groups[j]
                vals1 = df_well[df_well['Group'] == g1][feature].dropna()
                vals2 = df_well[df_well['Group'] == g2][feature].dropna()
                n1, n2 = len(vals1), len(vals2)
                if n1 >= min_wells_for_stats and n2 >= min_wells_for_stats:
                    _, p_val = ttest_ind(vals1, vals2, equal_var=False)
                    stats_list.append({
                        'Window': w_idx, 'Window_Label': _window_label(windows[w_idx]), 'Day': day,
                        'pair': (g1, g2), 'p_raw': p_val,
                        'hedges_g': _hedges_g(vals1, vals2),
                        'n1': n1, 'n2': n2, 'Skipped_Insufficient_N': False,
                    })
                elif n1 >= 1 and n2 >= 1:
                    print(
                        f"Skipped {g1} vs {g2}, {_window_label(windows[w_idx])} Day {day} ({feature}): "
                        f"insufficient wells (n={n1}, {n2}) below min_wells_for_stats={min_wells_for_stats}."
                    )
                    skipped_stats_list.append({
                        'Window': w_idx, 'Window_Label': _window_label(windows[w_idx]), 'Day': day,
                        'pair': (g1, g2), 'p_raw': np.nan, 'p_corr': np.nan,
                        'stars': 'skipped', 'hedges_g': np.nan,
                        'n1': n1, 'n2': n2, 'Skipped_Insufficient_N': True,
                    })

    if stats_list:
        raw_pvals = [s['p_raw'] for s in stats_list]
        _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')
        print(f"\n--- FDR Corrected Statistics for {metric_label} by developmental window/day, tested on well-level means ---")
        for i, s in enumerate(stats_list):
            s['p_corr'] = corrected_pvals[i]
            p = s['p_corr']
            s['stars'] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(
                f"{s['Window_Label']} Day {s['Day']} {s['pair'][0]} vs {s['pair'][1]}: p_raw={s['p_raw']:.5f} "
                f"p_adj={p:.5f} ({s['stars']}), Hedges' g = {s['hedges_g']:.3f} (n={s['n1']} vs {s['n2']} wells)"
            )
        pd.DataFrame(stats_list + skipped_stats_list).to_csv(
            os.path.join(plot_data_dir, f"{feature}_{group_by}_boxplot_by_window_stats.csv"), index=False
        )
    elif skipped_stats_list:
        pd.DataFrame(skipped_stats_list).to_csv(
            os.path.join(plot_data_dir, f"{feature}_{group_by}_boxplot_by_window_stats.csv"), index=False
        )

    all_possible_groups = sorted({
        group for df in day_data.values() for group in df['Group'].dropna().unique()
    })
    palette = dict(zip(all_possible_groups, sns.color_palette("husl", len(all_possible_groups))))

    n_rows = len(windows)
    n_cols = max(len(window) for window in windows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(STANDARD_FIG_WIDTH_IN/3*n_cols, STANDARD_FIG_HEIGHT_IN * n_rows), sharey='row', squeeze=False)
    sns.set_style("whitegrid")

    group_handles = {}
    for w_idx, window in enumerate(windows):
        for col_idx in range(n_cols):
            ax = axes[w_idx][col_idx]
            if col_idx >= len(window) or (w_idx, window[col_idx]) not in day_data:
                ax.axis('off')
                continue

            day = window[col_idx]
            df_day = day_data[(w_idx, day)]
            u_groups = sorted(df_day['Group'].dropna().unique())

            sns.violinplot(
                data=df_day, x='Group', y=feature, order=u_groups,
                palette=palette, hue='Group', legend=False, ax=ax,
                inner=None, cut=0, alpha=0.35, linewidth=0.8, zorder=1,
            )
            sns.stripplot(
                data=df_day, x='Group', y=feature, order=u_groups,
                palette=palette, hue='Group', jitter=True, alpha=0.8,
                s=2, ax=ax, legend=False, zorder=2,
            )

            means = df_day.groupby('Group')[feature].mean().reindex(u_groups)
            for x_pos, mean_val in enumerate(means):
                ax.hlines(y=mean_val, xmin=x_pos - 0.25, xmax=x_pos + 0.25, color="black", lw=1, zorder=5)

            day_stats = [s for s in stats_list if s['Window'] == w_idx and s['Day'] == day]
            y_max = df_day[feature].max()
            y_offset = abs(y_max) * 0.3 if y_max else 0.1
            for s_idx, s in enumerate(day_stats):
                g1, g2 = s['pair']
                if g1 not in u_groups or g2 not in u_groups:
                    continue
                x1, x2 = u_groups.index(g1), u_groups.index(g2)
                h = y_max + (s_idx * y_offset)
                ax.plot([x1, x1, x2, x2], [h, h + y_offset * 0.2, h + y_offset * 0.2, h], lw=0.5, color='black')
                ax.text((x1 + x2) * 0.5, h + y_offset * 0.2, _stat_annotation_text(s), ha='center', va='bottom', color='black', fontsize=stats_fontsize)

            if col_idx > 0:
                ax.set_ylabel("")
                ax.spines['left'].set_visible(False)
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(metric_label)

            ax.tick_params(labelbottom=False)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("")
            ax.set_title(f"Day {day}", fontsize=10)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for group in u_groups:
                if group not in group_handles:
                    group_handles[group] = mlines.Line2D(
                        [], [], color=palette[group], marker='o',
                        linestyle='None', markersize=10, label=group
                    )

        # Row label naming the window, placed in the figure margin to the
        # left of that row (approximate vertical centering across rows;
        # fine for a handful of windows, may need manual nudging for many).
        #row_center_y = 1 - (w_idx + 0.5) / n_rows
        #fig.text(
        #    0.005, row_center_y, _window_label(window),
        #    rotation=90, ha='left', va='center', fontsize=9, fontweight='bold',
        #)

    handles = [group_handles[k] for k in sorted(group_handles.keys())]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=len(handles), frameon=False, title=None)

    save_path = os.path.join(plot_dir, f"{feature}_{group_by}_boxplots_by_window.png")
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")

    return pd.DataFrame(stats_list)


def analyze_expansion_rate_by_window(
    base_dir, project_dir, batches_complete_info, windows, feature="Area_um2",
    metric_label="Area (um^2)", group_by="batch", min_points=2, error_bar="sem",
    min_wells_for_stats=2,
):
    """
    Per-well expansion rate within each developmental window. Pools
    Organoid_Mode == "single" wells (one organoid's own measurement per
    day) with Organoid_Mode == "multiple" wells (that day's mean across
    every organoid detected in the well's image) -- see
    ``analyze_growth_trajectories`` for why the well itself stays a stable,
    trackable unit across days in both modes even though individual
    organoids within a "multiple" well do not.

    Fits log(feature) ~ Day (exponential growth model) separately per well
    per window, using only that window's own days -- unlike
    ``analyze_growth_trajectories``, which fits one rate across the whole
    available timecourse. A window needs at least ``min_points`` valid days
    for a given well to get a rate; with exactly 2 points, R^2 is reported
    as NaN (a line through 2 points fits perfectly by construction, so R^2
    there is not a meaningful fit-quality signal).

    Produces:
      - a point/line plot of the mean rate +/- error_bar ("sem" by default,
        or "sd"/"ci"; see _resolve_error_bar_values -- set via
        plotting_metadata.json's "line_plot_error_bar") per window (x =
        window, y = rate, colored by group) -- how the growth rate itself
        changes across development;
      - a grid of box/strip plots comparing groups' per-well rate within
        each window, with Welch's t-test + BH-FDR + Hedges' g.
    """
    plot_dir = _plots_root(base_dir, project_dir)
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    all_days = sorted({day for window in windows for day in window})

    records = []
    for entry in batches_complete_info:
        batch, label = entry['batch'], entry.get(group_by, entry['batch'])
        for day in entry['days']:
            if day not in all_days:
                continue
            path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df = _ensure_metric_identity_columns(df)
            if df.empty or feature not in df.columns:
                continue
            df = filter_by_diameter(df, day)
            if df.empty:
                continue
            # See analyze_growth_trajectories for why this per-well mean is
            # a no-op for organoid_mode == "single" and a genuine
            # mean-across-organoids for organoid_mode == "multiple".
            day_df = df.groupby('Well_ID', dropna=False)[feature].mean().reset_index()
            counts = df.groupby('Well_ID', dropna=False).size().reset_index(name='Object_Count_In_Image')
            day_df = day_df.merge(counts, on='Well_ID', how='left')
            if 'Organoid_Mode' in df.columns:
                modes = df.groupby('Well_ID', dropna=False)['Organoid_Mode'].first().reset_index()
                day_df = day_df.merge(modes, on='Well_ID', how='left')
            day_df['Batch'] = batch
            day_df['Group'] = label
            day_df['Day'] = day
            records.append(day_df)

    if not records:
        print(f"No data found for windowed expansion rate analysis of {feature}.")
        return pd.DataFrame()

    df_all = pd.concat(records, ignore_index=True)

    rate_rows = []
    for w_idx, window in enumerate(windows):
        window_df = df_all[df_all['Day'].isin(window)]
        for (batch, well_id, group), well_df in window_df.groupby(['Batch', 'Well_ID', 'Group']):
            well_df = well_df.dropna(subset=[feature]).sort_values('Day')
            well_df = well_df[well_df[feature] > 0]
            if len(well_df) < min_points:
                continue

            day_vals = well_df['Day'].values.astype(float)
            log_vals = np.log(well_df[feature].values)
            slope, intercept = np.polyfit(day_vals, log_vals, 1)

            if len(well_df) > 2:
                predicted = slope * day_vals + intercept
                ss_res = np.sum((log_vals - predicted) ** 2)
                ss_tot = np.sum((log_vals - log_vals.mean()) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            else:
                r_squared = np.nan

            rate_rows.append({
                'Batch': batch, 'Well_ID': well_id, 'Group': group,
                'Window_Index': w_idx, 'Window_Label': _window_label(window),
                'N_Days': len(well_df),
                'Growth_Rate_Per_Day': slope,
                'Doubling_Time_Days': np.log(2) / slope if slope > 0 else np.nan,
                'R_Squared': r_squared,
            })

    rate_df = pd.DataFrame(rate_rows)
    if rate_df.empty:
        print(f"No wells had >= {min_points} days within any window for {feature} expansion-rate fitting.")
        return rate_df

    rate_df.to_csv(os.path.join(plot_data_dir, f"{feature}_expansion_rate_by_window.csv"), index=False)

    unique_groups = sorted(rate_df['Group'].dropna().unique())
    palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))
    window_labels = [_window_label(w) for w in windows]

    # --- Plot 1: mean rate +/- error_bar across windows ---
    summary = rate_df.groupby(['Group', 'Window_Index'])['Growth_Rate_Per_Day'].agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['error_value'] = _resolve_error_bar_values(summary, error_bar)

    fig, ax = plt.subplots(figsize=(2 + 2 * len(windows), 6))
    sns.set_style("whitegrid")
    for group in unique_groups:
        data = summary[summary['Group'] == group].sort_values('Window_Index')
        ax.errorbar(
            data['Window_Index'], data['mean'], yerr=data['error_value'],
            label=group, marker='o', capsize=4, color=palette[group], linewidth=2,
        )
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(window_labels, rotation=30, ha='right')
    ax.set_xlabel('Developmental window')
    ax.set_ylabel(f"Growth rate ({metric_label}/day)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"{feature}_expansion_rate_by_window_{group_by}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")

    # --- Plot 2: boxplot grid, one panel per window, with stats ---
    stats_list = []
    skipped_stats_list = []
    for w_idx in sorted(rate_df['Window_Index'].unique()):
        window_df = rate_df[rate_df['Window_Index'] == w_idx]
        u_groups = sorted(window_df['Group'].dropna().unique())
        if len(u_groups) < 2:
            continue
        for i in range(len(u_groups)):
            for j in range(i + 1, len(u_groups)):
                g1, g2 = u_groups[i], u_groups[j]
                vals1 = window_df[window_df['Group'] == g1]['Growth_Rate_Per_Day'].dropna()
                vals2 = window_df[window_df['Group'] == g2]['Growth_Rate_Per_Day'].dropna()
                n1, n2 = len(vals1), len(vals2)
                if n1 >= min_wells_for_stats and n2 >= min_wells_for_stats:
                    _, p_val = ttest_ind(vals1, vals2, equal_var=False)
                    stats_list.append({
                        'window': w_idx, 'window_label': _window_label(windows[w_idx]),
                        'pair': (g1, g2), 'p_raw': p_val,
                        'hedges_g': _hedges_g(vals1, vals2),
                        'n1': n1, 'n2': n2, 'Skipped_Insufficient_N': False,
                    })
                elif n1 >= 1 and n2 >= 1:
                    print(
                        f"Skipped {g1} vs {g2}, {_window_label(windows[w_idx])} ({feature} expansion rate): "
                        f"insufficient wells (n={n1}, {n2}) below min_wells_for_stats={min_wells_for_stats}."
                    )
                    skipped_stats_list.append({
                        'window': w_idx, 'window_label': _window_label(windows[w_idx]),
                        'pair': (g1, g2), 'p_raw': np.nan, 'p_corr': np.nan,
                        'stars': 'skipped', 'hedges_g': np.nan,
                        'n1': n1, 'n2': n2, 'Skipped_Insufficient_N': True,
                    })

    if stats_list:
        raw_pvals = [s['p_raw'] for s in stats_list]
        _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')
        print(f"\n--- FDR Corrected Statistics for {metric_label} expansion rate by developmental window ---")
        for i, s in enumerate(stats_list):
            s['p_corr'] = corrected_pvals[i]
            p = s['p_corr']
            s['stars'] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(
                f"{s['window_label']} {s['pair'][0]} vs {s['pair'][1]}: p_raw={s['p_raw']:.5f} "
                f"p_adj={p:.5f} ({s['stars']}), Hedges' g = {s['hedges_g']:.3f} (n={s['n1']} vs {s['n2']} wells)"
            )
        pd.DataFrame(stats_list + skipped_stats_list).to_csv(
            os.path.join(plot_data_dir, f"{feature}_expansion_rate_by_window_stats.csv"), index=False
        )
    elif skipped_stats_list:
        pd.DataFrame(skipped_stats_list).to_csv(
            os.path.join(plot_data_dir, f"{feature}_expansion_rate_by_window_stats.csv"), index=False
        )

    window_indices = sorted(rate_df['Window_Index'].unique())
    n_cols, n_rows = _panel_grid_dims(len(window_indices))
    LEGEND_RESERVE_IN = 0.6
    fig = plt.figure(figsize=(STANDARD_FIG_WIDTH_IN/3*n_cols, STANDARD_FIG_HEIGHT_IN * n_rows))
    bottom_frac = LEGEND_RESERVE_IN / (STANDARD_FIG_HEIGHT_IN * n_rows + LEGEND_RESERVE_IN)
    # hspace considerably larger than plot_feature_grid_scatters's equivalent
    # gridspec -- this figure's panels show rotated category-name x-tick
    # labels (grid_scatterplot's don't), which need real vertical room
    # between one panel's labels and the next panel's title below them.
    gs = fig.add_gridspec(n_rows, n_cols, top=0.95, bottom=bottom_frac, hspace=0.15, wspace=0.3)
    sns.set_style("whitegrid")
    axes = [fig.add_subplot(gs[i // n_cols, i % n_cols]) for i in range(len(window_indices))]
    for ax in axes[1:]:
        ax.sharey(axes[0])

    # All window-panels share one y-axis (via sharey above), so the
    # vertical spacing between stacked bracket rows must come from ONE
    # shared reference rather than each panel's own y_max -- see the
    # identical fix (and rationale) in plot_feature_grid_scatters above.
    # Using the GLOBAL max across every window (rather than, say, the last
    # window specifically) since the window with the fewest groups/
    # narrowest range isn't necessarily the one whose bracket stack needs
    # the least headroom -- windows nearer the end of a timecourse can
    # have fewer surviving wells and thus a smaller max than an earlier,
    # fuller window.
    global_max = rate_df['Growth_Rate_Per_Day'].max()
    shared_y_offset = abs(global_max) * 0.3 if global_max else None

    group_handles = {}
    for ax_idx, w_idx in enumerate(window_indices):
        ax = axes[ax_idx]
        window_df = rate_df[rate_df['Window_Index'] == w_idx]
        u_groups = sorted(window_df['Group'].dropna().unique())

        sns.violinplot(
            data=window_df, x='Group', y='Growth_Rate_Per_Day', order=u_groups,
            palette=palette, hue='Group', legend=False, ax=ax,
            inner=None, cut=0, alpha=0.35, linewidth=0.8, zorder=1,
        )
        sns.stripplot(
            data=window_df, x='Group', y='Growth_Rate_Per_Day', order=u_groups,
            palette=palette, hue='Group', jitter=True, alpha=0.8, s=2, ax=ax,
            legend=False, zorder=2,
        )

        # Mean lines -- same style as plot_feature_grid_scatters's.
        means = window_df.groupby('Group')['Growth_Rate_Per_Day'].mean().reindex(u_groups)
        for x_pos, mean_val in enumerate(means):
            ax.hlines(y=mean_val, xmin=x_pos - 0.25, xmax=x_pos + 0.25, color="black", lw=1, zorder=5)

        window_stats = [s for s in stats_list if s['window'] == w_idx]
        y_max = window_df['Growth_Rate_Per_Day'].max()
        y_offset = shared_y_offset if shared_y_offset else (abs(y_max) * 0.3 if y_max else 0.01)
        for s_idx, s in enumerate(window_stats):
            g1, g2 = s['pair']
            if g1 not in u_groups or g2 not in u_groups:
                continue
            x1, x2 = u_groups.index(g1), u_groups.index(g2)
            h = y_max + (s_idx * y_offset)
            ax.plot([x1, x1, x2, x2], [h, h + y_offset * 0.2, h + y_offset * 0.2, h], lw=0.5, color='black')
            ax.text((x1 + x2) * 0.5, h + y_offset * 0.2, _stat_annotation_text(s), ha='center', va='bottom', fontsize=stats_fontsize)

        if ax_idx % n_cols > 0:
            ax.set_ylabel("")
            ax.spines['left'].set_visible(False)
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(f"Growth rate ({metric_label}/day)")

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("")
        # No visible x-tick labels -- the bottom legend already identifies
        # each group by color, so long category names never need to be
        # rendered (rotated or otherwise) at this figure's narrow per-panel
        # width.
        ax.tick_params(labelbottom=False)
        ax.set_title(_window_label(windows[w_idx]), fontsize=10)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for group in u_groups:
            if group not in group_handles:
                group_handles[group] = mlines.Line2D(
                    [], [], color=palette[group], marker='o',
                    linestyle='None', markersize=10, label=group
                )

    handles = [group_handles[k] for k in sorted(group_handles.keys())]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(handles), 2), frameon=False, title=None)

    save_path = os.path.join(plot_dir, f"{feature}_expansion_rate_boxplots_by_window.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")

    return rate_df
