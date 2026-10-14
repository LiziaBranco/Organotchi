"""
Morphology-metrics EXTRACTION (reads already-produced segmentation masks,
writes per-object/per-well CSVs) -- split out of utils_segmentation.py so
segmentation (producing masks) and morphology extraction (measuring them)
are two independent stages, each with its own script (segmentation.py /
morphology.py) and its own utils module (this one / utils_segmentation.py).

"""

import os
from os.path import join as opj

import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure, morphology
from skimage.filters import threshold_otsu

from utils_segmentation import (
    LABEL_CONNECTIVITY,
    _collect_tile_images,
    _convert_tif_to_jpg_array,
    _get_image_measurement_context,
    _normalize_foreground_polarity,
    _pixel_area_to_micron,
    _pixel_length_to_micron,
)


def _load_grayscale_for_intensity_analysis(tile_path):
    """Grayscale pixels for a photometric measurement (the light-dark Otsu
    split, etc.) read straight from the original raw file.
    """
    extension = os.path.splitext(tile_path)[1].lower()
    if extension not in {".tif", ".tiff"}:
        with Image.open(tile_path) as img:
            return np.array(img.convert("L"))

    with Image.open(tile_path) as img:
        img_8bit = _convert_tif_to_jpg_array(np.array(img))
    return np.array(img_8bit.convert("L"))


def _metrics_outputs_are_current(metrics_path, inference_npy_path, output_filenames=('mask_metrics.csv', 'well_metrics_summary.csv')):
    """Return True when saved morphology CSVs are newer than all source masks."""
    output_paths = [os.path.join(metrics_path, filename) for filename in output_filenames]

    if any(not os.path.exists(path) or os.path.getsize(path) == 0 for path in output_paths):
        return False

    mask_paths = [
        os.path.join(inference_npy_path, filename)
        for filename in os.listdir(inference_npy_path)
        if filename.endswith('.npy')
    ]
    if not mask_paths:
        return False

    oldest_output_mtime = min(os.path.getmtime(path) for path in output_paths)
    newest_mask_mtime = max(os.path.getmtime(path) for path in mask_paths)

    return oldest_output_mtime >= newest_mask_mtime


def _summarize_metrics_by_image(metrics_df, feature_columns):
    """Create one row per image/well with object counts and feature summaries."""
    if metrics_df.empty:
        return pd.DataFrame()

    group_cols = [
        "Batch", "Day", "Run", "Image_ID", "Well_ID",
        "Plate_Index", "Well_Index", "Organoid_Mode",
    ]
    group_cols = [column for column in group_cols if column in metrics_df.columns]
    summary = (
        metrics_df
        .groupby(group_cols, dropna=False)
        .agg(Detected_Organoid_Count=("Organoid_ID", "count"))
        .reset_index()
    )

    if "Area_um2" in metrics_df.columns:
        # Sum, not mean: the well's total organoid coverage. Unlike
        # Detected_Organoid_Count or any per-object shape descriptor, this is
        # unaffected by touching/merged organoids being traced as a single
        # contour.
        total_area = (
            metrics_df
            .groupby(group_cols, dropna=False)["Area_um2"]
            .sum()
            .reset_index()
            .rename(columns={"Area_um2": "Total_Segmented_Area_um2"})
        )
        summary = summary.merge(total_area, on=group_cols, how="left")

    for feature in feature_columns:
        feature_summary = (
            metrics_df
            .groupby(group_cols, dropna=False)[feature]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        feature_summary.rename(
            columns={
                "mean": f"{feature}_Mean",
                "std": f"{feature}_Std",
                "min": f"{feature}_Min",
                "max": f"{feature}_Max",
            },
            inplace=True,
        )
        summary = summary.merge(feature_summary, on=group_cols, how="left")
        mean_col = f"{feature}_Mean"
        std_col = f"{feature}_Std"
        cv_col = f"{feature}_CV"
        summary[cv_col] = np.where(
            summary[mean_col].abs() > 0,
            summary[std_col] / summary[mean_col],
            np.nan,
        )

    return summary


def extract_organoid_morphology_metrics(
    base_dir,
    project_dir,
    days,
    batches,
    pixel_to_micron_x,
    pixel_to_micron_y,
    organoid_mode="single",
    run_name=None,
    images_per_plate=24,
    force=False,
):
    """Extracts morphological features handling anisotropic pixels (X != Y)."""
    organoid_mode = str(organoid_mode).lower()
    organoid_mode = "multiple" if organoid_mode in {"multiple", "multi", "many"} else "single"
    print(f"\n📐 Extracting Organoid Morphology Metrics")

    for day in days:
        print(f"- Processing Day {day}")
        batch_processed_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}')
        for batch in batches:
            inference_npy_path = os.path.join(batch_processed_path, batch, 'inference_filtered_npy')
            if not os.path.exists(inference_npy_path):
                inference_npy_path = os.path.join(batch_processed_path, batch, 'inference_npy')

            if not os.path.exists(inference_npy_path):
                continue

            metrics_path = os.path.join(batch_processed_path, batch, 'inference_metrics')
            os.makedirs(metrics_path, exist_ok=True)
            if not force and _metrics_outputs_are_current(metrics_path, inference_npy_path):
                print(f"  Skipping {batch}, Day {day}: morphology metrics are up to date.")
                continue

            metrics_data = []
            empty_or_failed_images = []
            feature_columns = [
                'Area_um2', 'Diameter_um', 'Perimeter_um',
                'Feret_Diameter_Max_um', 'Aspect_Ratio', 'Circularity',
                'Roundness', 'Solidity', 'Eccentricity', 'Skeleton_Length_um'
            ]

            for filename in sorted(os.listdir(inference_npy_path)):
                if not filename.endswith('.npy'):
                    continue

                image_id = filename.replace('.npy', '')

                try:
                    mask = np.load(os.path.join(inference_npy_path, filename), mmap_mode='r')
                    labeled_mask = measure.label(mask > 0, connectivity=LABEL_CONNECTIVITY)
                    object_count = int(labeled_mask.max())
                    if object_count == 0:
                        empty_or_failed_images.append({
                            "Image_ID": image_id, "Reason": "zero_objects_detected"
                        })
                        continue

                    image_context = _get_image_measurement_context(image_id, images_per_plate)

                    # 1. Get properties WITH spacing (for Area and Axis lengths, which support it;
                    # major/minor axis lengths are supported with spacing)
                    props_um = measure.regionprops(labeled_mask, spacing=(pixel_to_micron_y, pixel_to_micron_x), cache=True)

                    # 2. Get properties WITHOUT spacing (pixels) to avoid the NotImplementedError
                    props_px = measure.regionprops(labeled_mask, cache=True)
                except Exception as exc:
                    print(f"⚠️  Failed to load/label mask {filename}: {exc}")
                    empty_or_failed_images.append({
                        "Image_ID": image_id, "Reason": f"mask_load_failed: {exc}"
                    })
                    continue

                for p_um, p_px in zip(props_um, props_px):
                    try:
                        organoid_id = image_id
                        if organoid_mode == "multiple" or object_count > 1:
                            organoid_id = f"{image_id}_object{p_px.label:03d}"

                        # --- Basic Metrics (Supported by spacing) ---
                        area_um2 = p_um.area
                        major_axis_um = p_um.axis_major_length
                        minor_axis_um = p_um.axis_minor_length
                        solidity = p_um.solidity
                        eccentricity = p_um.eccentricity

                        # --- 1D Metrics (Scaled manually to avoid spacing error).
                        # Uses the same average-spacing convention as the classical/DL
                        # contour filters (_pixel_length_to_micron) for consistency.
                        perimeter_um = _pixel_length_to_micron(p_px.perimeter, pixel_to_micron_x, pixel_to_micron_y)

                        try:
                            feret_max_um = _pixel_length_to_micron(
                                p_px.feret_diameter_max, pixel_to_micron_x, pixel_to_micron_y
                            )
                        except AttributeError:
                            feret_max_um = np.nan

                        # --- Derived Metrics ---
                        diameter_um = np.sqrt(4 * area_um2 / np.pi)
                        circularity = (4 * np.pi * area_um2) / (perimeter_um ** 2) if perimeter_um > 0 else np.nan
                        aspect_ratio = major_axis_um / minor_axis_um if minor_axis_um > 0 else np.nan
                        roundness = (4 * area_um2) / (np.pi * (feret_max_um ** 2)) if feret_max_um > 0 else np.nan
                        skeleton = morphology.skeletonize(p_px.image)
                        skeleton_length_um = _pixel_length_to_micron(
                            np.sum(skeleton), pixel_to_micron_x, pixel_to_micron_y
                        )

                        metrics_data.append([
                            batch,
                            day,
                            run_name,
                            organoid_mode,
                            image_context["Image_ID"],
                            image_context["Well_ID"],
                            image_context["Plate_Index"],
                            image_context["Well_Index"],
                            organoid_id,
                            p_px.label,
                            object_count,
                            area_um2, diameter_um, perimeter_um,
                            feret_max_um, aspect_ratio, circularity,
                            roundness, solidity, eccentricity, skeleton_length_um
                        ])
                    except Exception as exc:
                        print(f"⚠️  Failed to measure object {p_px.label} in {filename}: {exc}")
                        empty_or_failed_images.append({
                            "Image_ID": image_id,
                            "Reason": f"object_{p_px.label}_measurement_failed: {exc}",
                        })

            if metrics_data:
                columns = [
                    'Batch', 'Day', 'Run', 'Organoid_Mode', 'Image_ID', 'Well_ID',
                    'Plate_Index', 'Well_Index', 'Organoid_ID', 'Object_Label',
                    'Object_Count_In_Image',
                    'Area_um2', 'Diameter_um', 'Perimeter_um',
                    'Feret_Diameter_Max_um', 'Aspect_Ratio', 'Circularity',
                    'Roundness', 'Solidity', 'Eccentricity', 'Skeleton_Length_um'
                ]
                df = pd.DataFrame(metrics_data, columns=columns)
                df.to_csv(os.path.join(metrics_path, 'mask_metrics.csv'), index=False)
                image_summary = _summarize_metrics_by_image(df, feature_columns)
                image_summary.to_csv(os.path.join(metrics_path, 'well_metrics_summary.csv'), index=False)

            if empty_or_failed_images:
                # Makes "0 organoids detected" distinguishable from "never processed"
                # without cross-referencing other CSVs.
                pd.DataFrame(empty_or_failed_images).to_csv(
                    os.path.join(metrics_path, 'empty_or_failed_masks.csv'), index=False
                )


def _classify_light_dark_pixels(gray_image, mask, foreground_polarity="dark", min_eta_squared=0.3):
    """Otsu-split ``gray_image``'s pixels within ``mask`` into a lighter
    "light" class vs a darker "dark" class. Returns ``(light_pixel_mask,
    split_threshold, eta_squared, has_bimodal_split)`` -- ``split_threshold``
    is ``nan`` if ``mask`` has no foreground pixels at all.
    """
    foreground_polarity = _normalize_foreground_polarity(foreground_polarity)
    values = gray_image[mask > 0].astype(np.float64)

    if values.size == 0:
        return np.zeros_like(mask, dtype=bool), float("nan"), float("nan"), False

    if np.ptp(values) == 0:
        # Perfectly uniform region: no separable structure at all -- every
        # pixel is on the same side of any threshold, so call it all dark.
        split_threshold = float(values[0])
        eta_squared = 0.0
    else:
        split_threshold = float(threshold_otsu(values))
        below = values[values <= split_threshold]
        above = values[values > split_threshold]
        total_var = float(values.var())
        if below.size == 0 or above.size == 0 or total_var == 0:
            eta_squared = 0.0
        else:
            weight_below, weight_above = below.size / values.size, above.size / values.size
            between_class_var = weight_below * weight_above * (below.mean() - above.mean()) ** 2
            eta_squared = float(between_class_var / total_var)

    has_bimodal_split = eta_squared >= min_eta_squared

    if foreground_polarity == "dark":
        light_pixel_mask = (mask > 0) & (gray_image.astype(np.float64) > split_threshold)
    else:
        light_pixel_mask = (mask > 0) & (gray_image.astype(np.float64) < split_threshold)

    return light_pixel_mask, split_threshold, eta_squared, has_bimodal_split


def estimate_light_dark_split_from_mask(
    gray_image, mask, pixel_to_micron_x, pixel_to_micron_y, foreground_polarity="dark",
    min_eta_squared=0.3, return_pixel_mask=False,
):
    """Split one segmented (mask) region's own grayscale pixels into a
    lighter outer "light" region vs a darker inner "dark" region, from the
    intensity histogram of just those pixels. Called
    per organoid (one ``mask``/``gray_image`` pair per object, e.g.
    ``region.image``/``region.image_intensity`` from a labeled mask's own
    regionprops).

    Early on, the segmented region is close to uniformly dark (the
    histogram of its own pixel values has a single peak). As a lighter
    peripheral region grows (whether from cyst formation or from
    organoid size/shape change), a second, lighter peak appears alongside
    the dark region's peak. Otsu's method finds the threshold that best
    separates a bimodal histogram into two classes; run on just the masked
    pixels (not the whole image, and not the global segmentation
    `threshold`, which separates organoid-vs-background),
    it recovers a within-organoid light/dark split.

    Otsu always returns *some* threshold, even on a unimodal
    histogram (common on early days, before much of a light region has
    developed) -- so a nonzero light fraction on those days doesn't
    necessarily mean anything "biologically meaningful" was separated.
    ``has_bimodal_split`` flags this by comparing it's the two classes'
    eta-squared  and ``min_eta_squared``. Below that
    cutoff, the two "classes" barely differ from each other relative to the
    pixels' own spread -- there's no real separable structure yet, and
    ``light_dark_ratio``/``light_fraction`` should be read as "not yet
    separable", not "close to zero". There's no literature-standard cutoff
    for this; 0.3 is a practical default -- tune per project if it visibly
    over/under-flags on real images.

    For "dark" polarity, the segmented region's dark tissue is darker than
    its light region (both are still darker than the background outside the
    mask, or they wouldn't be segmented at all), so pixels above the
    intra-mask split are classed as light; for "bright" polarity this is
    reversed.

    Returns a dict with ``dark_area_um2``, ``light_area_um2``,
    ``light_dark_ratio`` (``light_area_um2 / dark_area_um2``, unbounded,
    ``nan`` if there is no dark area to divide by), ``light_fraction``
    (``light_area_um2 / (light_area_um2 + dark_area_um2)``, bounded to
    [0, 1] and defined even when the region is entirely light),
    ``dark_fraction`` (``1 - light_fraction``, the complementary bounded
    share -- the more direct read for "how much of this organoid is dense
    (dark)", prefer this or ``light_fraction`` over ``light_dark_ratio`` for
    summary statistics/plots, since the ratio diverges as the dark area
    shrinks toward zero), ``split_threshold`` (the raw grayscale cutoff
    used, for QC/inspection -- e.g. to check it isn't drifting across
    days/batches for reasons unrelated to biology), ``otsu_eta_squared``/
    ``has_bimodal_split`` (see above), and -- since a single ratio can't
    distinguish "one big light region" from "many small ones" that happen to
    sum to the same area -- ``light_region_count`` and the light regions'
    own ``light_region_mean_area_um2``/``_median_area_um2``/``_max_area_um2``
    (all 0/NaN if no light pixels survive the split), from
    connected-component labeling of just the light-classified pixels within
    the mask.

    ``return_pixel_mask=True`` additionally includes ``light_pixel_mask`` (a
    boolean array, same shape as ``mask``, True wherever a pixel was
    classified light) -- for callers building a visual classification
    overlay (see ``generate_light_dark_overlay_masks``).
    """
    empty_region_stats = {
        "light_region_count": 0,
        "light_region_mean_area_um2": float("nan"),
        "light_region_median_area_um2": float("nan"),
        "light_region_max_area_um2": float("nan"),
    }

    light_pixel_mask, split_threshold, eta_squared, has_bimodal_split = _classify_light_dark_pixels(
        gray_image, mask, foreground_polarity, min_eta_squared,
    )

    if np.isnan(split_threshold):
        empty_result = {
            "dark_area_um2": 0.0,
            "light_area_um2": 0.0,
            "light_dark_ratio": float("nan"),
            "light_fraction": float("nan"),
            "dark_fraction": float("nan"),
            "split_threshold": float("nan"),
            "otsu_eta_squared": float("nan"),
            "has_bimodal_split": False,
            **empty_region_stats,
        }
        if return_pixel_mask:
            empty_result["light_pixel_mask"] = light_pixel_mask
        return empty_result

    light_pixel_count = int(np.count_nonzero(light_pixel_mask))
    dark_pixel_count = int(np.count_nonzero(mask > 0)) - light_pixel_count

    light_area_um2 = _pixel_area_to_micron(light_pixel_count, pixel_to_micron_x, pixel_to_micron_y)
    dark_area_um2 = _pixel_area_to_micron(dark_pixel_count, pixel_to_micron_x, pixel_to_micron_y)
    total_area_um2 = light_area_um2 + dark_area_um2
    light_dark_ratio = light_area_um2 / dark_area_um2 if dark_area_um2 > 0 else float("nan")
    light_fraction = light_area_um2 / total_area_um2 if total_area_um2 > 0 else float("nan")
    dark_fraction = dark_area_um2 / total_area_um2 if total_area_um2 > 0 else float("nan")

    region_stats = dict(empty_region_stats)
    if light_pixel_count > 0:
        labeled_light = measure.label(light_pixel_mask, connectivity=LABEL_CONNECTIVITY)
        region_areas_um2 = [
            _pixel_area_to_micron(region.area, pixel_to_micron_x, pixel_to_micron_y)
            for region in measure.regionprops(labeled_light)
        ]
        if region_areas_um2:
            region_stats = {
                "light_region_count": len(region_areas_um2),
                "light_region_mean_area_um2": float(np.mean(region_areas_um2)),
                "light_region_median_area_um2": float(np.median(region_areas_um2)),
                "light_region_max_area_um2": float(np.max(region_areas_um2)),
            }

    result = {
        "dark_area_um2": dark_area_um2,
        "light_area_um2": light_area_um2,
        "light_dark_ratio": light_dark_ratio,
        "light_fraction": light_fraction,
        "dark_fraction": dark_fraction,
        "split_threshold": split_threshold,
        "otsu_eta_squared": eta_squared,
        "has_bimodal_split": has_bimodal_split,
        **region_stats,
    }
    if return_pixel_mask:
        result["light_pixel_mask"] = light_pixel_mask
    return result


def extract_light_dark_metrics(
    base_dir,
    project_dir,
    days,
    batches,
    pixel_to_micron_x,
    pixel_to_micron_y,
    organoid_mode="single",
    foreground_polarity="dark",
    min_eta_squared=0.3,
    images_per_plate=24,
    run_name=None,
    force=False,
):
    """For projects with ``track_light_dark_split`` enabled in
    ``segmentation_metadata.json``, estimate each individual organoid's own
    light-vs-dark split, from its own grayscale pixel histogram (see
    ``estimate_light_dark_split_from_mask`` for what "light"/"dark" mean
    here).

    Per-organoid, not pooled across a well/tile.

    Reads each tile's own raw grayscale pixels via
    ``_load_grayscale_for_intensity_analysis``.

    ``Organoid_ID``/``Object_Label``/``Object_Count_In_Image``/``Area_um2``/
    ``Diameter_um`` use the exact same per-object identity/geometry as
    ``extract_organoid_morphology_metrics``, so ``apply_outlier_removal_gate``
    and the rest of the object-level plotting machinery in
    ``utils_plot.py`` (outlier removal, well-mean aggregation,
    per-window trend plots) work on this CSV directly via
    plotting_metadata.json's ``window_instant_metrics``.

    Writes one row per organoid to ``inference_metrics/light_dark_metrics.csv``.
    """
    organoid_mode = str(organoid_mode).lower()
    organoid_mode = "multiple" if organoid_mode in {"multiple", "multi", "many"} else "single"
    print(f"\n🎯 Extracting Light-Dark Metrics")

    for day in days:
        print(f"- Processing Day {day}")
        raw_data_root = opj(base_dir, project_dir, 'raw_data', f'Day {day}')
        batch_processed_path = opj(base_dir, project_dir, 'processed_data', f'Day {day}')

        for batch in batches:
            raw_data_dir = opj(raw_data_root, batch)
            processed_data_dir = opj(batch_processed_path, batch)

            inference_npy_dir = opj(processed_data_dir, 'inference_filtered_npy')
            if not os.path.exists(inference_npy_dir):
                inference_npy_dir = opj(processed_data_dir, 'inference_npy')
            if not os.path.exists(inference_npy_dir):
                continue

            metrics_path = opj(processed_data_dir, 'inference_metrics')
            os.makedirs(metrics_path, exist_ok=True)
            if not force and _metrics_outputs_are_current(
                metrics_path, inference_npy_dir, output_filenames=('light_dark_metrics.csv',)
            ):
                print(f"  Skipping {batch}, Day {day}: light-dark metrics are up to date.")
                continue

            tile_images = _collect_tile_images(raw_data_dir)
            if not tile_images:
                continue

            records = []
            failed = []

            for tile_path in tile_images:
                tile_name = os.path.splitext(os.path.basename(tile_path))[0]
                mask_path = opj(inference_npy_dir, f"{tile_name}.npy")
                if not os.path.exists(mask_path):
                    continue

                try:
                    mask = np.load(mask_path)
                    gray_image = _load_grayscale_for_intensity_analysis(tile_path)

                    if gray_image.shape != mask.shape:
                        failed.append({
                            "Image_ID": tile_name,
                            "Reason": f"shape_mismatch: raw{gray_image.shape} vs mask{mask.shape}",
                        })
                        continue

                    labeled_mask = measure.label(mask > 0, connectivity=LABEL_CONNECTIVITY)
                    object_count = int(labeled_mask.max())
                    if object_count == 0:
                        continue

                    props = measure.regionprops(labeled_mask, intensity_image=gray_image, cache=True)
                except Exception as exc:
                    failed.append({"Image_ID": tile_name, "Reason": str(exc)})
                    continue

                image_context = _get_image_measurement_context(tile_name, images_per_plate)

                for region in props:
                    try:
                        organoid_id = tile_name
                        if organoid_mode == "multiple" or object_count > 1:
                            organoid_id = f"{tile_name}_object{region.label:03d}"

                        area_um2 = _pixel_area_to_micron(region.area, pixel_to_micron_x, pixel_to_micron_y)
                        diameter_um = np.sqrt(4 * area_um2 / np.pi)

                        result = estimate_light_dark_split_from_mask(
                            region.image_intensity, region.image,
                            pixel_to_micron_x, pixel_to_micron_y,
                            foreground_polarity, min_eta_squared,
                        )
                    except Exception as exc:
                        failed.append({
                            "Image_ID": tile_name,
                            "Reason": f"object_{region.label}_measurement_failed: {exc}",
                        })
                        continue

                    records.append({
                        "Batch": batch,
                        "Day": day,
                        "Run": run_name,
                        "Organoid_Mode": organoid_mode,
                        **image_context,
                        "Organoid_ID": organoid_id,
                        "Object_Label": region.label,
                        "Object_Count_In_Image": object_count,
                        "Area_um2": area_um2,
                        "Diameter_um": diameter_um,
                        "Dark_Area_um2": result["dark_area_um2"],
                        "Light_Area_um2": result["light_area_um2"],
                        "Light_Dark_Ratio": result["light_dark_ratio"],
                        "Light_Fraction": result["light_fraction"],
                        "Dark_Fraction": result["dark_fraction"],
                        "Split_Threshold": result["split_threshold"],
                        "Otsu_Eta_Squared": result["otsu_eta_squared"],
                        "Has_Bimodal_Split": result["has_bimodal_split"],
                        "Light_Region_Count": result["light_region_count"],
                        "Light_Region_Mean_Area_um2": result["light_region_mean_area_um2"],
                        "Light_Region_Median_Area_um2": result["light_region_median_area_um2"],
                        "Light_Region_Max_Area_um2": result["light_region_max_area_um2"],
                    })

            if records:
                pd.DataFrame(records).to_csv(opj(metrics_path, 'light_dark_metrics.csv'), index=False)

            if failed:
                pd.DataFrame(failed).to_csv(opj(metrics_path, 'light_dark_failed.csv'), index=False)


def generate_light_dark_overlay_masks(
    base_dir,
    project_dir,
    days,
    batches,
    foreground_polarity="dark",
    min_eta_squared=0.3,
    light_color=(255, 0, 255),
    force=False,
):
    """For projects with ``track_light_dark_split`` enabled in
    ``segmentation_metadata.json``, render one colored classification image
    per tile/well: black background (outside every organoid), white for
    each organoid's own dark tissue, and ``light_color`` (magenta by
    default, an RGB tuple) for the light regions
    ``estimate_light_dark_split_from_mask`` classifies -- a direct visual
    answer to "what is this pipeline actually calling light vs dark", to
    sanity-check the Otsu-based split against the real images.
    Uses the exact same classification logic (``_classify_light_dark_pixels``,
    with the same ``foreground_polarity``/``min_eta_squared``) that
    ``extract_light_dark_metrics`` uses for its statistics -- an
    organoid whose split isn't considered bimodal still gets its (likely
    small/spurious) light pixels colored here, so this also doubles as a
    visual check of whether ``min_eta_squared`` is behaving sensibly on
    real images.

    Written as its own sibling folder under each day/batch's
    ``processed_data/``, ``light_dark_overlay_jpg/`` -- alongside
    ``inference_jpg``/``inference_npy``/``inference_labels_npy`` -- one
    image per tile, the same ``{tile_name}.jpg`` naming as those.

    Uses each tile's own per-instance labels (``inference_labels_npy`` --
    OrgLine's actual SAM2 instance segmentation, or the classical
    pipeline's own connected-component labeling, whichever produced this
    project's masks) when available, so touching/adjacent organoids are
    correctly told apart in the overlay; falls back to
    ``measure.label(mask > 0)`` connected components otherwise.
    """
    print(f"\n🎨 Generating Light-Dark Overlay Masks")
    light_color_rgb = tuple(int(c) for c in light_color)

    for day in days:
        print(f"- Processing Day {day}")
        raw_data_root = opj(base_dir, project_dir, 'raw_data', f'Day {day}')
        batch_processed_path = opj(base_dir, project_dir, 'processed_data', f'Day {day}')

        for batch in batches:
            raw_data_dir = opj(raw_data_root, batch)
            processed_data_dir = opj(batch_processed_path, batch)

            inference_npy_dir = opj(processed_data_dir, 'inference_filtered_npy')
            if not os.path.exists(inference_npy_dir):
                inference_npy_dir = opj(processed_data_dir, 'inference_npy')
            if not os.path.exists(inference_npy_dir):
                continue

            labels_npy_dir = opj(processed_data_dir, 'inference_labels_npy')
            overlay_dir = opj(processed_data_dir, 'light_dark_overlay_jpg')

            tile_images = _collect_tile_images(raw_data_dir)
            if not tile_images:
                continue

            os.makedirs(overlay_dir, exist_ok=True)
            failed = []

            for tile_path in tile_images:
                tile_name = os.path.splitext(os.path.basename(tile_path))[0]
                mask_path = opj(inference_npy_dir, f"{tile_name}.npy")
                if not os.path.exists(mask_path):
                    continue

                overlay_path = opj(overlay_dir, f"{tile_name}.jpg")
                if (
                    not force and os.path.exists(overlay_path)
                    and os.path.getmtime(overlay_path) >= os.path.getmtime(mask_path)
                ):
                    continue

                try:
                    mask = np.load(mask_path)
                    gray_image = _load_grayscale_for_intensity_analysis(tile_path)

                    if gray_image.shape != mask.shape:
                        failed.append({
                            "Image_ID": tile_name,
                            "Reason": f"shape_mismatch: raw{gray_image.shape} vs mask{mask.shape}",
                        })
                        continue

                    labels_path = opj(labels_npy_dir, f"{tile_name}.npy")
                    if os.path.exists(labels_path):
                        labeled_mask = np.load(labels_path)
                    else:
                        labeled_mask = measure.label(mask > 0, connectivity=LABEL_CONNECTIVITY)

                    overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    object_count = int(labeled_mask.max())
                    if object_count == 0:
                        Image.fromarray(overlay).save(overlay_path)
                        continue

                    props = measure.regionprops(labeled_mask, intensity_image=gray_image, cache=True)
                    for region in props:
                        light_pixel_mask, _, _, _ = _classify_light_dark_pixels(
                            region.image_intensity, region.image, foreground_polarity, min_eta_squared,
                        )
                        min_row, min_col, max_row, max_col = region.bbox
                        region_overlay = overlay[min_row:max_row, min_col:max_col]
                        region_overlay[region.image & ~light_pixel_mask] = (255, 255, 255)
                        region_overlay[region.image & light_pixel_mask] = light_color_rgb

                    Image.fromarray(overlay).save(overlay_path)
                except Exception as exc:
                    failed.append({"Image_ID": tile_name, "Reason": str(exc)})
                    continue

            if failed:
                metrics_path = opj(processed_data_dir, 'inference_metrics')
                os.makedirs(metrics_path, exist_ok=True)
                pd.DataFrame(failed).to_csv(opj(metrics_path, 'light_dark_overlay_failed.csv'), index=False)
