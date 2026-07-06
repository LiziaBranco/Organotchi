from experiment_info import *

import csv
import os
import re
from datetime import datetime

from PIL import Image
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
DRY_RUN = False
DISCOVER_RAW_DATA_FOLDERS = True


def _natural_sort_key(path):
    filename = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", filename)
    ]


def _day_sort_key(day):
    try:
        return 0, float(day)
    except (TypeError, ValueError):
        return 1, str(day)


def _get_batch_days(batch):
    days = set()
    for run in get_segmentation_runs(batch):
        days.update(run.get("days", []))

    for legacy_key in ("days", "days_96", "days_24"):
        if batch.get(legacy_key):
            days.update(batch[legacy_key])

    return sorted(days, key=_day_sort_key)


def _collect_images(directory):
    if not os.path.isdir(directory):
        return []

    image_paths = []
    for entry in os.scandir(directory):
        if not entry.is_file():
            continue
        _, extension = os.path.splitext(entry.name)
        if extension.lower() in IMAGE_EXTENSIONS:
            image_paths.append(entry.path)

    return sorted(image_paths, key=_natural_sort_key)


def _target_name(index, extension):
    target_extension = ".jpg" if extension.lower() == ".png" else extension.lower()
    return f"Tile{index:06d}{target_extension}"


def _plan_action(source_path):
    extension = os.path.splitext(source_path)[1].lower()
    return "convert_png_to_jpg" if extension == ".png" else "rename"


def _build_rename_plan(image_paths):
    candidates = []
    target_paths = set()

    for index, source_path in enumerate(image_paths, start=1):
        directory = os.path.dirname(source_path)
        _, extension = os.path.splitext(source_path)
        target_path = os.path.join(directory, _target_name(index, extension))
        source_abs = os.path.abspath(source_path)
        target_abs = os.path.abspath(target_path)
        action = _plan_action(source_path)
        needs_action = source_abs != target_abs or action != "rename"

        if target_abs in target_paths:
            raise RuntimeError(f"Duplicate rename target detected: {target_path}")
        target_paths.add(target_abs)
        candidates.append((source_path, target_path, action, source_abs, target_abs, needs_action))

    moved_source_paths = {source_abs for *_, source_abs, _, needs_action in candidates if needs_action}
    plan = []

    for source_path, target_path, action, source_abs, target_abs, needs_action in candidates:
        if not needs_action:
            continue

        if os.path.exists(target_path) and target_abs != source_abs and target_abs not in moved_source_paths:
            raise FileExistsError(
                f"Cannot rename because target already exists and will not be moved: {target_path}"
            )

        plan.append((source_path, target_path, action))

    return plan


def _write_rename_log(directory, plan):
    if not plan:
        return

    log_path = os.path.join(directory, "tile_rename_log.csv")
    timestamp = datetime.now().isoformat(timespec="seconds")
    file_exists = os.path.exists(log_path)

    with open(log_path, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "original_name", "new_name"],
        )
        if not file_exists:
            writer.writeheader()

        for source_path, target_path, _ in plan:
            writer.writerow({
                "timestamp": timestamp,
                "original_name": os.path.basename(source_path),
                "new_name": os.path.basename(target_path),
            })


def _scale_array_to_uint8(array):
    array = np.asarray(array)
    if array.dtype == np.uint8 and array.size and array.min() <= 0 and array.max() >= 255:
        return array

    array = array.astype(float)
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    min_value = finite_values.min()
    max_value = finite_values.max()
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (array - min_value) / (max_value - min_value) * 255
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _png_to_rgb_image(source_path):
    with Image.open(source_path) as img:
        if img.mode == "P":
            img = img.convert("RGBA" if "transparency" in img.info else "RGB")

        image_array = np.array(img)

    if image_array.ndim == 2:
        image_array = _scale_array_to_uint8(image_array)
        return Image.fromarray(image_array, mode="L").convert("RGB")

    if image_array.ndim == 3 and image_array.shape[-1] in {2, 4}:
        signal_array = image_array[..., :-1]
        alpha_array = image_array[..., -1]
        if np.nanmax(signal_array.astype(float)) > np.nanmin(signal_array.astype(float)):
            image_array = signal_array
        else:
            image_array = alpha_array

    image_array = _scale_array_to_uint8(image_array)
    if image_array.ndim == 2:
        return Image.fromarray(image_array, mode="L").convert("RGB")

    if image_array.shape[-1] == 1:
        image_array = image_array[..., 0]
        return Image.fromarray(image_array, mode="L").convert("RGB")

    return Image.fromarray(image_array[..., :3], mode="RGB")


def _save_png_as_jpg(source_path, target_path):
    _png_to_rgb_image(source_path).save(target_path, "JPEG", quality=95)


def _apply_rename_plan(plan):
    if not plan:
        return

    temp_plan = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

    for index, (source_path, target_path, action) in enumerate(plan, start=1):
        directory = os.path.dirname(source_path)
        _, extension = os.path.splitext(source_path)
        temp_path = os.path.join(
            directory,
            f".tile_rename_tmp_{timestamp}_{index:06d}{extension}",
        )
        os.rename(source_path, temp_path)
        temp_plan.append((temp_path, target_path, action))

    for temp_path, target_path, action in temp_plan:
        if action == "convert_png_to_jpg":
            _save_png_as_jpg(temp_path, target_path)
            os.remove(temp_path)
        else:
            os.rename(temp_path, target_path)


def rename_images_in_directory(directory):
    image_paths = _collect_images(directory)
    if not image_paths:
        print(f"Skipping {directory}: no tif/jpg/png images found.")
        return 0

    plan = _build_rename_plan(image_paths)
    if not plan:
        print(f"Skipping {directory}: filenames already match Tile000001.ext format.")
        return 0

    print(f"\nRenaming/converting {len(plan)} image(s) in {directory}")
    for source_path, target_path, action in plan[:5]:
        action_label = "convert" if action == "convert_png_to_jpg" else "rename"
        print(f"  {action_label}: {os.path.basename(source_path)} -> {os.path.basename(target_path)}")
    if len(plan) > 5:
        print(f"  ... and {len(plan) - 5} more")

    if DRY_RUN:
        print("  Dry run enabled; no files were renamed.")
        return len(plan)

    _write_rename_log(directory, plan)
    _apply_rename_plan(plan)
    return len(plan)


def _metadata_raw_data_dirs():
    raw_data_dirs = []
    for batch in batches_info:
        batch_name = batch["batch"]
        for day in _get_batch_days(batch):
            raw_data_dirs.append(os.path.join(
                base_dir,
                project_dir,
                "raw_data",
                f"Day {day}",
                batch_name,
            ))
    return raw_data_dirs


def _discover_raw_data_dirs():
    raw_data_root = os.path.join(base_dir, project_dir, "raw_data")
    if not os.path.isdir(raw_data_root):
        return []

    raw_data_dirs = []
    for day_entry in os.scandir(raw_data_root):
        if not day_entry.is_dir() or not day_entry.name.startswith("Day "):
            continue
        for batch_entry in os.scandir(day_entry.path):
            if not batch_entry.is_dir() or batch_entry.name == "MetaData":
                continue
            if _collect_images(batch_entry.path):
                raw_data_dirs.append(batch_entry.path)

    return sorted(raw_data_dirs, key=_natural_sort_key)


def _raw_data_dirs_to_process():
    raw_data_dirs = _metadata_raw_data_dirs()
    if DISCOVER_RAW_DATA_FOLDERS:
        raw_data_dirs.extend(_discover_raw_data_dirs())

    unique_dirs = []
    seen = set()
    for directory in raw_data_dirs:
        directory_abs = os.path.abspath(directory)
        if directory_abs in seen:
            continue
        seen.add(directory_abs)
        unique_dirs.append(directory)

    return unique_dirs


total_renamed = 0
for raw_data_dir in _raw_data_dirs_to_process():
    total_renamed += rename_images_in_directory(raw_data_dir)

print(f"\nDone. Planned/renamed/converted {total_renamed} image file(s).")
