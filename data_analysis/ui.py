import json
import mimetypes
import os
import re
import urllib.parse
import uuid

import dash
import dash_bootstrap_components as dbc
import flask
from dash import ALL, Input, Output, State, dcc, html


DEFAULT_PATH = os.path.realpath(os.getcwd())

IMAGE_VIEWER_SUBDIRS = {
    "seg": "processed_data",
    "stats": "plots",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".tif"}

##################################################
# Helper functions
##################################################


def _list_images(folder):
    try:
        return sorted(os.path.join(folder, e.name) for e in os.scandir(folder)
                    if e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_EXTENSIONS
                )
    except OSError:
        return []


def new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


##################################################
# Run Layout
##################################################

SEGMENTATION_METHODS = [
    {"label": "Classical", "value": "cl"},
    {"label": "Deep Learning", "value": "dl"},
]

RUN_FIELD_SPECS = [
    {"key": "name", "label": "Name:", "type": "text"},
    {
        "key": "method",
        "label": "Segmentation Method:",
        "type": "dropdown",
        "options": SEGMENTATION_METHODS,
    },
    {
        "key": "tile_grid",
        "label": "Tile grid:",
        "type": "dropdown",
        "options": [{"label": f"{n}x{n}", "value": f"{n}x{n}"} for n in range(1, 10)],
    },
    {"key": "tile_order", "label": "Tile Order:", "type": "text", "placeholder": "e.g. column_snake"},
    {
        "key": "stitched_images_per_batch",
        "label": "Stitched images per batch:",
        "type": "number",
        "placeholder": "e.g. 4",
    },
    {"key": "magnification", "label": "Magnification:", "type": "text", "placeholder": "e.g. 4x"},
    {"key": "images_per_plate", "label": "Images per Plate:", "type": "number", "placeholder": "e.g. 24"},
    {
        "key": "organoids_per_image",
        "label": "Organoids per image:",
        "type": "dropdown",
        "options": [
            {"label": "Single", "value": "single"},
            {"label": "Multiple", "value": "multiple"},
        ],
    },
    {"key": "pixel_to_micron_key", "label": "Pixel to micron key:", "type": "text", "placeholder": "e.g. 1.25x"},
    {"key": "threshold", "label": "Threshold:", "type": "number", "placeholder": "e.g. 128"},
    {"key": "min_area", "label": "Minimum area:", "type": "number", "placeholder": "e.g. 30000"},
    {"key": "max_area", "label": "Maximum area:", "type": "number", "placeholder": "e.g. 3000000"},
    {
        "key": "min_circularity",
        "label": "Minimum circularity:",
        "type": "number",
        "step": "any",
        "placeholder": "e.g. 0.2",
    },
    {
        "key": "filter_masks",
        "label": "Filter masks:",
        "type": "dropdown",
        "options": [{"label": "True", "value": True}, {"label": "False", "value": False}],
    },
]

# Second-column fields depending on chosen Segmentation Method.
FIELD_METHOD = {
    "threshold": "cl",
    "min_area": "cl",
    "max_area": "cl",
    "min_circularity": "cl",
    "filter_masks": "dl",
}

# Second-column fields, only shown when Tile grid > 1x1
TILE_DEPENDENT_FIELDS = {"tile_order", "stitched_images_per_batch"}


def new_run(index):
    return {
        "id": new_id("run"),
        "name": f"Segmentation Run {index}",
        "method": "cl",
        "tile_grid": "1x1",
        "tile_order": "",
        "stitched_images_per_batch": None,
        "magnification": "",
        "images_per_plate": None,
        "organoids_per_image": "single",
        "pixel_to_micron_key": "",
        "threshold": None,
        "min_area": None,
        "max_area": None,
        "min_circularity": None,
        "filter_masks": True,
        "days": [],
    }


def render_run_field(run, spec):
    field_id = {"type": "run-field", "field": spec["key"], "index": run["id"]}
    value = run.get(spec["key"])
    if spec["type"] == "dropdown":
        control = dbc.Select(id=field_id, options=spec["options"], value=value)
    elif spec["type"] == "number":
        control = dbc.Input(
            id=field_id,
            type="number",
            min=spec.get("min", 0),
            step=spec.get("step", 1),
            value=value,
            placeholder=spec.get("placeholder", ""),
        )
    else:
        control = dbc.Input(
            id=field_id,
            type="text",
            value=value,
            placeholder=spec.get("placeholder", ""),
        )
    return dbc.Row(
        [
            dbc.Col(html.Label(spec["label"]), width=3),
            dbc.Col(control),
        ],
        className="mb-2 align-items-center",
    )


def new_day():
    return {"id": new_id("day"), "value": None}


def render_day_row(run_id, day):
    return dbc.Row(
        [
            dbc.Col(
                dbc.Input(
                    id={"type": "day-field", "run": run_id, "day": day["id"]},
                    type="number",
                    min=0,
                    step=1,
                    value=day["value"],
                    placeholder="e.g. 3",
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Button(
                    "×",
                    id={"type": "remove-day-btn", "run": run_id, "day": day["id"]},
                    size="sm",
                    outline=True,
                    color="secondary",
                ),
                width="auto",
            ),
        ],
        className="mb-1 align-items-center",
    )


def render_days_section(run):
    return dbc.Row(
        [
            dbc.Col(html.Label("Days:"), width=3),
            dbc.Col(
                [
                    html.Div(
                        [render_day_row(run["id"], d) for d in run["days"]],
                        id={"type": "days-list", "run": run["id"]},
                    ),
                    dbc.Button(
                        "+ Add day",
                        id={"type": "add-day-btn", "run": run["id"]},
                        size="sm",
                        outline=True,
                        color="secondary",
                    ),
                ]
            ),
        ],
        className="mb-2",
        style={"minHeight": "38px"},
    )


def render_tile_dependent_row(run, field_key):
    field_spec = next(spec for spec in RUN_FIELD_SPECS if spec["key"] == field_key)
    return html.Div(
        render_run_field(run, field_spec),
        id={"type": "tile-dependent-row", "field": field_key, "index": run["id"]},
        style={"display": "none"} if run.get("tile_grid") == "1x1" else {},
    )


def render_method_dependent_row(run, field_key):
    field_spec = next(spec for spec in RUN_FIELD_SPECS if spec["key"] == field_key)
    return html.Div(
        render_run_field(run, field_spec),
        id={"type": "method-dependent-row", "field": field_key, "index": run["id"]},
        style={} if run.get("method") == FIELD_METHOD[field_key] else {"display": "none"},
    )


def render_run_tab(run, index):
    dedicated_row_keys = {"tile_grid", "organoids_per_image", "pixel_to_micron_key"}
    skip_keys = set(FIELD_METHOD) | TILE_DEPENDENT_FIELDS | dedicated_row_keys
    left_top = []
    for spec in RUN_FIELD_SPECS:
        if spec["key"] in skip_keys:
            continue
        left_top.append(render_run_field(run, spec))
        if spec["key"] == "method":
            left_top.append(render_days_section(run))
    right_top = [
        html.Div([render_method_dependent_row(run, "threshold"), render_method_dependent_row(run, "filter_masks")]),
        render_method_dependent_row(run, "min_area"),
        render_method_dependent_row(run, "max_area"),
        render_method_dependent_row(run, "min_circularity"),
    ]
    top_block = dbc.Row([dbc.Col(left_top, width=6), dbc.Col(right_top, width=6)])

    def field(key):
        return next(s for s in RUN_FIELD_SPECS if s["key"] == key)

    tile_grid_row = dbc.Row(
        [
            dbc.Col(render_run_field(run, field("tile_grid")), width=6),
            dbc.Col(render_tile_dependent_row(run, "tile_order"), width=6),
        ]
    )
    organoids_row = dbc.Row(
        [
            dbc.Col(render_run_field(run, field("organoids_per_image")), width=6),
            dbc.Col(render_tile_dependent_row(run, "stitched_images_per_batch"), width=6),
        ]
    )
    bottom_row = dbc.Row([dbc.Col(render_run_field(run, field("pixel_to_micron_key")), width=6), dbc.Col(width=6)])

    content = dbc.Card(
        dbc.CardBody([top_block, tile_grid_row, organoids_row, bottom_row]),
        className="mt-2",
    )
    return dbc.Tab(
        content,
        label=run["name"] or f"Segmentation Run {index}",
        tab_id=run["id"],
        id={"type": "run-tab", "index": run["id"]},
        key=run["id"],
    )


def render_runs_tabs(runs):
    return [render_run_tab(r, i + 1) for i, r in enumerate(runs)]


##################################################
# Batch Layout
##################################################


def new_batch(index):
    return {
        "id": new_id("batch"),
        "batch": f"Batch {index}",
        "num_plates": None,
        "segmentation_runs": [new_run(1)],
    }

DEFAULT_CONFIG = {
    "data_path": "",
    "batches_info": [new_batch(1)],
}

def render_batch_tab(batch, index):
    content = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Label("Batch name:"), width=3),
                        dbc.Col(
                            dbc.Input(
                                id={"type": "batch-field", "field": "batch", "index": batch["id"]},
                                type="text",
                                value=batch["batch"],
                            )
                        ),
                    ],
                    className="mb-2 align-items-center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("No. plates:"), width=3),
                        dbc.Col(
                            dbc.Input(
                                id={"type": "batch-field", "field": "num_plates", "index": batch["id"]},
                                type="number",
                                min=0,
                                step=1,
                                value=batch["num_plates"],
                            )
                        ),
                    ],
                    className="align-items-center",
                ),
                html.Hr(),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Tabs(
                                        id={"type": "run-tabs", "batch": batch["id"]},
                                        active_tab=batch["segmentation_runs"][0]["id"],
                                        children=render_runs_tabs(batch["segmentation_runs"]),
                                    )
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "+",
                                        id={"type": "add-run-btn", "batch": batch["id"]},
                                        size="sm",
                                        outline=True,
                                        color="secondary",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        [html.I(className="bi bi-x"), " Remove run"],
                                        id={"type": "remove-run-btn", "batch": batch["id"]},
                                        size="sm",
                                        outline=True,
                                        color="danger",
                                    ),
                                    width="auto",
                                ),
                            ],
                            align="start",
                        ),
                    ],
                ),
            ]
        ),
        className="mt-2",
    )
    return dbc.Tab(
        content,
        label=batch["batch"] or f"Batch {index}",
        tab_id=batch["id"],
        id={"type": "batch-tab", "index": batch["id"]},
        key=batch["id"],
    )


def render_batches_tabs(batches):
    return [render_batch_tab(b, i + 1) for i, b in enumerate(batches)]


##################################################
# Overall Layout
##################################################

def build_configuration_tab_content():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.Label("Load data:"), width=2),
                    dbc.Col(dbc.Input(id="data-path-input", type="text", value=DEFAULT_PATH)),
                    dbc.Col(
                        dbc.Button(html.I(className="bi bi-folder2-open"), id="browse-btn", outline=True, color="secondary"),
                        width="auto",
                    ),
                ],
                className="my-4 align-items-center",
            ),
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(dbc.Tabs(id="batch-tabs", active_tab=DEFAULT_CONFIG["batches_info"][0]["id"],
                                             children=render_batches_tabs(DEFAULT_CONFIG["batches_info"]))),
                            dbc.Col(dbc.Button("+", id="add-batch-btn", size="sm", outline=True, color="secondary"),
                                    width="auto"),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-x"), " Remove batch"],
                                    id="remove-batch-btn",
                                    size="sm",
                                    outline=True,
                                    color="danger",
                                ),
                                width="auto",
                            ),
                        ],
                        align="start",
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Save JSON", id="save-json-btn", color="primary"), width="auto"),
                    dbc.Col(
                        dbc.Button(
                            "Run Segmentations",
                            id="run-seg-btn",
                            color="secondary",
                            style={"opacity": "0.5", "cursor": "not-allowed"},
                        ),
                        width="auto",
                    ),
                ],
            ),
        ]
    )


def build_image_viewer_content(prefix):
    return html.Div(
        [
            dcc.Store(id=f"{prefix}-image-list-store", data=[]),
            dcc.Store(id=f"{prefix}-image-index-store", data=0),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("◀", id=f"{prefix}-image-prev-btn", outline=True, color="secondary", disabled=True),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Div(id=f"{prefix}-image-name-display", className="text-center fw-bold"),
                    ),
                    dbc.Col(
                        dbc.Button("▶", id=f"{prefix}-image-next-btn", outline=True, color="secondary", disabled=True),
                        width="auto",
                    ),
                ],
                className="justify-content-center align-items-center mt-2 mb-3",
            ),
            html.Div(
                html.Img(id=f"{prefix}-image-display", style={"maxWidth": "100%", "maxHeight": "75vh", "objectFit": "contain"}),
                className="text-center",
            ),
        ],
        className="p-4",
    )


def build_segmentations_tab_content():
    return build_image_viewer_content("seg")


def build_statistical_results_tab_content():
    return build_image_viewer_content("stats")


def build_layout():
    return dbc.Container(
        fluid=True,
        className="p-4",
        children=[
            dcc.Store(id="config-store", data=DEFAULT_CONFIG),
            dcc.Store(id="last-saved-config-store", data=None),
            dcc.Store(id="browse-path-store", data=DEFAULT_PATH),
            dbc.Modal(
                id="browse-modal",
                is_open=False,
                size="lg",
                children=[
                    dbc.ModalHeader(dbc.ModalTitle("Select data folder")),
                    dbc.ModalBody(
                        [
                            html.Div(id="browse-current-path", className="text-muted mb-2 font-monospace small"),
                            dbc.Button("⬆ Up", id="browse-up-btn", size="sm", outline=True, color="secondary", className="mb-2"),
                            html.Div(id="browse-dir-list"),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Cancel", id="browse-cancel-btn", outline=True, color="secondary"),
                            dbc.Button("Select this folder", id="browse-select-btn", color="primary"),
                        ]
                    ),
                ],
            ),
            dbc.Modal(
                id="overwrite-confirm-modal",
                is_open=False,
                children=[
                    dbc.ModalHeader(dbc.ModalTitle("Overwrite existing file?")),
                    dbc.ModalBody(id="overwrite-confirm-body"),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Cancel", id="overwrite-cancel-btn", outline=True, color="secondary"),
                            dbc.Button("Overwrite", id="overwrite-confirm-btn", color="danger"),
                        ]
                    ),
                ],
            ),
            dbc.Modal(
                id="warning-modal",
                is_open=False,
                children=[
                    dbc.ModalHeader(dbc.ModalTitle("Warning", id="warning-modal-title")),
                    dbc.ModalBody(id="warning-modal-body", style={"whiteSpace": "pre-wrap"}),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="warning-modal-cancel-btn",
                                outline=True,
                                color="secondary",
                                style={"display": "none"},
                            ),
                            dbc.Button("OK", id="warning-modal-ok-btn", color="primary"),
                        ]
                    ),
                ],
            ),
            html.H3("Organotchi"),
            dbc.Tabs(
                id="main-tabs",
                active_tab="tab-config",
                children=[
                    dbc.Tab(label="Configuration", tab_id="tab-config", children=build_configuration_tab_content()),
                    dbc.Tab(label="Segmentations", tab_id="tab-segmentations", children=build_segmentations_tab_content()),
                    dbc.Tab(label="Statistical results", tab_id="tab-stats", children=build_statistical_results_tab_content()),
                ],
            ),
            dbc.Alert(id="status-alert", is_open=False, dismissable=True, className="mt-3", style={"whiteSpace": "pre-wrap"}),
        ],
    )


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
app.layout = build_layout()


@app.server.route("/organotchi-image")
def serve_image():
    path = flask.request.args.get("path", "")
    if (
        not path
        or not os.path.isfile(path)
        or os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS
    ):
        flask.abort(404)
    mimetype, _ = mimetypes.guess_type(path)
    return flask.send_file(path, mimetype=mimetype)


##################################################
# Callbacks — Adding/Removing batches/runs
##################################################

def _triggered_id_or_prevent_update():
    triggered_id = dash.ctx.triggered_id
    if not triggered_id or not dash.ctx.triggered[0]["value"]:
        raise dash.exceptions.PreventUpdate
    return triggered_id


@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("batch-tabs", "children", allow_duplicate=True),
    Output("batch-tabs", "active_tab", allow_duplicate=True),
    Input("add-batch-btn", "n_clicks"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def add_batch(n_clicks, config):
    config = dict(config)
    batch = new_batch(len(config["batches_info"]) + 1)
    config["batches_info"] = config["batches_info"] + [batch]
    return config, render_batches_tabs(config["batches_info"]), batch["id"]


@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    Output({"type": "run-tabs", "batch": ALL}, "children", allow_duplicate=True),
    Output({"type": "run-tabs", "batch": ALL}, "active_tab", allow_duplicate=True),
    Input({"type": "add-run-btn", "batch": ALL}, "n_clicks"),
    State({"type": "run-tabs", "batch": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def add_run(n_clicks_list, run_tabs_ids, config):
    triggered_id = _triggered_id_or_prevent_update()
    batch_id = triggered_id["batch"]

    config = dict(config)
    new_run_obj = None
    new_batches = []
    for b in config["batches_info"]:
        if b["id"] == batch_id:
            new_run_obj = new_run(len(b["segmentation_runs"]) + 1)
            b = {**b, "segmentation_runs": b["segmentation_runs"] + [new_run_obj]}
        new_batches.append(b)
    config["batches_info"] = new_batches

    runs_by_batch = {b["id"]: b["segmentation_runs"] for b in config["batches_info"]}
    children = [render_runs_tabs(runs_by_batch.get(tid["batch"], [])) for tid in run_tabs_ids]
    active_tabs = [
        new_run_obj["id"] if tid["batch"] == batch_id else dash.no_update
        for tid in run_tabs_ids
    ]
    return config, children, active_tabs


@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("batch-tabs", "children", allow_duplicate=True),
    Output("batch-tabs", "active_tab", allow_duplicate=True),
    Input("remove-batch-btn", "n_clicks"),
    State("batch-tabs", "active_tab"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def remove_batch(n_clicks, active_batch_id, config):
    if not active_batch_id:
        raise dash.exceptions.PreventUpdate

    config = dict(config)
    old_batches = config["batches_info"]
    removed_index = next((i for i, b in enumerate(old_batches) if b["id"] == active_batch_id), None)
    if removed_index is None:
        raise dash.exceptions.PreventUpdate
    config["batches_info"] = [b for b in old_batches if b["id"] != active_batch_id]

    if config["batches_info"]:
        new_active = config["batches_info"][max(removed_index - 1, 0)]["id"]
    else:
        new_active = None
    return config, render_batches_tabs(config["batches_info"]), new_active


@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    Output({"type": "run-tabs", "batch": ALL}, "children", allow_duplicate=True),
    Output({"type": "run-tabs", "batch": ALL}, "active_tab", allow_duplicate=True),
    Input({"type": "remove-run-btn", "batch": ALL}, "n_clicks"),
    State({"type": "run-tabs", "batch": ALL}, "active_tab"),
    State({"type": "run-tabs", "batch": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def remove_run(n_clicks_list, active_tabs_list, run_tabs_ids, config):
    triggered_id = _triggered_id_or_prevent_update()
    owning_batch_id = triggered_id["batch"]

    active_by_batch = {tid["batch"]: at for tid, at in zip(run_tabs_ids, active_tabs_list)}
    run_id = active_by_batch.get(owning_batch_id)
    if not run_id:
        raise dash.exceptions.PreventUpdate

    config = dict(config)
    removed_index = None
    new_batches = []
    for b in config["batches_info"]:
        runs = b["segmentation_runs"]
        if b["id"] == owning_batch_id:
            removed_index = next((i for i, r in enumerate(runs) if r["id"] == run_id), None)
            if removed_index is not None:
                runs = [r for r in runs if r["id"] != run_id]
        new_batches.append({**b, "segmentation_runs": runs})
    if removed_index is None:
        raise dash.exceptions.PreventUpdate
    config["batches_info"] = new_batches

    runs_by_batch = {b["id"]: b["segmentation_runs"] for b in config["batches_info"]}
    children = [render_runs_tabs(runs_by_batch.get(tid["batch"], [])) for tid in run_tabs_ids]
    active_tabs = []
    for tid in run_tabs_ids:
        if tid["batch"] != owning_batch_id:
            active_tabs.append(dash.no_update)
            continue
        remaining = runs_by_batch.get(owning_batch_id, [])
        if remaining:
            active_tabs.append(remaining[max(removed_index - 1, 0)]["id"])
        else:
            active_tabs.append(None)
    return config, children, active_tabs


##################################################
# Callbacks — value sync
##################################################

@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    Input({"type": "batch-field", "field": ALL, "index": ALL}, "value"),
    State({"type": "batch-field", "field": ALL, "index": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def sync_batch_fields(values, ids, config):
    updates = {}
    for id_dict, value in zip(ids, values):
        updates[(id_dict["index"], id_dict["field"])] = value
    config = dict(config)
    config["batches_info"] = [
        {**b, **{f: updates[(b["id"], f)] for f in ("batch", "num_plates") if (b["id"], f) in updates}}
        for b in config["batches_info"]
    ]
    return config


@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    Input({"type": "run-field", "field": ALL, "index": ALL}, "value"),
    State({"type": "run-field", "field": ALL, "index": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def sync_run_fields(values, ids, config):
    updates = {}
    for id_dict, value in zip(ids, values):
        updates[(id_dict["index"], id_dict["field"])] = value
    field_keys = [spec["key"] for spec in RUN_FIELD_SPECS]
    config = dict(config)
    config["batches_info"] = [
        {
            **b,
            "segmentation_runs": [
                {**r, **{f: updates[(r["id"], f)] for f in field_keys if (r["id"], f) in updates}}
                for r in b["segmentation_runs"]
            ],
        }
        for b in config["batches_info"]
    ]
    return config


@app.callback(
    Output({"type": "tile-dependent-row", "field": ALL, "index": ALL}, "style"),
    Input({"type": "run-field", "field": "tile_grid", "index": ALL}, "value"),
    State({"type": "run-field", "field": "tile_grid", "index": ALL}, "id"),
    State({"type": "tile-dependent-row", "field": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)
def toggle_tile_dependent_rows(tile_grid_values, tile_grid_ids, row_ids):
    value_by_run = {i["index"]: v for i, v in zip(tile_grid_ids, tile_grid_values)}
    return [{"display": "none"} if value_by_run.get(rid["index"]) == "1x1" else {} for rid in row_ids]


@app.callback(
    Output({"type": "method-dependent-row", "field": ALL, "index": ALL}, "style"),
    Input({"type": "run-field", "field": "method", "index": ALL}, "value"),
    State({"type": "run-field", "field": "method", "index": ALL}, "id"),
    State({"type": "method-dependent-row", "field": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)
def toggle_method_dependent_rows(method_values, method_ids, row_ids):
    method_by_run = {i["index"]: v for i, v in zip(method_ids, method_values)}
    return [
        {} if method_by_run.get(rid["index"]) == FIELD_METHOD[rid["field"]] else {"display": "none"}
        for rid in row_ids
    ]


def _all_runs(config):
    return [r for b in config["batches_info"] for r in b["segmentation_runs"]]


def _update_run_in_batches(config, run_id, updater):
    config = dict(config)
    config["batches_info"] = [
        {**b, "segmentation_runs": [updater(r) if r["id"] == run_id else r for r in b["segmentation_runs"]]}
        for b in config["batches_info"]
    ]
    return config


def _render_days_lists(runs, days_list_ids):
    days_by_run = {r["id"]: r["days"] for r in runs}
    return [
        [render_day_row(lid["run"], d) for d in days_by_run.get(lid["run"], [])]
        for lid in days_list_ids
    ]


@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    Output({"type": "days-list", "run": ALL}, "children"),
    Input({"type": "add-day-btn", "run": ALL}, "n_clicks"),
    State({"type": "days-list", "run": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def add_day(n_clicks_list, days_list_ids, config):
    triggered_id = _triggered_id_or_prevent_update()
    run_id = triggered_id["run"]
    config = _update_run_in_batches(config, run_id, lambda r: {**r, "days": r["days"] + [new_day()]})
    return config, _render_days_lists(_all_runs(config), days_list_ids)


@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    Output({"type": "days-list", "run": ALL}, "children", allow_duplicate=True),
    Input({"type": "remove-day-btn", "run": ALL, "day": ALL}, "n_clicks"),
    State({"type": "days-list", "run": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def remove_day(n_clicks_list, days_list_ids, config):
    triggered_id = _triggered_id_or_prevent_update()
    run_id, day_id = triggered_id["run"], triggered_id["day"]
    config = _update_run_in_batches(
        config, run_id, lambda r: {**r, "days": [d for d in r["days"] if d["id"] != day_id]}
    )
    return config, _render_days_lists(_all_runs(config), days_list_ids)


@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    Input({"type": "day-field", "run": ALL, "day": ALL}, "value"),
    State({"type": "day-field", "run": ALL, "day": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def sync_day_fields(values, ids, config):
    updates = {(i["run"], i["day"]): v for i, v in zip(ids, values)}
    config = dict(config)
    config["batches_info"] = [
        {
            **b,
            "segmentation_runs": [
                {
                    **r,
                    "days": [{**d, "value": updates.get((r["id"], d["id"]), d["value"])} for d in r["days"]],
                }
                for r in b["segmentation_runs"]
            ],
        }
        for b in config["batches_info"]
    ]
    return config


@app.callback(
    Output({"type": "batch-tab", "index": ALL}, "label"),
    Input({"type": "batch-field", "field": "batch", "index": ALL}, "value"),
    State({"type": "batch-field", "field": "batch", "index": ALL}, "id"),
    State({"type": "batch-tab", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def sync_batch_tab_labels(names, name_ids, tab_ids):
    name_by_index = {nid["index"]: (name or "").strip() for nid, name in zip(name_ids, names)}
    return [name_by_index.get(tid["index"]) or f"Batch {i + 1}" for i, tid in enumerate(tab_ids)]


@app.callback(
    Output({"type": "run-tab", "index": ALL}, "label"),
    Input({"type": "run-field", "field": "name", "index": ALL}, "value"),
    State({"type": "run-field", "field": "name", "index": ALL}, "id"),
    State({"type": "run-tab", "index": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def sync_run_tab_labels(names, name_ids, tab_ids, config):
    name_by_index = {nid["index"]: (name or "").strip() for nid, name in zip(name_ids, names)}
    position_by_run_id = {
        r["id"]: i + 1 for b in config["batches_info"] for i, r in enumerate(b["segmentation_runs"])
    }
    return [
        name_by_index.get(tid["index"]) or f"Segmentation Run {position_by_run_id.get(tid['index'], 1)}"
        for tid in tab_ids
    ]


@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("batch-tabs", "children", allow_duplicate=True),
    Output("batch-tabs", "active_tab", allow_duplicate=True),
    Output("last-saved-config-store", "data", allow_duplicate=True),
    Output("status-alert", "children", allow_duplicate=True),
    Output("status-alert", "color", allow_duplicate=True),
    Output("status-alert", "is_open", allow_duplicate=True),
    Output("warning-modal", "is_open", allow_duplicate=True),
    Output("warning-modal-body", "children", allow_duplicate=True),
    Output("warning-modal-title", "children", allow_duplicate=True),
    Output("warning-modal-cancel-btn", "style", allow_duplicate=True),
    Input("data-path-input", "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def sync_data_path(value, config):
    no_tabs_change = (dash.no_update, dash.no_update)
    no_saved_change = dash.no_update
    no_alert = (dash.no_update, dash.no_update, dash.no_update)
    no_warning = (dash.no_update, dash.no_update, dash.no_update, dash.no_update)
    hidden_cancel = {"display": "none"}

    if not value or not os.path.isdir(value):
        config = dict(config)
        config["data_path"] = value or ""
        return (config, *no_tabs_change, no_saved_change, *no_alert, *no_warning)

    metadata_path = os.path.join(value, "experiment_metadata.json")
    if not os.path.isfile(metadata_path):
        config = dict(config)
        config["data_path"] = value
        message = f'No "experiment_metadata.json" found in "{value}".'
        return (config, *no_tabs_change, no_saved_change, *no_alert, True, message, "Warning", hidden_cancel)

    try:
        with open(metadata_path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        config = dict(config)
        config["data_path"] = value
        return (
            config, *no_tabs_change, no_saved_change, *no_alert,
            True, f"Failed to load {metadata_path}: {e}", "Warning", hidden_cancel,
        )

    loaded = _json_to_config(raw, value)
    return (
        loaded,
        render_batches_tabs(loaded["batches_info"]),
        loaded["batches_info"][0]["id"],
        _config_to_json(loaded),
        *no_alert,
        True,
        f"Loaded existing config from {metadata_path}",
        "Info",
        hidden_cancel,
    )


##################################################
# Callbacks — folder browse
##################################################

@app.callback(
    Output("browse-modal", "is_open", allow_duplicate=True),
    Output("browse-path-store", "data", allow_duplicate=True),
    Input("browse-btn", "n_clicks"),
    State("data-path-input", "value"),
    prevent_initial_call=True,
)
def open_browse_modal(n_clicks, current_path):
    start = current_path if current_path and os.path.isdir(current_path) else DEFAULT_PATH
    return True, start


@app.callback(
    Output("browse-current-path", "children"),
    Output("browse-dir-list", "children"),
    Input("browse-path-store", "data"),
)
def render_browse_dir(path):
    try:
        with os.scandir(path) as it:
            entries = list(it)
    except OSError:
        entries = []
    dir_names = sorted(
        e.name for e in entries if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")
    )
    json_names = sorted(
        e.name for e in entries
        if e.is_file(follow_symlinks=False) and not e.name.startswith(".") and e.name.lower().endswith(".json")
    )
    dir_items = [
        dbc.ListGroupItem(
            [html.I(className="bi bi-folder me-2"), name],
            id={"type": "browse-dir-item", "path": os.path.join(path, name)},
            action=True,
            n_clicks=0,
        )
        for name in dir_names
    ]
    file_items = [
        dbc.ListGroupItem(
            [html.I(className="bi bi-file-earmark-code me-2"), name],
            id={"type": "browse-file-item", "path": os.path.join(path, name)},
            action=True,
            n_clicks=0,
        )
        for name in json_names
    ]
    items = dir_items + file_items or [dbc.ListGroupItem("(empty)", disabled=True)]
    return path, dbc.ListGroup(items)


@app.callback(
    Output("browse-path-store", "data", allow_duplicate=True),
    Input({"type": "browse-dir-item", "path": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def navigate_into_dir(n_clicks_list):
    return _triggered_id_or_prevent_update()["path"]


@app.callback(
    Output("browse-path-store", "data", allow_duplicate=True),
    Input("browse-up-btn", "n_clicks"),
    State("browse-path-store", "data"),
    prevent_initial_call=True,
)
def navigate_up(n_clicks, path):
    parent = os.path.dirname(path.rstrip(os.sep)) or os.sep
    return parent


@app.callback(
    Output("data-path-input", "value", allow_duplicate=True),
    Output("browse-modal", "is_open", allow_duplicate=True),
    Input("browse-select-btn", "n_clicks"),
    State("browse-path-store", "data"),
    prevent_initial_call=True,
)
def select_folder(n_clicks, path):
    return path, False


@app.callback(
    Output("data-path-input", "value", allow_duplicate=True),
    Output("browse-modal", "is_open", allow_duplicate=True),
    Input({"type": "browse-file-item", "path": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_file(n_clicks_list):
    file_path = _triggered_id_or_prevent_update()["path"]
    return os.path.dirname(file_path), False


@app.callback(
    Output("browse-modal", "is_open", allow_duplicate=True),
    Input("browse-cancel-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_browse(n_clicks):
    return False


##################################################
# Callbacks — image viewer
##################################################


def register_image_viewer_callbacks(prefix, active_tab_id):
    subdir = IMAGE_VIEWER_SUBDIRS[prefix]

    def _folder(config):
        data_path = config.get("data_path", "")
        return os.path.join(data_path, subdir) if data_path else ""

    def _subfolders(current_folder):
        paths = []
        if os.path.isdir(current_folder):
            for sub in os.listdir(current_folder):
                sub_folder = os.path.join(current_folder, sub)
                if os.path.isdir(sub_folder):
                    paths.append(sub_folder)
        return paths

    @app.callback(
        Output(f"{prefix}-image-list-store", "data"),
        Output(f"{prefix}-image-index-store", "data"),
        Input("main-tabs", "active_tab"),
        Input("config-store", "data"),
        prevent_initial_call=True,
    )
    def scan_images(active_tab, config):
        if active_tab != active_tab_id:
            raise dash.exceptions.PreventUpdate
        current_folder = _folder(config)
        images = []
        if prefix == "seg":
            day_folders = _subfolders(current_folder)
            for day_folder in day_folders:
                batch_folders = _subfolders(day_folder)
                for batch_folder in batch_folders:
                    inference_folder = os.path.join(batch_folder, "inference_comparison")
                    images.extend(_list_images(inference_folder))
        else:
            images.extend(_list_images(current_folder))

        if not current_folder or not os.path.isdir(current_folder):
            return [], 0
        return images, 0

    @app.callback(
        Output(f"{prefix}-image-index-store", "data", allow_duplicate=True),
        Input(f"{prefix}-image-prev-btn", "n_clicks"),
        Input(f"{prefix}-image-next-btn", "n_clicks"),
        State(f"{prefix}-image-index-store", "data"),
        State(f"{prefix}-image-list-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_image(prev_clicks, next_clicks, index, images):
        if not images:
            raise dash.exceptions.PreventUpdate
        step = -1 if dash.ctx.triggered_id == f"{prefix}-image-prev-btn" else 1
        return (index + step) % len(images)

    @app.callback(
        Output(f"{prefix}-image-name-display", "children"),
        Output(f"{prefix}-image-display", "src"),
        Output(f"{prefix}-image-prev-btn", "disabled"),
        Output(f"{prefix}-image-next-btn", "disabled"),
        Input(f"{prefix}-image-index-store", "data"),
        Input(f"{prefix}-image-list-store", "data"),
        State("config-store", "data"),
    )
    def render_image(index, images, config):
        if not images:
            return "No images found", None, True, True
        path = images[index]
        src = "/organotchi-image?" + urllib.parse.urlencode({"path": path})
        name = path
        return name, src, len(images) <= 1, len(images) <= 1


register_image_viewer_callbacks("seg", "tab-segmentations")
register_image_viewer_callbacks("stats", "tab-stats")


##################################################
# Callbacks — Save JSON / Run Segmentations
##################################################

def _run_to_json(r):
    return {
        **{spec["key"]: r[spec["key"]] for spec in RUN_FIELD_SPECS},
        "days": [d["value"] for d in r["days"]],
    }


def _config_to_json(config):
    return {
        "batches_info": [
            {
                "batch": b["batch"],
                "num_plates": b["num_plates"],
                "segmentation_runs": [_run_to_json(r) for r in b["segmentation_runs"]],
            }
            for b in config["batches_info"]
        ],
    }


def _json_to_run(r, index):
    run = new_run(index)
    for spec in RUN_FIELD_SPECS:
        if spec["key"] in r:
            run[spec["key"]] = r[spec["key"]]
    run["days"] = [{"id": new_id("day"), "value": v} for v in r.get("days", [])]
    return run


def _json_to_config(raw, data_path):
    batches = []
    for b in raw.get("batches_info", []):
        batch = new_batch(len(batches) + 1)
        batch["batch"] = b.get("batch", "")
        batch["num_plates"] = b.get("num_plates")
        segmentation_runs = [_json_to_run(r, i + 1) for i, r in enumerate(b.get("segmentation_runs", []))]
        batch["segmentation_runs"] = segmentation_runs or [new_run(1)]
        batches.append(batch)
    if not batches:
        batches = [new_batch(1)]

    return {"data_path": data_path, "batches_info": batches}


def _experiment_metadata_path(config):
    data_path = config.get("data_path", "")
    target_dir = data_path if os.path.isdir(data_path) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(target_dir, "experiment_metadata.json")


def _write_experiment_metadata(config, path):
    try:
        with open(path, "w") as f:
            json.dump(_config_to_json(config), f, indent=2)
        return True, f"Saved config to {path}"
    except OSError as e:
        return False, f"Failed to save config: {e}"


@app.callback(
    Output("overwrite-confirm-modal", "is_open", allow_duplicate=True),
    Output("overwrite-confirm-body", "children"),
    Output("status-alert", "children", allow_duplicate=True),
    Output("status-alert", "color", allow_duplicate=True),
    Output("status-alert", "is_open", allow_duplicate=True),
    Output("last-saved-config-store", "data", allow_duplicate=True),
    Output("warning-modal", "is_open", allow_duplicate=True),
    Output("warning-modal-body", "children", allow_duplicate=True),
    Output("warning-modal-title", "children", allow_duplicate=True),
    Output("warning-modal-cancel-btn", "style", allow_duplicate=True),
    Input("save-json-btn", "n_clicks"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def handle_save_click(n_clicks, config):
    path = _experiment_metadata_path(config)
    if os.path.isfile(path):
        return (
            True, f"{path} already exists. Overwrite it?",
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, False, dash.no_update, dash.no_update,
            dash.no_update,
        )
    ok, message = _write_experiment_metadata(config, path)
    if ok:
        return (
            False, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, _config_to_json(config), True,
            "Saved config as json-file.", "Info", {"display": "none"},
        )
    return (
        False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
        True, message, "Warning", {"display": "none"},
    )


@app.callback(
    Output("overwrite-confirm-modal", "is_open", allow_duplicate=True),
    Output("status-alert", "children", allow_duplicate=True),
    Output("status-alert", "color", allow_duplicate=True),
    Output("status-alert", "is_open", allow_duplicate=True),
    Output("last-saved-config-store", "data", allow_duplicate=True),
    Output("warning-modal", "is_open", allow_duplicate=True),
    Output("warning-modal-body", "children", allow_duplicate=True),
    Output("warning-modal-title", "children", allow_duplicate=True),
    Output("warning-modal-cancel-btn", "style", allow_duplicate=True),
    Input("overwrite-confirm-btn", "n_clicks"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def confirm_overwrite(n_clicks, config):
    path = _experiment_metadata_path(config)
    ok, message = _write_experiment_metadata(config, path)
    if ok:
        return (
            False, dash.no_update, dash.no_update, dash.no_update, _config_to_json(config), False,
            dash.no_update, dash.no_update, dash.no_update,
        )
    return (
        False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, message, "Warning",
        {"display": "none"},
    )


@app.callback(
    Output("run-seg-btn", "style"),
    Input("config-store", "data"),
    Input("last-saved-config-store", "data"),
)
def toggle_run_seg_style(config, last_saved):
    dirty = _config_to_json(config) != last_saved
    return {"opacity": "0.5", "cursor": "not-allowed"} if dirty else {}


@app.callback(
    Output("overwrite-confirm-modal", "is_open", allow_duplicate=True),
    Input("overwrite-cancel-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_overwrite(n_clicks):
    return False


def _is_blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _missing_fields(config):
    missing = []
    for b in config["batches_info"]:
        if _is_blank(b.get("batch")):
            missing.append("Batch name")
        if _is_blank(b.get("num_plates")):
            missing.append("No. plates")
        if not b.get("segmentation_runs"):
            missing.append("Segmentation run (at least one)")
        for r in b.get("segmentation_runs", []):
            for spec in RUN_FIELD_SPECS:
                key = spec["key"]
                if key in FIELD_METHOD and FIELD_METHOD[key] != r.get("method"):
                    continue  # not applicable for the selected method
                if key in TILE_DEPENDENT_FIELDS and r.get("tile_grid") == "1x1":
                    continue  # not shown when tile grid is 1x1
                if _is_blank(r.get(key)):
                    missing.append(spec["label"].rstrip(":"))
            if not r.get("days"):
                missing.append("Days (at least one)")
            for j, d in enumerate(r.get("days", []), start=1):
                if _is_blank(d.get("value")):
                    missing.append(f"Days entry {j}")
    return missing


DAY_FOLDER_RE = re.compile(r"^Day (\d+)")


def _raw_data_issue(data_path):
    raw_data_path = os.path.join(data_path, "raw_data")
    has_day_folder = any(
        entry.is_dir() and DAY_FOLDER_RE.match(entry.name) for entry in os.scandir(raw_data_path)
    )
    if not has_day_folder:
        return f'"{raw_data_path}" does not contain any folder starting with "Day " followed by a number.'
    return None


def _day_mismatch_issue(data_path, batches):
    raw_data_path = os.path.join(data_path, "raw_data")
    day_folder_numbers = {
        int(match.group(1))
        for entry in os.scandir(raw_data_path)
        if entry.is_dir()
        for match in [DAY_FOLDER_RE.match(entry.name)]
        if match
    }
    for b in batches:
        batch_label = b.get("batch") or "?"
        for i, r in enumerate(b.get("segmentation_runs", []), start=1):
            for d in r.get("days", []):
                value = d.get("value")
                if value is None:
                    continue  # already caught by the missing-fields check
                if int(value) not in day_folder_numbers:
                    return (
                        f'Batch "{batch_label}", Segmentation Run {i}: '
                        f'no "Day {int(value)}" folder found in "{raw_data_path}".'
                    )
    return None


@app.callback(
    Output("status-alert", "children", allow_duplicate=True),
    Output("status-alert", "color", allow_duplicate=True),
    Output("status-alert", "is_open", allow_duplicate=True),
    Output("warning-modal", "is_open", allow_duplicate=True),
    Output("warning-modal-body", "children", allow_duplicate=True),
    Output("warning-modal-title", "children", allow_duplicate=True),
    Output("warning-modal-cancel-btn", "style", allow_duplicate=True),
    Input("run-seg-btn", "n_clicks"),
    State("config-store", "data"),
    State("last-saved-config-store", "data"),
    prevent_initial_call=True,
)
def run_segmentations_stub(n_clicks, config, last_saved):
    hidden_cancel = {"display": "none"}

    if _config_to_json(config) != last_saved:
        message = "There are unsaved changes. Please save the configuration (Save JSON) before running segmentation."
        return dash.no_update, dash.no_update, dash.no_update, True, message, "Warning", hidden_cancel

    missing = _missing_fields(config)
    if missing:
        return (
            dash.no_update, dash.no_update, dash.no_update, True,
            html.Ul([html.Li(m) for m in missing]), "Warning", hidden_cancel,
        )

    if not config.get("data_path") or not os.path.isdir(config["data_path"]):
        message = "Set a valid data path on the Configuration tab before running segmentation."
        return dash.no_update, dash.no_update, dash.no_update, True, message, "Warning", hidden_cancel

    if not config["batches_info"]:
        return dash.no_update, dash.no_update, dash.no_update, True, "Add at least one batch.", "Warning", hidden_cancel

    raw_data_issue = _raw_data_issue(config["data_path"])
    if raw_data_issue:
        return dash.no_update, dash.no_update, dash.no_update, True, raw_data_issue, "Warning", hidden_cancel

    day_mismatch_issue = _day_mismatch_issue(config["data_path"], config["batches_info"])
    if day_mismatch_issue:
        return dash.no_update, dash.no_update, dash.no_update, True, day_mismatch_issue, "Warning", hidden_cancel

    preview = json.dumps(_config_to_json(config), indent=2)
    message = f"Run segmentation with the following config?\n{preview}"
    return dash.no_update, dash.no_update, dash.no_update, True, message, "Info", {}


@app.callback(
    Output("warning-modal", "is_open", allow_duplicate=True),
    Input("warning-modal-ok-btn", "n_clicks"),
    Input("warning-modal-cancel-btn", "n_clicks"),
    prevent_initial_call=True,
)
def close_warning_modal(ok_clicks, cancel_clicks):
    return False


if __name__ == "__main__":
    app.run(debug=True)
