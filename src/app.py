import os
import uuid
import orjson
import base64
import threading
from datetime import datetime

import flask
import dash
from redis import Redis
import plotly.express as px
from dash import html, dcc, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


# open source dash functionality
# https://github.com/np-8/dash-uploader
import dash_uploader as du

# Local Modules
from worker import poll_remove_user_data
from dataset import (
    move_unzip_uploaded_file,
    create_clusters_zip,
    setup_umap,
    get_labels,
    gen_img_uri,
    prepare_preview_download,
    del_folder_contents,
    find_similar_imgs,
)
from componentBuilder import (
    create_LR_label,
    gen_img_preview,
    gen_download_button,
    create_info_loading,
    create_title_with_button,
)
from figureGen import blankFig, generate_fig_3D, generate_fig_2D

################################################################################
""" Initialize Dash App, Redis, Uploader: """
################################################################################

server = flask.Flask(__name__)
# Initialize Cache
# use host="redis" if running redis server with docker #
# "127.0.0.1" if locally (without Docker)
redis_client = Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
# [TODO]: before deploying consider memory management i.e when should you clear redis
redis_client.flushall()
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.CYBORG])

du.configure_upload(app, "assets/temp")

################################################################################
""" Dash Components: """
################################################################################

horz_line = html.Hr()

# title of the dash
title = html.H4(
    "Dataset Clustering Dashboard",
    style={"color": "white", "text-align": "center", "padding": "10px"},
)

# ------------------------------------------------------------------------------
#   3D Graph and it's Control Components
# ------------------------------------------------------------------------------
graphWithLoadingAnimation = dcc.Loading(
    children=[
        dcc.Graph(
            id="mainGraph",
            clear_on_unhover=True,
            figure=blankFig(),
            loading_state={"is_loading": False},
            style={"height": "70vh"},
        )
    ],
    type="graph",
)
fileInfo = create_info_loading(
    id="fileInfo", children=["Please upload FolderOfImagesYouWishToCluster.zip"]
)
dataInfo = create_info_loading(
    id="dataInfo", children=["Then click the generate graphs button below."]
)
n_neighbors_left_text = (
    "This parameter controls how UMAP balances local versus global structure in the data. "
    "As n_neighbors is increased UMAP manages to see more of the overall "
    "structure of the data, gluing more components together, and better coverying "
    "the broader structure of the data. As n_neighbors increases further, more and "
    "more focus in placed on the overall structure of the data. This results in "
    "a plot where the overall structure is well captured, but at the loss of some of "
    "the finer local structure (individual images may no longer necessarily be "
    "immediately next to their closest match)."
)
n_neighbors_slider = [
    create_LR_label(
        id="n_neighbors_label",
        leftText="[UMAP]:",
        rightText="n_neighbors",
        tip_text_left=(
            "Uniform Manifold Approximation and Projection (UMAP) is a "
            "dimension reduction technique used here to allow visualisation."
        ),
        tip_text_right=n_neighbors_left_text,
    ),
    dcc.Slider(
        min=10,
        max=240,
        step=10,
        value=80,
        id="n_neighbors_slider",
        marks=None,
        tooltip={"placement": "bottom"},
    ),
]

min_dist_left_text = (
    "The min_dist parameter controls how tightly UMAP is allowed to pack points "
    "together. It, quite literally, provides the minimum distance apart that points "
    "are allowed to be in the low dimensional representation. This means that low "
    "values of min_dist will result in clumpier embeddings. This can be useful if "
    "you are interested in clustering, or in finer topological structure. Larger "
    "values of min_dist will prevent UMAP from packing points together and will "
    "focus on the preservation of the broad topological structure instead."
)
min_dist_slider = [
    create_LR_label(
        id="min_dist_label",
        leftText="[UMAP]:",
        rightText="min_dist",
        tip_text_left=(
            "Uniform Manifold Approximation and Projection (UMAP) is a "
            "dimension reduction technique used here to allow visualisation."
        ),
        tip_text_right=min_dist_left_text,
    ),
    dcc.Slider(
        min=0.0,
        max=1,
        step=0.01,
        value=0.0,
        id="min_dist_slider",
        marks=None,
        tooltip={"placement": "bottom"},
    ),
]

min_cluster_size_left_text = (
    "Set it to the smallest size grouping that you wish to consider a cluster. "
    "It can have slightly non-obvious effects however (unless you fix min_samples)."
)
min_cluster_size_slider = [
    create_LR_label(
        id="min_cluster_size_label",
        leftText="[HDBSCAN]:",
        rightText="min_cluster_size",
        tip_text_left="Hierarchical Density-Based Spatial Clustering of Applications with Noise.",
        tip_text_right=min_cluster_size_left_text,
    ),
    dcc.Slider(
        min=5,
        max=405,
        step=10,
        value=155,
        id="min_cluster_size_slider",
        marks=None,
        tooltip={"placement": "bottom"},
    ),
]

min_samples_left_text = (
    "The simplest intuition for what min_samples does is provide a measure of how "
    "conservative you want you clustering to be. The larger the value of min_samples "
    "you provide, the more conservative the clustering - more points will be "
    "declared as noise, and clusters will be restricted to progressively more "
    "dense areas. Steadily increasing min_samples will, as we saw in the examples "
    "above, make the clustering progressively more conservative."
)
min_samples_slider = [
    create_LR_label(
        id="min_samples_label",
        leftText="[HDBSCAN]:",
        rightText="min_samples",
        tip_text_left="Hierarchical Density-Based Spatial Clustering of Applications with Noise.",
        tip_text_right=min_samples_left_text,
    ),
    dcc.Slider(
        min=1,
        max=200,
        step=1,
        value=10,
        id="min_samples_slider",
        marks=None,
        tooltip={"placement": "bottom"},
    ),
]
download_clusters_button = gen_download_button(
    id="download_clusters_button",
    children=["Download Clusters"],
    href=app.get_asset_url("clusters.zip"),
)
graph3DButton = dbc.Button(
    children=["Generate Graphs"],
    id="graph3DButton",
    n_clicks=0,
    disabled=True,
    color="primary",
)
card3DButtons = html.Div(
    children=[html.Div(download_clusters_button), html.Div(graph3DButton)],
    style={"display": "flex", "justify-content": "space-around"},
)
controls_title = html.H5(
    "Control Parameters:",
    style={
        "text-align": "center",
        "padding": "10px",
    },
)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#   2D Graph Components
# ------------------------------------------------------------------------------
graph2D = dcc.Loading(
    children=[
        dcc.Graph(
            id="graph2D",
            clear_on_unhover=True,
            figure=blankFig(),
            loading_state={"is_loading": True},
        )
    ],
    type="graph",
)
imagePreview = html.Div(
    id="imagePreview",
    children=[
        "Use the box or lasso selector on the 2D graph to preview selected images."
    ],
    style={
        "textAlign": "center",
        #'margin': '10px'
    },
)
download_preview_button = gen_download_button(
    id="download_preview_button",
    children=["Download"],
    href=app.get_asset_url("preview_2D.zip"),
)
preview_download_name_input = html.Div(
    [
        dbc.Input(
            placeholder="Enter preview download name. Default: 'selection_preview'...",
            type="text",
        ),
    ]
)
preview_title = create_title_with_button([download_preview_button], html.Div())
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#   Image Similarity Search Components
# ------------------------------------------------------------------------------
upload_image_file_button = dcc.Upload(
    id="upload_image_file_button",
    children=["Drag/Select Image File"],
    style={
        "height": "41px",
        "lineHeight": "41px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
    },
    # Don't allow multiple files to be uploaded
    multiple=False,
    disabled=False,
)
download_search_button = html.Div(
    [
        gen_download_button(
            id="download_search_button",
            children=["Download Results"],
            href="",
        )
    ],
    style={"display": "flex", "align-items": "center", "justify-content": "center"},
)
# image_search_title = html.Div(
#     children=[
#         html.Div(
#             [upload_image_file_button],
#             style={
#                 "textAlign": "left",
#                 "width": "47%",
#                 "display": "inline-block",
#                 "margin-left": "2%",
#             },
#         ),
#         html.Div(
#             [download_search_button],
#             style={
#                 "textAlign": "right",
#                 "width": "50%",
#                 "display": "inline-block",
#             },
#         ),
#     ],
# )
image_file_info = create_info_loading(id="image_file_info", children=[""])
preview_test_image = html.Div(
    id="preview_test_image",
    children=[
        html.H6(
            children=[""],
            style={"textAlign": "center", "padding": "10px"},
        )
    ],
)
search_preview = html.Div(
    id="search_preview",
    children=["Upload image to find similar images."],
    style={
        "textAlign": "center",
        #'margin': '10px'
    },
)
search_preview_results_title = html.H6(
    children="Results:",
    style={
        "textAlign": "left",
        "padding": "10px",
    },
)
search_preview_main_title = html.H5(
    children="Reverse Image Search",
    style={
        "textAlign": "center",
        "padding": "10px",
    },
)
# ------------------------------------------------------------------------------

# Not in use currently
jump_up_button = dbc.Button(
    children=["Jump to Top"],
    id="jump_up_button",
    n_clicks=0,
    disabled=False,
    color="primary",
)

preview_and_search_card = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Selection Preview", tab_id="tab-1"),
                    dbc.Tab(label="Reverse Image Search", tab_id="tab-2"),
                ],
                id="card-tabs",
                active_tab="tab-1",
            )
        ),
        dbc.CardBody(html.Div(id="card-content")),
    ]
)

################################################################################
""" Dash UI Layout: """
################################################################################
def serve_layout():
    session_id = str(uuid.uuid4())
    # add user id to redis list
    redis_client.rpush("users", session_id)
    return dbc.Container(
        [
            # User timestamp updater
            dcc.Interval(
                id="interval-component",
                interval=1 * 20000,  # 20000 milliseconds = 20 sec
                n_intervals=0,
            ),
            # Empty div used since callbacks with no output are not allowed
            html.Div(id="dummy1"),
            # In browser storage objects
            dcc.Store(id="session-id", data=session_id),
            dcc.Store(id="dataProcessedFlag", data=False),
            dcc.Store(id="dataClusteredFlag", data=False),
            dbc.Row(title),
            dbc.Row(
                [
                    # Controls card here
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.Row(
                                    children=[
                                        dbc.Row(fileInfo),
                                        dbc.Row(dataInfo),
                                        dbc.Row(
                                            # This component is verbosely placed in layout as it
                                            # requires session_id to be dynamically generated
                                            html.Div(
                                                du.Upload(
                                                    id="dash_uploader",
                                                    text="Drag/Select Zip",
                                                    max_files=1,
                                                    filetypes=["zip"],
                                                    # changes default size breaks it when download starts
                                                    # default_style={
                                                    #     "overflow": "hide",
                                                    #     "minHeight": "2vh",
                                                    #     "lineHeight": "2vh",
                                                    # },
                                                    upload_id=session_id,
                                                ),
                                                style={  # wrapper div style
                                                    "textAlign": "center",
                                                    "width": "100%",
                                                    # "height": "50px",
                                                    "padding": "10px",
                                                    "display": "inline-block",
                                                },
                                            ),
                                        ),
                                        horz_line,
                                        # horz_line,
                                        dbc.Col(
                                            [
                                                controls_title,
                                                dbc.Row(n_neighbors_slider),
                                                dbc.Row(min_dist_slider),
                                                horz_line,
                                                dbc.Row(min_cluster_size_slider),
                                                dbc.Row(min_samples_slider),
                                            ]
                                        ),
                                        horz_line,
                                        dbc.Row(
                                            [card3DButtons],
                                            justify="center",
                                            align="center",
                                        ),
                                    ],
                                    justify="center",
                                    align="center",
                                ),
                            ],
                            body=True,
                            style={"height": "124vh"},
                        ),
                        md=3,
                        align="center",
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                children=[
                                    graphWithLoadingAnimation,
                                    dcc.Tooltip(
                                        id="mainGraphTooltip", direction="right"
                                    ),
                                    dcc.Download(id="mainGraphDownload"),
                                ],
                                style={"height": "70vh"},
                            ),
                            horz_line,
                            dbc.Row(
                                [
                                    # 2d graph here
                                    dbc.Col(
                                        children=[
                                            graph2D,
                                            dcc.Tooltip(
                                                id="graph2DTooltip",
                                                direction="right",
                                            ),
                                            dcc.Download(id="graph2DDownload"),
                                            horz_line,
                                            # tabbed card here
                                            dbc.Row(
                                                children=(
                                                    dbc.Col(
                                                        children=[
                                                            preview_and_search_card
                                                        ],
                                                        width="auto",
                                                        md=12,
                                                    ),
                                                ),
                                                justify="center",
                                                align="center",
                                            ),
                                            horz_line,
                                            # content will be rendered in this element
                                            html.Div(id="page-content"),
                                        ],
                                    ),
                                ],
                                style={"height": "50vh"},
                                justify="center",
                                align="start",
                            ),
                        ],
                        md=9,
                    ),
                ],
                justify="center",
                align="center",
            ),
        ],
        fluid=True,
    )


app.layout = serve_layout


################################################################################
""" Callback fxns: """
################################################################################
########################################################################
""" [CALLBACK]: upload zip file and extract features/paths """
########################################################################
@app.callback(
    [
        Output("dataProcessedFlag", "data"),
        Output("fileInfo", "children"),
        Output("graph3DButton", "disabled"),
    ],
    [Input("dash_uploader", "isCompleted")],
    [
        State("dash_uploader", "fileNames"),
        State("session-id", "data"),
    ],
)
def upload_data_reset_components(isCompleted, filename, session_id):
    # the content needs to be split. It contains the type and the real content
    if isCompleted:
        global redis_client
        # remove the extension part (.zip) from the filename
        dataset_name = os.path.splitext(filename[0])[0]
        # store dataset name in redis
        if not redis_client.sismember(f"{session_id}:datasets", dataset_name):
            # arrange and extract files
            move_unzip_uploaded_file(session_id, filename)
            # should only add dataset name to our redis Set, IF zip upload/extract succesful
            redis_client.sadd(f"{session_id}:datasets", dataset_name)
        # Update the current dataset being used by user
        redis_client.set(f"{session_id}:curr_dataset", dataset_name)

        outputText = create_LR_label(
            id="file_info_label",
            leftText="[FILE]:",
            rightText=dataset_name,
        )
        return [True], outputText, False
    return no_update, no_update, no_update


########################################################################
""" [CALLBACK]: upload image file and find similar images """
########################################################################
@app.callback(
    [
        Output("image_file_info", "children"),
        Output("preview_test_image", "children"),
        Output("search_preview", "children"),
        Output("download_search_button", "href"),
    ],
    [Input("upload_image_file_button", "contents")],
    [
        State("upload_image_file_button", "filename"),
        State("session-id", "data"),
        State("dataClusteredFlag", "data"),
    ],
)
def uploadData(content, filename, session_id, dataClusteredFlag):
    if content is not None and dataClusteredFlag:
        content_type, content_str = content.split(",")
        output_filename = (
            html.P(
                children=["[Filename]: ", html.Br(), filename],
                style={
                    "textAlign": "left",
                    "padding": "10px",
                    "word-wrap": "break-word",
                },
            ),
        )
        # TODO: Why am I using content and not content_str? maybe cuz html img default something?
        test_image = html.Div(
            [
                html.Img(
                    src=content,
                    style={
                        "width": "100%",
                        "height": "100%",
                        "min-height": "10vh",
                        "max-height": "15vh",
                        "object-fit": "contain",
                    },
                )
            ],
        )
        # content is an encoded picture
        # decode picture and save as file
        dataset_name = redis_client.get(f"{session_id}:curr_dataset")
        test_image_folder = f"assets/{session_id}/{dataset_name}/image_search/input/"
        if not os.path.exists(test_image_folder):
            os.makedirs(test_image_folder)
        del_folder_contents(test_image_folder)
        test_image_path = os.path.join(test_image_folder, filename)
        with open(test_image_path, "wb") as f:
            img_bytes = base64.b64decode(content.split("base64,")[-1])
            f.write(img_bytes)
        # print("test_image_path is:", test_image_path)
        # search for 6 similar images
        result_idxs = find_similar_imgs(session_id, dataset_name, test_image_path)
        # display them
        result_preview = gen_img_preview(
            redis_client, session_id, dataset_name, result_idxs[0], scale=0.99
        )
        prepare_preview_download(
            redis_client,
            session_id,
            dataset_name,
            result_idxs[0],
            "image_search/search_results",
        )
        dataset_dir = f"assets/{session_id}/{dataset_name}/image_search"
        preview_zip_path = os.path.join(dataset_dir, "search_results.zip")
        return output_filename, test_image, result_preview, preview_zip_path
    return no_update, no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Create 3D graph and update cluster download """
########################################################################
@app.callback(
    [
        Output("mainGraph", "figure"),
        Output("dataClusteredFlag", "data"),
        Output("dataInfo", "children"),
        Output("download_clusters_button", "disabled"),
        Output("download_clusters_button", "href"),
    ],
    [Input("graph3DButton", "n_clicks")],
    [
        State("dataProcessedFlag", "data"),
        State("n_neighbors_slider", "value"),
        State("min_dist_slider", "value"),
        State("min_cluster_size_slider", "value"),
        State("min_samples_slider", "value"),
        State("session-id", "data"),
    ],
)
def update_output(
    n_clicks,
    dataProcessedFlag,
    n_neighbors,
    min_dist,
    min_cluster_size,
    min_samples,
    session_id,
):
    # After feature extraction, enable 3D graph gen button
    if dataProcessedFlag:
        global redis_client
        # Generate the 3D graph and update global variable
        dataset_name = redis_client.get(f"{session_id}:curr_dataset")
        figure, percent_clustered = generate_fig_3D(
            redis_client,
            session_id,
            dataset_name,
            n_neighbors,
            min_dist,
            min_cluster_size,
            min_samples,
        )
        # Output Clustering statistics
        output_text = create_LR_label(
            id="percentClusteredText",
            leftText="[INFO]:",
            rightText=f"{percent_clustered}% clustered",
        )
        # arrange zip files to create download
        create_clusters_zip(
            redis_client,
            session_id,
            dataset_name,
            n_neighbors,
            min_dist,
            min_cluster_size,
            min_samples,
        )
        dataset_dir = f"assets/{session_id}/{dataset_name}"
        cluster_zip_path = os.path.join(dataset_dir, "clusters.zip")

        return figure, [True], output_text, False, cluster_zip_path
    else:
        return no_update, no_update, no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Create 2D Graph """
########################################################################
@app.callback(
    [Output("graph2D", "figure")],
    [Input("dataClusteredFlag", "data")],
    [
        State("n_neighbors_slider", "value"),
        State("min_dist_slider", "value"),
        State("min_cluster_size_slider", "value"),
        State("min_samples_slider", "value"),
        State("session-id", "data"),
    ],
)
def create_graph_2D(
    dataClusteredFlag,
    n_neighbors,
    min_dist,
    min_cluster_size,
    min_samples,
    session_id,
):
    if dataClusteredFlag:
        global redis_client
        # Calculate UMAP embeddings with two components.
        dataset_name = redis_client.get(f"{session_id}:curr_dataset")
        embeddings_2D = setup_umap(
            redis_client,
            session_id,
            dataset_name,
            n_neighbors,
            min_dist,
            n_components=2,
        )
        labels = get_labels(
            session_id,
            dataset_name,
            n_neighbors,
            min_dist,
            min_cluster_size,
            min_samples,
        )
        img_paths = orjson.loads(
            redis_client.get(f"{session_id}:{dataset_name}:img_paths")
        )
        fig = generate_fig_2D(embeddings_2D, labels, img_paths)

        return [fig]
    else:
        return no_update


########################################################################
""" [CALLBACK]: 3D hover to show image """
########################################################################
@app.callback(
    [
        Output("mainGraphTooltip", "show"),
        Output("mainGraphTooltip", "bbox"),
        Output("mainGraphTooltip", "children"),
        # Output("mainGraphTooltip", "border_color"),
    ],
    [Input("mainGraph", "hoverData")],
    [
        State("dataClusteredFlag", "data"),
        State("session-id", "data"),
    ],
)
def display_hover(hoverData, dataClusteredFlag, session_id):
    if (hoverData is None) or (not dataClusteredFlag):
        return False, no_update, no_update
    global redis_client
    dataset_name = redis_client.get(f"{session_id}:curr_dataset")
    hover_data = hoverData["points"][0]
    # Load image with pillow
    hover_img_index = hover_data["pointNumber"]
    # demo only shows the first point, but other points may also be available
    img_path = hover_data["hovertext"]
    cluster = hover_data["marker.color"]
    im_uri = gen_img_uri(
        redis_client, session_id, dataset_name, hover_img_index, img_path
    )
    bbox = hover_data["bbox"]
    img_name = img_path.split("/")[-1]
    children = [
        html.Img(
            src=im_uri,
            style={"width": "150px"},
        ),
        # dbc.FormText(f"Name: {img_name}", style={"font-size": "x-small"}),
        html.P(
            f"cluster: {cluster}   ,   index: {hover_img_index}",
            style={"font-size": "small", "color": "black", "textAlign": "center"},
        ),
    ]
    return True, bbox, children


########################################################################
""" [CALLBACK]: 2D Hover preview"""
########################################################################
@app.callback(
    [
        Output("graph2DTooltip", "show"),
        Output("graph2DTooltip", "bbox"),
        Output("graph2DTooltip", "children"),
    ],
    [Input("graph2D", "hoverData")],
    State("session-id", "data"),
)
def display_hover2D(hoverData, session_id):
    if hoverData is None:
        return False, no_update, no_update
    # Load image with pillow
    global redis_client
    dataset_name = redis_client.get(f"{session_id}:curr_dataset")
    hover_data = hoverData["points"][0]
    hover_img_index = hover_data["pointNumber"]
    img_path = hover_data["customdata"]
    cluster = hover_data["marker.color"]
    try:
        im_uri = gen_img_uri(
            redis_client, session_id, dataset_name, hover_img_index, img_path
        )
    except:
        return False, no_update, no_update
    # demo only shows the first point, but other points may also be available

    bbox = hover_data["bbox"]
    children = [
        html.Img(
            src=im_uri,
            style={"width": "150px"},
        ),
        html.P(
            f"cluster: {cluster}   ,   index: {hover_img_index}",
            style={"font-size": "small", "color": "black", "textAlign": "center"},
        ),
    ]
    return True, bbox, children


########################################################################
""" [CALLBACK]: 3D Click Download"""
########################################################################
@app.callback(
    Output("mainGraphDownload", "data"),
    Input("mainGraph", "clickData"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def func(clickData, session_id):
    global redis_client
    dataset_name = redis_client.get(f"{session_id}:curr_dataset")
    img_paths = orjson.loads(redis_client.get(f"{session_id}:{dataset_name}:img_paths"))
    clicked_img_idx = clickData["points"][0]["pointNumber"]
    image_path = img_paths[clicked_img_idx]
    return dcc.send_file(image_path)


########################################################################
""" [CALLBACK]: 2D Click Download"""
########################################################################
@app.callback(
    Output("graph2DDownload", "data"),
    Input("graph2D", "clickData"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def func(clickData, session_id):
    global redis_client
    dataset_name = redis_client.get(f"{session_id}:curr_dataset")
    img_paths = orjson.loads(redis_client.get(f"{session_id}:{dataset_name}:img_paths"))
    clicked_img_idx = clickData["points"][0]["pointNumber"]
    image_path = img_paths[clicked_img_idx]
    return dcc.send_file(image_path)


########################################################################
""" [CALLBACK]: 2D Lasso to preview"""
########################################################################
@app.callback(
    [
        Output("imagePreview", "children"),
        Output("download_preview_button", "disabled"),
        Output("download_preview_button", "href"),
    ],
    [Input("graph2D", "selectedData")],
    [State("session-id", "data"), State("card-tabs", "active_tab")],
    prevent_initial_call=True,
)
def display_selected_data(selectedData, session_id, active_tab):
    """get all the selected point idx from the selectedData dictionary
    then get the img path for each idx, then create previwe with paths"""
    if selectedData["points"] and active_tab == "tab-1":
        global redis_client

        points = selectedData["points"]

        selected_img_idxs = []
        for i in range(len(points)):
            idx = points[i]["pointNumber"]
            selected_img_idxs.append(idx)

        dataset_name = redis_client.get(f"{session_id}:curr_dataset")
        prepare_preview_download(
            redis_client, session_id, dataset_name, selected_img_idxs, "preview_2D"
        )
        dataset_dir = f"assets/{session_id}/{dataset_name}"
        preview_zip_path = os.path.join(dataset_dir, "preview_2D.zip")
        return (
            gen_img_preview(redis_client, session_id, dataset_name, selected_img_idxs),
            False,
            preview_zip_path,
        )
    else:
        # no points selected
        return no_update, no_update, no_update


@app.callback(
    Output("dummy1", "children"),
    Input("interval-component", "n_intervals"),
    State("session-id", "data"),
)
def update_activity_timestamp(n, session_id):
    # store and update redis timestamps for each user
    global redis_client
    curr_time = str(datetime.now())
    redis_client.set(f"{session_id}:latest_timestamp", curr_time)
    return no_update


@app.callback(
    Output("card-content", "children"),
    [Input("card-tabs", "active_tab")],
    State("session-id", "data"),
)
def tab_content(active_tab, session_id):
    if active_tab == "tab-1":
        content = [
            dbc.Row(
                [preview_title],
            ),
            horz_line,
            dbc.Row(
                dbc.Col(imagePreview),
                justify="center",
                align="center",
            ),
        ]

    else:
        content = [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row([upload_image_file_button]),
                            horz_line,
                            dbc.Row([download_search_button]),
                            horz_line,
                            dbc.Row([image_file_info]),
                        ],
                        style={"max-height": "30vh"},
                        md=2,
                    ),
                    dbc.Col(
                        [
                            preview_test_image,
                        ],
                        style={"max-height": "30vh"},
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Row([search_preview]),
                        ],
                        style={"max-height": "30vh"},
                        md=8,
                    ),
                ],
                justify="center",
                align="center",
            ),
        ]

    return content


if __name__ == "__main__":
    # start thread for user activity worker here
    x = threading.Thread(
        target=poll_remove_user_data, args=(redis_client,), daemon=True
    )
    x.start()
    server.run(debug=True, host="0.0.0.0", port=5000)
