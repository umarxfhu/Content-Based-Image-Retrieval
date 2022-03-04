import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State

from dataset import Dataset
from componentBuilder import (
    create_LR_label,
    gen_img_preview,
    gen_download_button,
    create_info_loading,
    create_title_with_button,
)
from figureGen import blankFig, generate_fig_3D, generate_fig_2D

################################################################################
""" Global variables: """
################################################################################

from config import config

dataset_obj = Dataset()


################################################################################
""" Initialize Dash App: """
################################################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])


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
uploadButton = dcc.Upload(
    id="upload-image-folder",
    children=["Drag/Select ZIP"],
    style={
        # 'width': '100%',
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px",
    },
    accept=".zip",
    # Don't allow multiple files to be uploaded
    multiple=False,
)
graphWithLoadingAnimation = dcc.Graph(
    id="mainGraph",
    clear_on_unhover=True,
    figure=blankFig(),
    loading_state={"is_loading": True},
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
    "the broader structure of the data. As n_neighbors increases further more and "
    "more focus in placed on the overall structure of the data. This results in, "
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
    dcc.Slider(min=40, max=240, step=40, value=80, id="n_neighbors_slider"),
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
    dcc.Slider(min=0.0, max=0.5, step=0.1, value=0.0, id="min_dist_slider"),
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
        min=20,
        max=400,
        step=20,
        value=240,
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
        max=100,
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
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#   2D Graph Components
# ------------------------------------------------------------------------------
graph2D = dcc.Graph(
    id="graph2D",
    clear_on_unhover=True,
    figure=blankFig(),
    loading_state={"is_loading": True},
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
preview_title = create_title_with_button(["Selection Preview"], download_preview_button)
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
)
download_search_button = gen_download_button(
    id="download_search_button",
    children=["Download"],
    href=app.get_asset_url("search.zip"),
)
image_search_title = html.Div(
    children=[
        html.Div(
            [upload_image_file_button],
            style={
                "textAlign": "left",
                "width": "47%",
                "display": "inline-block",
                "margin-left": "2%",
            },
        ),
        html.Div(
            [download_search_button],
            style={
                "textAlign": "right",
                "width": "50%",
                "display": "inline-block",
            },
        ),
    ],
)
image_file_info = create_info_loading(id="image_file_info", children=[""])
preview_test_image = html.Div(
    id="preview_test_image",
    children=[
        html.H6(
            children=["Test"],
            style={"textAlign": "center", "padding": "10px"},
        )
    ],
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

# In browser storage objects
dataProcessedFlagStore = dcc.Store(id="dataProcessedFlag", data=False)
dataClusteredFlagStore = dcc.Store(id="dataClusteredFlag", data=False)

################################################################################
""" Dash UI Layout: """
################################################################################

app.layout = dbc.Container(
    [
        dataProcessedFlagStore,
        dataClusteredFlagStore,
        dbc.Row(title),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.Row(
                                children=[
                                    dbc.Row(fileInfo),
                                    dbc.Row(dataInfo),
                                    dbc.Row(uploadButton),
                                    horz_line,
                                    dbc.Row(n_neighbors_slider),
                                    dbc.Row(min_dist_slider),
                                    horz_line,
                                    dbc.Row(min_cluster_size_slider),
                                    dbc.Row(min_samples_slider),
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
                        style={"height": "70vh"},
                    ),
                    md=3,
                    align="start",
                ),
                dbc.Col(
                    children=[
                        dbc.Row(
                            children=[
                                graphWithLoadingAnimation,
                                dcc.Tooltip(id="mainGraphTooltip", direction="right"),
                                dcc.Download(id="mainGraphDownload"),
                            ],
                            style={"height": "70vh"},
                        ),
                    ],
                    md=9,
                ),
            ],
            align="center",
        ),
        horz_line,
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.Row(
                                    [image_search_title],
                                ),
                                horz_line,
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [preview_test_image],
                                            style={"max-height": "20vh"},
                                            md=6,
                                        ),
                                        dbc.Col([image_file_info], md=6),
                                    ]
                                ),
                                horz_line,
                            ],
                            body=True,
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    children=[
                        graph2D,
                        dcc.Tooltip(id="graph2DTooltip", direction="right"),
                        dcc.Download(id="graph2DDownload"),
                        horz_line,
                        dbc.Row(
                            children=(
                                dbc.Col(
                                    children=[
                                        dbc.Card(
                                            [
                                                dbc.Row(
                                                    [preview_title],
                                                ),
                                                horz_line,
                                                imagePreview,
                                            ],
                                            body=True,
                                        ),
                                    ],
                                    width="auto",
                                    md=12,
                                ),
                            ),
                            justify="center",
                            align="center",
                        ),
                        horz_line,
                    ],
                    md=8,
                ),
            ],
            style={"height": "50vh"},
            justify="center",
            align="start",
        ),
    ],
    fluid=True,
)

################################################################################
""" Callback fxns: """
################################################################################
########################################################################
""" [CALLBACK]: upload zip file and extract features/paths """
########################################################################
@app.callback(
    [Output("dataProcessedFlag", "data"), Output("fileInfo", "children")],
    [Input("upload-image-folder", "contents")],
    State("upload-image-folder", "filename"),
)
def uploadData(content, filename):
    # the content needs to be split. It contains the type and the real content
    global dataset_obj
    if content is not None:
        content_type, content_str = content.split(",")
        dataset_obj = dataset_obj.reset_dataset()
        # Create Dataset object, remove the extension part (.zip) from the filename
        dataset_obj.name = filename.split(".")[0]
        # read uploaded data and create ZipFile obj
        dataset_obj.decode_and_extract_zip(content_str)
        dataset_obj.load_features_paths()
        outputText = create_LR_label(
            id="file_info_label",
            leftText="[FILE]:",
            rightText=dataset_obj.name,
        )
        return [True], outputText

    return no_update, no_update


########################################################################
""" [CALLBACK]: upload image file and find similar images """
########################################################################
@app.callback(
    [Output("image_file_info", "children"), Output("preview_test_image", "children")],
    [Input("upload_image_file_button", "contents")],
    [
        State("upload_image_file_button", "filename"),
    ],
)
def uploadData(content, filename):
    global dataset_obj
    if content is not None:
        content_type, content_str = content.split(",")
        output_filename = (
            html.H6(
                children=["[Filename]: ", html.Br(), filename],
                style={
                    "textAlign": "left",
                    "padding": "10px",
                    "word-wrap": "break-word",
                },
            ),
        )
        test_image = html.Div(
            [
                html.Img(
                    src=content,
                    style={
                        # "max-inline-size": "100%",
                        # "block-size": "auto",
                        "width": "100%",
                        "height": "100%",
                        "min-height": "10vh",
                        # "display": "block",
                        "object-fit": "contain",
                        "max-height": "20vh",
                    },
                )
            ],
        )
        return output_filename, test_image
    return no_update, no_update


########################################################################
""" [CALLBACK]: Create 3D graph and update cluster download """
########################################################################
@app.callback(
    [
        Output("mainGraph", "figure"),
        Output("dataClusteredFlag", "data"),
        Output("dataInfo", "children"),
        Output("graph3DButton", "disabled"),
        Output("download_clusters_button", "disabled"),
    ],
    [Input("dataProcessedFlag", "data"), Input("graph3DButton", "n_clicks")],
    [
        State("n_neighbors_slider", "value"),
        State("min_dist_slider", "value"),
        State("min_cluster_size_slider", "value"),
        State("min_samples_slider", "value"),
    ],
)
def update_output(
    dataProcessedFlag, n_clicks, n_neighbors, min_dist, min_cluster_size, min_samples
):
    if dataProcessedFlag:
        # After feature extraction, enable 3D graph gen button
        if n_clicks == 0:
            return no_update, no_update, no_update, False, no_update
        else:
            # Generate the 3D graph and update global variable
            figure = generate_fig_3D(
                dataset_obj, n_neighbors, min_dist, min_cluster_size, min_samples
            )
            # Output Clustering statistics
            output_text = create_LR_label(
                id="percentClusteredText",
                leftText="[INFO]:",
                rightText=f"{dataset_obj.calculate_percent_clustered()}% clustered",
            )
            # arrange zip files to create download
            dataset_obj.create_clusters_zip()
            return figure, [True], output_text, no_update, False
    else:
        return no_update, no_update, no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Create 2D Graph """
########################################################################
@app.callback(
    [Output("graph2D", "figure")],
    [Input("dataClusteredFlag", "data")],
    [State("n_neighbors_slider", "value"), State("min_dist_slider", "value")],
)
def create_graph_2D(dataClusteredFlag, n_neighbors_value, min_dist_value):
    if dataClusteredFlag:
        # Calculate UMAP embeddings with two components.
        embeddings_2D = dataset_obj.setup_umap(
            n_neighbors_value, min_dist_value, n_components=2
        )
        # Use localinputdata (global) to get precalculated 3D labels
        # Make 2D px scattergl graph and update callback
        fig = generate_fig_2D(embeddings_2D, dataset_obj.labels)

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
    ],
    [Input("mainGraph", "hoverData")],
    [State("dataClusteredFlag", "data")],
)
def display_hover(hoverData, dataClusteredFlag):
    if (hoverData is None) or (not dataClusteredFlag):
        return False, no_update, no_update
    # Load image with pillow
    hover_img_index = hoverData["points"][0]["pointNumber"]
    im_uri = dataset_obj.gen_img_uri(hover_img_index)
    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    children = [
        html.Img(
            src=im_uri,
            style={"width": "150px"},
        ),
        # html.P("Image from base64 string"),
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
)
def display_hover2D(hoverData):
    if hoverData is None:
        return False, no_update, no_update
    # Load image with pillow
    hover_img_index = hoverData["points"][0]["pointNumber"]
    im_uri = dataset_obj.gen_img_uri(hover_img_index)
    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    children = [
        html.Img(
            src=im_uri,
            style={"width": "150px"},
        ),
        # html.P("Image from base64 string"),
    ]
    return True, bbox, children


########################################################################
""" [CALLBACK]: 3D Click Download"""
########################################################################
@app.callback(
    Output("mainGraphDownload", "data"),
    Input("mainGraph", "clickData"),
    prevent_initial_call=True,
)
def func(clickData):
    clicked_img_idx = clickData["points"][0]["pointNumber"]
    image_path = dataset_obj.img_paths[clicked_img_idx]
    return dcc.send_file(image_path)


########################################################################
""" [CALLBACK]: 2D Click Download"""
########################################################################
@app.callback(
    Output("graph2DDownload", "data"),
    Input("graph2D", "clickData"),
    prevent_initial_call=True,
)
def func(clickData):
    print("entered 2d click fxn")
    clicked_img_idx = clickData["points"][0]["pointNumber"]
    image_path = dataset_obj.img_paths[clicked_img_idx]
    return dcc.send_file(image_path)


########################################################################
""" [CALLBACK]: 2D Lasso to preview"""
########################################################################
@app.callback(
    [
        Output("imagePreview", "children"),
        Output("download_preview_button", "disabled"),
    ],
    [Input("graph2D", "selectedData")],
    prevent_initial_call=True,
)
def display_selected_data(selectedData):
    """get all the selected point idx from the selectedData dictionary
    then get the img path for each idx, then create previwe with paths"""
    if selectedData["points"]:
        points = selectedData["points"]
        selected_img_idxs = []

        for i in range(len(points)):
            idx = points[i]["pointNumber"]
            selected_img_idxs.append(idx)

        dataset_obj.prepare_preview_download(selected_img_idxs)

        return gen_img_preview(dataset_obj, selected_img_idxs), False
    else:
        # no points selected
        return no_update, no_update


################################################################################
""" __main__: """
################################################################################

if __name__ == "__main__":
    app.run_server(debug=True)
