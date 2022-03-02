import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State

from dataset import Dataset
from componentBuilder import create_LR_label, gen_img_preview
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

# title of the dash
title = html.H4(
    "Dataset Clustering Dashboard",
    style={"color": "white", "text-align": "center", "padding": "10px"},
)

fileInfo = dcc.Loading(
    id="fileInfo",
    children=[
        html.H6(
            children=["Please upload FolderOfImagesYouWishToCluster.zip"],
            style={"textAlign": "center", "padding": "10px"},
        )
    ],
)

dataInfo = dcc.Loading(
    id="dataInfo",
    children=[
        html.Div(
            children=["Clustering begins upon feature extraction."],
            style={"textAlign": "Center", "padding": "10px"},
        )
    ],
)

graph2DInfo = html.Div(
    id="graph2DInfoDiv",
    children=[
        dcc.Loading(
            id="graph2DInfo",
            children=[
                html.H6(
                    id="graph2DInfoChild",
                    children=["A 2D graph allows for lasso preview."],
                )
            ],
        )
    ],
    style={"textAlign": "Center", "padding": "10px"},
)

slider = [
    create_LR_label(
        id="n_neighbors_label", leftText="[UMAP]:", rightText="n_neighbors"
    ),
    dcc.Slider(min=40, max=240, step=40, value=80, id="n_neighbors_slider"),
]

graphWithLoadingAnimation = dcc.Graph(
    id="mainGraph",
    clear_on_unhover=True,
    figure=blankFig(),
    loading_state={"is_loading": True},
)

graph2D = dcc.Graph(
    id="graph2D",
    clear_on_unhover=True,
    figure=blankFig(),
    loading_state={"is_loading": True},
)

downloadClustersButton = dbc.Button(
    children=["Upload Zip of Images to Download Clusters"],
    id="downloadClustersButton",
    download="example.jpg",
    n_clicks=0,
    disabled=True,
    external_link=True,
    color="primary",
)

graph2DButton = dbc.Button(
    children=["Generate 2D Graph"],
    id="graph2DButton",
    n_clicks=0,
    disabled=True,
    color="primary",
)

graph3DButton = dbc.Button(
    children=["Generate 3D Graph"],
    id="graph3DButton",
    n_clicks=0,
    disabled=True,
    color="primary",
)

uploadButton = dcc.Upload(
    id="upload-image-folder",
    children=["Drag or ", html.A("Select Zip File")],
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

# Tooltips for umap sliders
n_neighbors_tooltip = dbc.Tooltip(
    "This parameter controls how UMAP balances local versus global structure in the data. "
    "As n_neighbors is increased UMAP manages to see more of the overall "
    "structure of the data, gluing more components together, and better coverying "
    "the broader structure of the data. As n_neighbors increases further more and "
    "more focus in placed on the overall structure of the data. This results in, "
    "a plot where the overall structure is well captured, but at the loss of some of "
    "the finer local structure (individual images may no longer necessarily be "
    "immediately next to their closest match).",
    target="n_neighbors_label",
)

horzLine = html.Hr()

# In browser storage objects
dataProcessedFlagStore = dcc.Store(id="dataProcessedFlag", data=False)
dataClusteredFlagStore = dcc.Store(id="dataClusteredFlag", data=False)


################################################################################
""" Dash UI Layout: """
################################################################################

app.layout = dbc.Container(
    [
        title,
        dataProcessedFlagStore,
        dataClusteredFlagStore,
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.Row(fileInfo),
                            dbc.Row(uploadButton, align="center"),
                            # horzLine,
                            dbc.Row(dataInfo),
                            # horzLine,
                            n_neighbors_tooltip,
                            dbc.Row(slider),
                            horzLine,
                            dbc.Row(downloadClustersButton),
                            dbc.Row(graph3DButton),
                        ],
                        body=True,
                    ),
                    md=2,
                ),
                dbc.Col(
                    children=[
                        graphWithLoadingAnimation,
                        dcc.Tooltip(id="mainGraphTooltip", direction="right"),
                        dcc.Download(id="mainGraphDownload"),
                    ],
                    md=10,
                ),
            ],
            align="center",
        ),
        horzLine,
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card([dbc.Row(graph2DInfo), dbc.Row(graph2DButton)], body=True),
                    md=2,
                ),
                dbc.Col(
                    children=[
                        graph2D,
                        dcc.Tooltip(id="graph2DTooltip", direction="right"),
                        dcc.Download(id="graph2DDownload"),
                    ],
                    md=10,
                ),
            ],
            align="center",
        ),
        horzLine,
        dbc.Row(dbc.Card([imagePreview], body=True), align="center"),
        horzLine,
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
""" [CALLBACK]: Create 3D graph, n_neighbors umap slider """
########################################################################
@app.callback(
    [
        Output("mainGraph", "figure"),
        Output("dataClusteredFlag", "data"),
        Output("dataInfo", "children"),
        Output("graph2DButton", "disabled"),
        Output("graph3DButton", "disabled"),
    ],
    [Input("dataProcessedFlag", "data"), Input("graph3DButton", "n_clicks")],
    [State("n_neighbors_slider", "value")],
)
def update_output(dataProcessedFlag, n_clicks, n_neighbors_value):
    if dataProcessedFlag:
        # After feature extraction, enable 3D graph gen button
        if n_clicks == 0:
            return no_update, no_update, no_update, no_update, False
        else:
            # Generate the 3D graph and update global variable
            figure = generate_fig_3D(dataset_obj, n_neighbors_value)
            # Output Clustering statistics
            percentClustered = dataset_obj.calculate_percent_clustered()
            output_text = create_LR_label(
                id="percentClusteredText",
                leftText="[INFO]:",
                rightText=f"{percentClustered}% clustered",
            )
            # arrange zip files to create download
            dataset_obj.create_clusters_zip()
            return figure, [True], output_text, False, no_update
    else:
        return no_update, no_update, no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Create 2D Projection """
########################################################################
@app.callback(
    [Output("graph2D", "figure"), Output("graph2DInfo", "children")],
    [Input("graph2DButton", "n_clicks")],
    State("n_neighbors_slider", "value"),
    prevent_initial_call=True,
)
def create_graph_2D(n_clicks, slider_value):
    # Calculate UMAP embeddings with two components.
    embeddings_2D = dataset_obj.setup_umap(slider_value, n_components=2)
    # Use localinputdata (global) to get precalculated 3D labels
    # Make 2D px scattergl graph and update callback
    fig = generate_fig_2D(embeddings_2D, dataset_obj.labels)

    return fig, "2D UMAP Projection Ready"


########################################################################
""" [CALLBACK]: Download Clusters """
########################################################################
@app.callback(
    [
        Output("downloadClustersButton", "href"),
        Output("downloadClustersButton", "children"),
        Output("downloadClustersButton", "disabled"),
    ],
    [Input("dataClusteredFlag", "data"), Input("downloadClustersButton", "n_clicks")],
)
def create_clusters(dataClusteredFlag, n_clicks):
    if dataClusteredFlag:
        filePath = "clusters.zip"
        # return path to running assets folder, update the download button text,
        # and enable the button to be clicked
        return app.get_asset_url(filePath), "Download Clusters", False
    else:
        return no_update, no_update, no_update


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
    Output("imagePreview", "children"),
    Input("graph2D", "selectedData"),
    prevent_initial_call=True,
)
def display_selected_data(selectedData):
    """get all the selected point idx from the selectedData dictionary
    then get the img path for each idx, then create previwe with paths"""
    ctx = dash.callback_context

    points = selectedData["points"]
    selected_img_idxs = []

    for i in range(len(points)):
        idx = points[i]["pointNumber"]
        selected_img_idxs.append(idx)

    return gen_img_preview(dataset_obj, selected_img_idxs)


################################################################################
""" __main__: """
################################################################################

if __name__ == "__main__":
    app.run_server(debug=False)
