from dash import html
from dataset import Dataset

# Only certain component building functions are here


def gen_img_preview(dataset_obj: Dataset, selected_img_idxs):
    """Generate the selection preview when the lasso/box tool on 2D graph is used.
    inputs:
        - img_paths: (type: string) Component identifier for callback use
    output:
        - imagesList list of html div components where each contains one html.Img"""
    # Helper function to create html img
    def generate_thumbnail(image_uri):
        return html.Div(
            [
                html.Img(
                    src=image_uri,
                    style={
                        "height": "10%",
                        "width": "10%",
                        "float": "left",
                        "position": "relative",
                        "padding-top": 1,
                        "padding-right": 1,
                    },
                ),
            ]
        )

    # Functionality
    imagesList = []
    for idx in selected_img_idxs:
        image_uri = dataset_obj.gen_img_uri(idx)
        imagesList.append(generate_thumbnail(image_uri))

    return imagesList


def create_LR_label(id, leftText, rightText):
    """Insert an html.Div with text on one line with left and right
    justified portions.
    inputs:
        - id: (type: string) Component identifier for callback use
        - leftText: (type: string) Left justified text
        - rightText: (type: string) Right justified text
    output:
        - html.Div component object"""
    label = html.Div(
        id=id,
        children=[
            html.P(
                children=[leftText],
                style={"textAlign": "left", "width": "49%", "display": "inline-block"},
            ),
            html.P(
                children=[rightText],
                style={
                    "textAlign": "center",
                    "width": "50%",
                    "display": "inline-block",
                },
            ),
        ],
    )
    return label
