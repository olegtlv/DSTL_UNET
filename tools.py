import numpy as np
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
from matplotlib.patches import Polygon as MplPolygon
import cv2
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff
import pandas as pd
from PIL import Image
import zipfile
import plotly
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
import numpy as np
import plotly.io as pio
import matplotlib.pyplot as plt
import torch

def viz_one(df):
    for ids in ['6170_4_1']:
        image = df[df.ImageId == ids]

        fig, ax = plt.subplots(figsize=(6, 6))

        for _, row in image.iterrows():
            geom = wkt.loads(row['MultipolygonWKT'])

            if isinstance(geom, ShapelyPolygon):
                multipoly = [geom]  # single polygon
            elif isinstance(geom, MultiPolygon):
                multipoly = list(geom.geoms)  # unpack into list
            else:
                print(f"Unknown geometry type for image {ids}, class {row['ClassType']}")
                continue

            for poly in multipoly:
                coords = np.array(poly.exterior.coords)
                patch = MplPolygon(coords, color=plt.cm.Set1(row['ClassType'] % 4), alpha=0.3)
                ax.add_patch(patch)

        ax.set_title(ids)
        ax.relim()
        ax.autoscale_view()
        plt.axis("equal")
        plt.show()


def get_scalers(W,H,x_max,y_min):
    h, w = W,H  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min



pio.renderers.default = "browser"  # Open plot in a new browser window

def mpl_to_plotly_rgba(color, alpha=0.3):
    """Convert matplotlib RGBA to Plotly-compatible rgba string."""
    r, g, b = [int(255 * c) for c in color[:3]]
    return f'rgba({r},{g},{b},{alpha})'

def viz_one_plotly(df):
    for ids in ['6170_4_1']:
        image = df[df.ImageId == ids]
        fig = go.Figure()

        for _, row in image.iterrows():
            geom = wkt.loads(row['MultipolygonWKT'])

            if isinstance(geom, ShapelyPolygon):
                multipoly = [geom]
            elif isinstance(geom, MultiPolygon):
                multipoly = list(geom.geoms)
            else:
                print(f"Unknown geometry type for image {ids}, class {row['ClassType']}")
                continue

            color = plt.cm.Set1(row['ClassType'] % 9)  # Set1 supports up to 9 distinct colors
            plotly_color = mpl_to_plotly_rgba(color, alpha=0.3)

            for poly in multipoly:
                x, y = np.array(poly.exterior.xy[0]), np.array(poly.exterior.xy[1])
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    fill='toself',
                    fillcolor=plotly_color,
                    line=dict(color='black'),
                    mode='lines',
                    name=f"Class {row['ClassType']}"
                ))

        fig.update_layout(
            title=f"Image ID: {ids}",
            xaxis=dict(constrain='domain'),
            # yaxis=dict(scaleanchor="x", scaleratio=1),  # <- REMOVE THIS for auto aspect
            showlegend=True
        )

        fig.show()


def mask_for_polygons(polygons, W,H):
    img_mask = np.zeros((W,H), np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons.geoms]
    interiors = [int_coords(pi.coords) for poly in polygons.geoms
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask



def get_one_RGBImage_n_label(ID, df, zip_file_path, pickClass, sizes):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        with zip_file.open('three_band/' + ID + '.tif') as image_file:
            # Read the image using tifffile
            image = tiff.imread(image_file)

    # Ensure the image has three channels (RGB)
    if image.shape[0] == 3:
        # Transpose the image to (height, width, channels)
        image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.float32) / 2 ** 11  # Assuming the original image is in 16-bit format

    W, H, _ = image.shape

    polygon_str = df[(df.ImageId == ID) & (df.ClassType == pickClass)].MultipolygonWKT.values[0].strip()
    geometry = wkt.loads(polygon_str)

    x_scaler, y_scaler = get_scalers(W, H, sizes.loc[ID].Xmax, sizes.loc[ID].Ymin)

    geometry_scaled = shapely.affinity.scale(geometry, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

    train_mask = mask_for_polygons(geometry_scaled, W, H)
    return (image, train_mask, geometry_scaled)


import plotly.graph_objects as go
import numpy as np

def show_one_image_with_pred(inputs, outputs, targets, img_num=0, return_img=False):
    # Convert tensors to numpy arrays
    input_img = inputs[img_num].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    pred_mask = outputs[img_num].squeeze().cpu().detach().numpy()  # [H, W]
    target_mask = targets[img_num].squeeze().cpu().detach().numpy()  # [H, W]

    # De-normalize input image if in [-1, 1]
    input_img = (input_img + 1) / 2
    input_img = np.clip(input_img, 0, 1)

    fig = make_triple_subplot(input_img, pred_mask, target_mask)

    if return_img:
        # Convert Plotly figure to image (PIL) for W&B
        img_bytes = fig.to_image(format="png")
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        return img
    else:
        fig.show()


def make_triple_subplot(input_img, pred_mask, target_mask):
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Input", "Prediction", "Target"))

    fig.add_trace(go.Image(z=(input_img * 255).astype(np.uint8)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=pred_mask, colorscale='gray', showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=target_mask, colorscale='gray', showscale=False), row=1, col=3)

    fig.update_layout(height=400, width=1000, margin=dict(t=40))
    return fig
