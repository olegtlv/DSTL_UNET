import matplotlib.pyplot as plt
import cv2
import matplotlib
from train import calc_validation_loss_one_epoch, train_one_epoch
from tools import viz_one, get_one_RGBImage_n_label, get_scalers, mask_for_polygons, viz_one_plotly
from model_u_orig import UNet_orig
from model_effnet_based import UNet_effnet
from dataset import RGBChunkedDataset
from training_tools import CombinedLoss
matplotlib.use('Qt5Agg')
import numpy as np
import pandas as pd
import IPython.display as display
import plotly
import wandb

import random
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
from torchview import draw_graph

df = pd.read_csv(R"C:\data\dstl-satellite-imagery-feature-detection\train_wkt_v4.csv.zip")
sizes = pd.read_csv(r"C:\data\dstl-satellite-imagery-feature-detection\grid_sizes.csv.zip")
sizes = sizes.set_index('Unnamed: 0')
viz_one_plotly(df)

for c in df.ClassType.unique():
    print('class:', c)
    print("num of polygons:", df[(df.ClassType==c) & (df.MultipolygonWKT!='MULTIPOLYGON EMPTY')].MultipolygonWKT.apply(lambda x: x.count(',')).sum())

################################
### image and mask extraction
################################
ids = df.ImageId.unique()
masks = []
images = []
pickClass = 5 # vegetation
# pickClass = 6 # crops
# pickClass = 2 # structures
# pickClass = 1 # buildins
# pickClass = 4  # track
zip_file_path = r"C:\data\dstl-satellite-imagery-feature-detection\three_band.zip"

images = []
labels = []
geos = []

for ii, ID in enumerate(ids):
    print('image: ', ID)
    if ii < 1:
        continue
    image, label, geo = get_one_RGBImage_n_label(ID, df, zip_file_path, pickClass, sizes)

    images.append(image)
    labels.append(label)
    geos.append(geo)

# ####################
# DATASET
# ####################
sub_image_size = (224, 224)
overlap_ratio = 0.5
N_test = 50
device = 'cuda'
batch_size = 75

random.seed(2)
images_test = random.sample(list(df.ImageId.unique()),2)
images_train = np.setdiff1d(df.ImageId.unique(), images_test)

train_sub_image_options = []
test_sub_image_options = []

min_h, min_w = np.array([[i.shape[0], i.shape[1]] for i in images]).min(axis=0) \
               // (np.array(sub_image_size) * overlap_ratio) * (np.array(sub_image_size) * overlap_ratio)

min_h = int(min_h)
min_w = int(min_w)

step_size_h = int(sub_image_size[0] * (1 - overlap_ratio))
step_size_w = int(sub_image_size[1] * (1 - overlap_ratio))

top_coordinates = list(range(0, min_h - sub_image_size[0] + 1, step_size_h))
left_coordinates = list(range(0, min_w - sub_image_size[1] + 1, step_size_w))

for im in range(len(images_train)):
    all_cors = list(itertools.product(top_coordinates, left_coordinates))
    test_cors = random.sample(list(itertools.product(top_coordinates, left_coordinates)), N_test)
    exclude = []
    for i in test_cors:
        exclude.append((int(min(i[0] + overlap_ratio * sub_image_size[0], min_h - sub_image_size[0])),
                        int(min(i[1] + overlap_ratio * sub_image_size[1], min_w - sub_image_size[0]))))
        exclude.append((int(max(i[0] - overlap_ratio * sub_image_size[0], 0)),
                        int(max(i[1] - overlap_ratio * sub_image_size[1], 0))))
    #         print((i[0],i[1]),'   ',#               exclude[-1],'   ',#               exclude[-2],'   ')
    exclude = test_cors + exclude

    for e in list(set(exclude)):
        all_cors.remove(e)

    exclude = [(im,) + ii for ii in exclude]
    include = [(im,) + ii for ii in all_cors]
    test_cors = [(im,) + ii for ii in test_cors]

    test_sub_image_options = test_sub_image_options + test_cors
    train_sub_image_options = train_sub_image_options + include

print()
print('train images#:     ', len(train_sub_image_options))
print('test images#:      ', len(test_sub_image_options))
print('test/train ratio %:', len(test_sub_image_options) / len(train_sub_image_options) * 100)

images_train = images[:]
labels_train = labels[:]

# Create an instance of dataset
dataset_train = RGBChunkedDataset(images_train, labels_train, train_sub_image_options,
                                  top_coordinates, left_coordinates,
                                  sub_image_size=sub_image_size)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_test = RGBChunkedDataset(images_train, labels_train, test_sub_image_options,
                                 top_coordinates, left_coordinates,
                                 sub_image_size=sub_image_size)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


# Instantiate the model
num_classes = 1  # Binary segmentation
pretrained = True
model_orig = UNet_effnet(num_classes)
# model_orig = UNet_orig(num_classes)
model_orig.to(device)

# model_graph = draw_graph(model_orig, input_size=(1,3,224,224), expand_nested=True, depth=3)
# model_graph.visual_graph

# define and init epoch params
lr_e = 0.0005
lr_d = 0.0005
model_parameters = [
    {'params': model_orig.encoder1.parameters(), 'lr': lr_e},
    {'params': model_orig.encoder2.parameters(), 'lr': lr_e},
    {'params': model_orig.encoder3.parameters(), 'lr': lr_e},
    {'params': model_orig.encoder4.parameters(), 'lr': lr_e},
    {'params': model_orig.encoder5.parameters(), 'lr': lr_e},
    {'params': model_orig.bottleneck.parameters(), 'lr': lr_d},
    {'params': model_orig.decoder1.parameters(), 'lr': lr_d},
    {'params': model_orig.decoder2.parameters(), 'lr': lr_d},
    {'params': model_orig.decoder3.parameters(), 'lr': lr_d},
    {'params': model_orig.decoder4.parameters(), 'lr': lr_d},
    {'params': model_orig.decoder5.parameters(), 'lr': lr_d},
    {'params': model_orig.final_conv.parameters(), 'lr': lr_d},
]
combined_loss = CombinedLoss()
optimizer, scheduler = combined_loss.get_optimizer_and_scheduler(model_parameters)
num_epochs = 10
# Initialize lists to store results
train_loss_hist = []
train_loss_hist_monitor = []
train_loss_hist_batch = []
val_loss_hist = []
optimizer_hist_batch = []

# define WnB:
wandb.init(
    project="DSTL-veg",  # e.g., "segmentation-unet"
    name=f"run_{model_orig.__class__.__name__}",  # run name
    config={
        "epochs": num_epochs,
        "batch_size": dataloader_train.batch_size,
        "lr": optimizer.param_groups[0]['lr'],
        "optimizer": "Adam",
        "loss": "BCE + Dice",
        "gamma": combined_loss.gamma,
        "weight_dice": combined_loss.weight_dice
    }
)

# Iterate through epochs for training
for epoch in range(num_epochs):
    # Train one epoch
    epoch_loss = train_one_epoch(model_orig, dataloader_train, optimizer, combined_loss, epoch, train_loss_hist_batch,
                                 optimizer_hist_batch)
    print()
    print("EPOCH #", epoch)
    print("        ", " TRAIN LOSS BCE : %.4f" % float(epoch_loss))

    # Calculate validation loss for the epoch
    epoch_loss_test = calc_validation_loss_one_epoch(model_orig, dataloader_test, combined_loss, epoch)
    model_orig.train()
    print("          TEST LOSS BCE : %.4f" % float(epoch_loss_test))

    # Other updates and storage as needed
    scheduler.step()
    train_loss_hist.append(epoch_loss)
    val_loss_hist.append(epoch_loss_test)

    print('LR ENC: ', optimizer.param_groups[0]['lr'],
          'LR_DEC: ', optimizer.param_groups[6]['lr'])

wandb.finish()
