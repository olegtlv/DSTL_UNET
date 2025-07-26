import itertools
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
device = 'cuda'


data_transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.float()),
])

data_transform_label = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float() * 255),

])



# Define your custom dataset class
class RGBChunkedDataset(Dataset):
    def __init__(self, images, labels, sub_image_options,
                 top_coordinates,
                 left_coordinates,
                 data_transform_image=data_transform_image,
                 data_transform_label=data_transform_label,
                 sub_image_size = (224, 224)
                 ):
        self.images = images
        self.labels = labels
        self.sub_image_size = sub_image_size
        self.transform_image = data_transform_image
        self.transform_label = data_transform_label
        self.top_coordinates = top_coordinates
        self.left_coordinates = left_coordinates
        self.sub_image_options = sub_image_options

    def __len__(self):
        return len(self.sub_image_options)

    def __getitem__(self, idx):
        # Get coordinates for sub-image extraction
        image_id = self.sub_image_options[idx][0]
        top_i = self.sub_image_options[idx][1]
        left_i = self.sub_image_options[idx][2]
        rgb_array = self.images[image_id]
        mask = self.labels[image_id]

        # Crop sub-images from the original images
        rgb_sub_image = rgb_array[top_i:top_i + self.sub_image_size[0], left_i:left_i + self.sub_image_size[1], :]
        mask_sub_image = mask[top_i:top_i + self.sub_image_size[0], left_i:left_i + self.sub_image_size[1]]

        # Apply transformations
        if self.transform_image:
            rgb_sub_image = self.transform_image(rgb_sub_image)
        if self.transform_label:
            mask_sub_image = self.transform_label(mask_sub_image)

        return {'image': rgb_sub_image, 'mask': mask_sub_image, 'sub_coordinates': (top_i, left_i)}
