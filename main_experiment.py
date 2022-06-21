import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pvt import PyramidVisionTransformer
from pvt_decode_v4 import Res_ViT_decode
import albumentations as A
from albumentations.pytorch import ToTensorV2
from miou import ConfusionMatrix
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

n_epochs = 5000
learning_rate = 3e-06
batch_size = 10
image_height = 512  # 224
image_width = 512  # 224
SMOOTH = 1e-6
classes_num = 21

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


class VOCDataset(Dataset):
    """
    This class is for the creation of VOC2012 dataset
    """

    def __init__(self, root_path='../dataset_adl/VOCdevkit/VOC2012/',
                 type='train', image_size=(224, 224), resize_method='resize', transform=None):
        self.root_path = root_path
        self.resize_method = resize_method
        self.type = type
        self.transform = transform

        # define directory where train and val pictures names are stored
        self.names_path = self.root_path + 'ImageSets/Segmentation/' + self.type + '.txt'

        # define image and labels path
        self.image_path = self.root_path + 'JPEGImages/'
        self.label_path = self.root_path + 'SegmentationClass/'
        self.image_size = image_size

        # define array of picture names, images and labels
        self.names = []

        # parms for converting label pixels to class pixels
        self.colormap2label = self.build_colormap2label()
        self.mode = torchvision.io.image.ImageReadMode.RGB

        # reading file names
        self.read_names()

    def __getitem__(self, index):

        image = cv2.imread(self.image_path + self.names[index] + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_path + self.names[index] + '.png')
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=label)
            image = augmentations["image"]
            label = augmentations["mask"]

        label = self.voc_label_indices(label, self.colormap2label)
        return image.to(device), label.to(device)

    def __len__(self):
        return self.names.shape[0]

    def read_names(self):
        """
        Read the filenames of training images and labels into self.names
        """
        f = open(self.names_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.names.append(line)
        self.names = np.array(self.names)
        f.close()

    def build_colormap2label(self):
        """
        Build an RGB color to label mapping for segmentation.
        """
        colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
        for i, colormap in enumerate(VOC_COLORMAP):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                           colormap[2]] = i
        return colormap2label

    def voc_label_indices(self, colormap, colormap2label):
        """
        Map an RGB color to a label.
        """
        # colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
        colormap = colormap.numpy().astype('int32')
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 +
               colormap[:, :, 2])
        return colormap2label[idx]


def decode_segmap(image, nc=classes_num):
    label_colors = np.array(VOC_COLORMAP)

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


Encoder = PyramidVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 8],
                                   mlp_ratios=[8, 8, 4, 4],
                                   qkv_bias=True, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                                   sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
save_model = torch.load("pvt_small.pth")
model_dict = Encoder.state_dict()
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
model_dict.update(state_dict)
Encoder.load_state_dict(model_dict)

Decoder = Res_ViT_decode(image_height, image_height, batch_size, classes_num)


class FCN_VIT(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.resnet34 = models.resnet34(pretrained=False)
        self.resnet34.load_state_dict(torch.load("resnet34-b627a593.pth"))
        self.conv1 = self.resnet34.conv1
        self.bn1 = self.resnet34.bn1
        self.relu = self.resnet34.relu
        self.maxpool = self.resnet34.maxpool
        self.layer1 = self.resnet34.layer1
        self.layer2 = self.resnet34.layer2
        self.layer3 = self.resnet34.layer3
        self.layer4 = self.resnet34.layer4
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        self.decoder = decoder.to(device)

    def aggregate(self, x, ins, outs):
        self.convs = nn.Conv2d(ins, outs, 1)
        self.convs = self.convs.to(self.device)
        return self.convs(x)

    def forward(self, inputs):
        xx = self.encoder(inputs)
        yy = self.relu(self.maxpool(self.bn1(self.conv1(inputs))))
        outs = []
        for i in range(4):
            xc = xx[i].shape[1]
            yy = self.layers[i](yy)
            yc = yy.shape[1]
            zz = torch.cat((xx[i], yy), 1)
            zz = self.aggregate(zz, xc+yc, yc)
            outs.append(zz)

        return self.decoder(outs)


train_transform = A.Compose(
    [
        A.Resize(height=image_height, width=image_width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)
val_transform = A.Compose(
    [
        A.Resize(height=image_height, width=image_width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ],
)

preprocess_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# get datasets
train_dataset = VOCDataset(image_size=(image_height, image_width),
                           type='train_new', transform=train_transform)

val_dataset = VOCDataset(image_size=(image_height, image_width),
                         type='testval', transform=val_transform)

train_path = "miou_train_v10.txt"
test_path = "miou_test_v10.txt"


def test(val_dataset, model, epoch, save_path):
    model.eval()
    mean_iou = []
    draw(model, "2010_005871.jpg", "test_v11.jpg")
    draw(model, "2010_005860.jpg", "test_v11_1.jpg")
    threshold = 5
    ious = [0] * 21
    times = [0] * 21
    with torch.no_grad():
        for i, (img, label) in enumerate(val_dataset):
            pred = model(img.unsqueeze(0).float())
            pred = F.softmax(pred, dim=1)
            values, prediction = torch.max(pred, dim=1)
            cm = ConfusionMatrix(prediction.detach().cpu().squeeze(0), label.detach().cpu(), image_height, image_width,
                             classes_num)
            cm.construct()
            iou = cm.computeIou()
            pos = np.where(cm.actual_count > 0)
            pos = pos[0]
            for j in range(pos.shape[0]):
                ious[pos[j]] += iou[pos[j]]
                times[pos[j]] += 1

            if i < threshold:
                print(cm.computeIou())
        for i in range(21):
            ious[i] = float(ious[i] / times[i])
        print(ious)
        print("mIoU of test set is " + "(" + str(epoch) + ")" + " :", np.mean(np.array(ious)))
    with open(save_path, "a") as f:
        f.write(str(epoch) + " :" + str(np.mean(np.array(ious))) + '\n')


def draw(model, inputs, outputs):
    #inputs = "2010_005871.jpg"
    image_PIL = Image.open(inputs)
    image_tensor = preprocess_transform(image_PIL)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)
    pred = model(image_tensor)
    pred = F.softmax(pred, dim=1)
    values, prediction = torch.max(pred, dim=1)
    prediction = prediction.detach().cpu().squeeze(0)
    pics = decode_segmap(prediction, 21)
    segs = Image.fromarray(pics, mode='RGB')
    segs.save(outputs)

def train(n_epochs, data_loader, model, optimizer, loss_fn):
    """
    this function is for training model. one epoch is one call of this function
    """
    print('training start')
    model.train()
    for epoch in range(n_epochs):
        for i, (image, target) in enumerate(data_loader):
            image = image.to(device)
            target = target.to(device)
            predictions = model(image)
            loss = loss_fn(predictions, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
        if epoch % 25 == 0:
            test(train_dataset, model, epoch, train_path)
            test(val_dataset, model, epoch, test_path)
            # torch.save(model, "./models/v10/latest.pth")


print("length of train set:", len(train_dataset))
# create DataLoaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

print("length of test set:", len(val_dataset))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# define loss, model and optimizer
loss_fn = nn.CrossEntropyLoss()
#model = FCN_VIT(Encoder, Decoder, device).to(device)
model = torch.load("./models/v10/latest_aug.pth")
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
train(n_epochs, train_loader, model, optimizer, loss_fn)
