import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load images and preprocess
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

# Display image
def imshow(tensor, title=None):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

# Define content and style layers
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Model: pre-trained VGG19 with extracted features
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features.eval()
        self.layer_mapping = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_mapping:
                features[self.layer_mapping[name]] = x
        return features

def gram_matrix(tensor):
    _, n_filters, h, w = tensor.size()
    tensor = tensor.view(n_filters, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Load your content and style images
content = load_image('content.jpg').to('cuda')
style = load_image('style.jpg', shape=content.shape[-2:]).to('cuda')

# Model
vgg = VGGFeatures().to('cuda')

# Extract features
content_features = vgg(content)
style_features = vgg(style)

# Initialize target image and optimizer
target = content.clone().requires_grad_(True).to('cuda')
optimizer = optim.Adam([target], lr=0.003)

style_weights = {'conv1_1':1.0, 'conv2_1':0.8, 'conv3_1':0.5, 'conv4_1':0.3, 'conv5_1':0.1}
content_weight = 1e4
style_weight = 1e2

# Optimization
