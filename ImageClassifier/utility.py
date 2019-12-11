import json
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms, models
from PIL import Image

def prepare_data(datadir):
    
    train_dir = datadir + '/train'
    test_dir = datadir + '/test'
    valid_dir = datadir + '/valid'
    
    data_transforms = {
        'train' : transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'test' : transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid' : transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True)
    }
    
    return image_datasets, dataloaders


def label_mapping(cat_path):
    cat_to_name = {}
    with open(cat_path, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name
    

def get_inputs():
    
    parser = argparse.ArgumentParser(description="Train your Neural Network")
    parser.add_argument('datadir', help="Directory of data")
    parser.add_argument('--save_dir', default="ImageClassifier/checkpoints", help="Directory to save checkpoints")
    parser.add_argument('--arch',  default="vgg11", help="name of neural model to use in training the network")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="The rate at which the network learns")
    parser.add_argument('--hidden_units', nargs='*', default=[1000, 1000], type=int, help="The units of the hidden layers")
    parser.add_argument('--epochs', default=5, type=int, help="iterations we need to make to fully train our network")
    parser.add_argument('--gpu', nargs='?', const='cuda', default='cpu', help="use gpu or cpu")
    
    args = parser.parse_args()
    
    return args


def get_predict_inputs():
    
    parser = argparse.ArgumentParser(description="Train your Neural Network")
    parser.add_argument('imagepath', help="path to image")
    parser.add_argument('checkpoint', default="ImageClassifier/checkpoints/checkpoint.pth", help="Directory to save checkpoints")
    parser.add_argument('--top_k',  default=1, type=int, help="name of neural model to use in training the network")
    parser.add_argument('--category_names', help="The rate at which the network learns")
    parser.add_argument('--gpu', nargs='?', const='cuda', default='cpu', help="use gpu or cpu")
    
    args = parser.parse_args()
    
    return args


def save_checkpoint(model, folder):
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': model.optimizer_state_dict,
        'name':model.name
                 
    }
    
    torch.save(checkpoint, folder + '/checkpoint.pth')
    

def load_checkpoint(path):
    
    checkpoint = torch.load(path)
    model = getattr(models, checkpoint['name'])
    model = model(pretrained=False)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer_state_dict = checkpoint['optimizer_state_dict']
    
    return model
    
    
def process_image(imagepath):
    image = Image.open(imagepath)
    # Resize the images where shortest side is 256 pixels, keeping aspect ratio. 
    if image.width > image.height: 
        factor = image.width/image.height
        image = image.resize(size=(int(round(factor*256,0)),256))
    else:
        factor = image.height/image.width
        image = image.resize(size=(256, int(round(factor*256,0))))
    # Crop out the center 224x224 portion of the image.

    image = image.crop(box=((image.width/2)-112, (image.height/2)-112, (image.width/2)+112, (image.height/2)+112))

    # Convert to numpy array
    np_image = np.array(image)
    np_image = np_image/255
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std 
    # Reorder dimension for PyTorch
    np_image = np.transpose(np_image, (2, 1, 0))

    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    
    return tensor_image