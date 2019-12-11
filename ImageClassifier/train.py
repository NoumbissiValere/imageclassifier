from collections import OrderedDict
import numpy as np
import torch
import utility
from workspace_utils import active_session
import argparse
from torch import nn, optim
from torchvision import transforms, datasets, models 




def build_model(arch, hidden_units):
    model = getattr(models, arch)
    
    model = model(pretrained=True)
    model.name = arch
    
    for params in model.parameters():
        params.requires_grad = False
        
    input_size = model.classifier[0].in_features
    hidden_units = hidden_units
    output_size = 102
    
    layers = OrderedDict()
    layers['{}'.format(0)] = nn.Linear(input_size, hidden_units[0])
    layers['{}'.format(1)] = nn.ReLU('inplace')
    layers['{}'.format(2)] = nn.Dropout(p=0.5)
    key = 3
    
    for hidden1, hidden2 in zip(hidden_units[:-1], hidden_units[1:]):
        
        layers['{}'.format(key)] = nn.Linear(hidden1, hidden2)
        layers['{}'.format(key+1)] = nn.ReLU('inplace')
        layers['{}'.format(key+2)] = nn.Dropout(p=0.5)
        
        key += 3
    
    layers['{}'.format(key)] = nn.Linear(hidden_units[-1], output_size)
    layers['{}'.format(key+1)] = nn.LogSoftmax(dim=1)
    
    classifier = nn.Sequential(layers)
    model.classifier = classifier
    return model


def validation(model, testloader, criterion, device):
    test, accuracy = 0, 0
    
    for images, labels in testloader:
    
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test = criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])

        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test, accuracy

def train():
    args = utility.get_inputs()
    image_dataset, dataloader = utility.prepare_data(args.datadir)
    model = build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    model.to(args.gpu)

    with active_session():
        steps = 0
        running_loss = 0
        print_every = 40
        for e in range(args.epochs):

            for images, labels in dataloader['train']:
                model.train()
                steps += 1
                images, labels = images.to(args.gpu), labels.to(args.gpu)

                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()

                    with torch.no_grad():
                        test_loss, accuracy = validation(model, dataloader['test'], criterion, args.gpu)

                    print("Epoch: {}/{}.. ".format(e+1, args.epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(test_loss/len(dataloader['test'])),
                          "Test Accuracy: {:.3f}".format(accuracy/len(dataloader['test'])))

                    running_loss = 0
                    model.train()
    
    model.optimizer_state_dict = optimizer.state_dict()
    model.class_to_idx = image_dataset['train'].class_to_idx
    return model, args.save_dir
        
    

if __name__ == "__main__":
    
    model, save_dir = train()
    utility.save_checkpoint(model, save_dir)
            