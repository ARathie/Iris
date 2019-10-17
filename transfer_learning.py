from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from ToyDataset import ToyDataset
from tqdm import tqdm

# loading data
transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
train_set = ToyDataset('data/train',transforms)#datasets.ImageFolder("data/train",transforms)
val_set   = datasets.ImageFolder("data/train",transforms)
test_set = ToyDataset('data/test',transforms)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10,
                                       shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                       shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                       shuffle=True, num_workers=4)
classes = train_set.classes
device = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")


# model building
model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, train_set.class_count())
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader)):
        inputs = data['image']
        labels = data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(running_loss)
print('Finished Training')


# test with an Image
print('Testing Images')
from PIL import Image
model.eval()
correct_count = 0
for i, data in enumerate(test_loader):
    inputs = data['image']
    labels = data['label']
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    print("Correct label: {} Predicted label: {}".format(test_set.get_label_from_id(labels[0]),test_set.get_label(outputs)))
    correct_count += 1 if test_set.get_label(labels) == test_set.get_label(outputs) else 0
print("\n\nTotal Correct:\t{}/{}\n\n".format(correct_count,len(test_loader)))


"""
# validation
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

"""
