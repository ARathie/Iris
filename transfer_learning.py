from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from ToyDataset import ToyDataset

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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                       shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                       shuffle=True, num_workers=4)
classes = train_set.classes
device = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")


# model building
model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# training
for epoch in range(25):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
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

# test with an Image
from PIL import Image
model.eval()
img_name = "1.jpeg" # change this to the name of your image file.
def predict_image(image_path, model):
    image = Image.open(image_path)
    image_tensor = transforms(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    index = output.argmax().item()
    if index == 0:
        return "Cat"
    elif index == 1:
        return "Dog"
    else:
        return
predict(img_name,model)
