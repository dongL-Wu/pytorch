'''
1.data:2020.3.4
2.author:eastLight
3.Purpose:training a classifier
4.datasets:CIFAR-10 
'''

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn   #class torch.nn.Module所有网络的基类
import torch.nn.functional as F   #Convolution 函数
import torch.optim as optim

'''
Loading and normalizing CIFAR10
'''
#transforms.Compose归一化到[-1.0, 1.0]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])

trainset = torchvision.datasets.CIFAR10('./data', train=True,
                                        download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,    #实现对一个batch进行处理
                                        shuffle = True, num_workers = 2)

testset =torchvision.datasets.CIFAR10(root = './data', train = False,
                                    download = True, transform = transform)
testloader= torch.utils.data.DataLoader(testset, batch_size = 4,
                                        shuffle= False, num_workers= 2)

classes = ('plane', 'car', 'bird', 'cat', 
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')                                        

#show some training images
def imshow(image):
    image = image/2+0.5
    npimage = image.numpy()
    plt.imshow(np.transpose(npimage,(1,2,0)))
    plt.show()

#show training images
'''
if __name__ == '__main__':
    #get some random training images
    #dataiter = iter(testloader)
    dataiter = iter(trainloader)   #dataloader本质是一个可迭代对象，使用iter()访问，不能使用next()访问
    images, labels = dataiter.next()  #使用iter(dataloader)返回的是一个迭代器，然后可以使用next访问
    #show images
    imshow(torchvision.utils.make_grid(images))  #make_grid的作用是将若干幅图像拼成一幅图像
    #print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''

#Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 *5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


#define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr= 0.001, momentum=  0.9)

if __name__ == '__main__':
    #train the work
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            #get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward +optimize
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            #print statistics
            running_loss += loss.item()
            if i % 2000 ==1999:  #print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss/2000))
                running_loss =0.0

    print("Finished Training")

    #save our trained model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    #test the network on the test data
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    #print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ',' '.join('%5s'  % classes[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images))

# next let's load back in out saved model:
net = Net()
net.load_state_dict(torch.load(PATH))

# let's see what the neural network thinks these examples above are:
outputs = net(images)

# The outputs are energies for the 10 classes. The higher the energy for a 
# class, the more the network thinks that the image if of the particular 
# class. So, let's get the index of highest energy:
_, predicted = torch.max(outputs, 1)
print('Predicted:', ' '.join('%5s' % classes[predicted[j]]
                             for j in range(4)))

# how the network performs on the whole dataset
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the net work on the 10000 test images: %d %%' %(
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


''' Training on GPU
    Just like how you transfer a Tensor onto the GPU, you transfer the neural
    net onto the GPU.
'''
# Let's first define out device as the first visible cuda device if we have
# CUDA available:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors
net.to(device)

# Remember that you will have to send the inputs and targets at every step to the GPU too.
inputs, labels = data[0].to(device), data[1].to(device)