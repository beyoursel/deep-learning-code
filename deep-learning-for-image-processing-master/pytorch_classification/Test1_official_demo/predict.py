import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


# def main():
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))

im = Image.open('2.jpg')
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W] unsqueeze用于增加一个维度

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy() # .data做了一个内存共享
    # predict = torch.softmax(outputs, dim=1)
print(classes[int(predict)])
# print(predict)


# if __name__ == '__main__':
#     main()
