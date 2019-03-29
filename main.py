import time 
import argparse

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

import resnet
import memory_usage

parser = argparse.ArgumentParser(description='Resnet')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', type=int, default=0.1)
parser.add_argument('--epochs', type=int, default=1)

parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--gradient-averaging', action='store_true', default=False)

args = parser.parse_args()
print(args)


def get_data(batch_size, data_dir='/tmp/data', num_workers=4):
    # Handle CIFAR10 and Imagenet
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir,
                                            train=True,
                                            download=True,
                                            transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=num_workers)
    return train_loader

def train(model, train_loader, params):
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    print('Starting training...')
    for epoch in range(params['epochs']):
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                print('{} sec/update'.format((time.time() - start_time) / (i+1)))
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
                cpu_mem, cuda_mem = memory_usage.get_memory_usage()
                print('Mem uage: CPU: {}, CUDA: {}.'.format(cpu_mem, cuda_mem))
            

def main():
    params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr
    }
    model = resnet.resnet50()
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    model.to(device)
    train_loader = get_data(params['batch_size'])
    train(model, train_loader, params)

if __name__ == '__main__':
    main()
