import os
import sys
import time
import inspect
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from skimage.util import random_noise

from EVINet import EVINet


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

####################################################################################################
#
#   Author: Chris Angelini
#
#   Purpose: Extension of Dera et. Al. Bayesian eVI framework into Pytorch
#            The file is used for the creation of the eVI network structure and training loop
#
#   ToDo: Comment
#
####################################################################################################


class Args():
    def __init__(self):
        self.loadModel = 'y'
        self.cuda = True
        self.epochs     = 10
        self.batch_size = 100
        self.lr         = 0.001
        self.num_labels = 10
        self.gaussian_noise=0.5
        self.momentum = 0.9

if __name__ == '__main__':
    args = Args()

    startTime = time.time()

    if args.loadModel == 'Y':
        model_path = False
        training = 't'
    else:
        model_path = True
        training = 'f'

    lr = args.lr
    cuda = args.cuda
    epochs = args.epochs
    batch_size = args.batch_size
    num_labels = args.num_labels
    g_noise=args.gaussian_noise
    mom=args.momentum
    sigma_noise=np.sqrt(g_noise)

    print('Hyper Parameters')
    print('Learning Rate: ' + str(lr))
    print('Epochs: ' + str(epochs))
    print('Batch Size: ' + str(batch_size))
    print('Gaussian noise std: ' + str(sigma_noise))
    print(' ')

    if cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device('cuda')

        eVINet = EVINet()
        print('Using Cuda')
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            eVINet = nn.DataParallel(eVINet)
        eVINet.to(device)
    else:
        eVINet = EVINet()
        device = torch.device('cpu')
        print('Using CPU')

    print(' ')

    if not model_path:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/mnist/data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])),
            batch_size=batch_size, shuffle=True)

        optim = torch.optim.SGD(eVINet.parameters(), lr=lr, momentum=mom)

        eVINet.train()

        batch_format_thing = '{:>' + str(len(str(len(train_loader)))) + '}'
        epoch_format_thing = '{:>' + str(len(str(epochs))) + '}'

        #new part to graph
        train_acc = []
        train_counter = []
        test_losses = []       

        for i_ep in range(epochs):
            epoch_acc = 0
            epochTime = time.time()
            batch100Time = time.time()
            acc_group = np.zeros([1, 100])
            loss_group = np.zeros([1, 100])
            for idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)

                labels = nn.functional.one_hot(targets, num_labels)

                mu_y_out, sigma_y_out = eVINet.forward(data)

                loss = eVINet.batch_loss(mu_y_out, sigma_y_out, labels)
            
                if idx == 0:
                    target_group = targets.unsqueeze(1).detach().cpu().numpy()
                    _, out_mean = torch.max(mu_y_out.detach().cpu(), dim=1)
                    pred_group = out_mean.unsqueeze(1).numpy()
                    loss_group = loss.detach().cpu().numpy()
                else:
                    target_group = np.vstack((target_group, targets.unsqueeze(1).detach().cpu().numpy()))
                    _, out_mean = torch.max(mu_y_out.detach().cpu(), dim=1)
                    pred_group = np.vstack((pred_group, out_mean.unsqueeze(1).numpy()))
                    loss_group = np.vstack((target_group, loss.detach().cpu().numpy()))

                # Print Update on Epoch
                if idx != 0 and ((idx % 100 == 0) or (idx) == len(train_loader)):
                    # Compute Epoch's current accuracy
                    comp = pred_group == target_group
                    epoch_accuracy = np.round(np.true_divide(comp.sum(), len(target_group)), 3)
                    #train_acc.append(epoch_accuracy)
                    # Print information
                    print('Epoch: ' + epoch_format_thing.format(str(i_ep + 1)) +
                          '\tBatch: ' + batch_format_thing.format(str(idx)) + ' of ' + str(len(train_loader)) + ' in ' +
                          time.strftime("%M:%S", time.gmtime(time.time() - batch100Time)) +
                          '\t Batch Mean Loss (past ' + str(100) + '): ' + str(
                        np.round(np.mean(loss_group[-100:]), 3)) +
                          '\t Epoch Accuracy: ' + str(epoch_accuracy))

                    batch100Time = time.time()
                comp2 = pred_group == target_group
                epoch_acc = np.round(np.true_divide(comp2.sum(), len(target_group)), 3)
                train_acc.append(epoch_acc)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_counter.append(i_ep)
            print('Epoch Time: ' + time.strftime("%H:%M:%S", time.gmtime(time.time() - epochTime)))
        print('Total Training Time: ' + time.strftime("%H:%M:%S", time.gmtime(time.time() - startTime)))

        start_date_str = time.strftime("%m_%d_%y", time.localtime(startTime))
        start_time_str = time.strftime("%H_%M", time.localtime(startTime))
        if not os.path.exists('./models'):
            os.makedirs('./models')
        if not os.path.exists('./models/' + start_date_str):
            os.makedirs('./models/' + start_date_str)
        if not os.path.exists('./models/' + start_date_str + '/' + start_time_str):
            os.makedirs('./models/' + start_date_str + '/' + start_time_str)
        model_path = './models/' + start_date_str + '/' + start_time_str + '/'

        torch.save(eVINet.state_dict(), model_path + 'model.pkl')
        fig = plt.figure()
        plt.plot(train_counter, train_acc, color='blue')
        plt.title('model accuracy_final value: {}'.format(epoch_acc))
        plt.legend(['Training Accuracy'], loc='upper right')
        plt.xlabel('epochs')
        plt.ylabel(' loss')
        fig.savefig( model_path  +'/training.png')
    else:
        model_path ='./models/10_08_20/18_59/'
        eVINet.load_state_dict(torch.load(model_path+'model.pkl'))

    splits, file = os.path.split(model_path)
    if g_noise>0:
        # added on 10_04
        #testing with Gaussian Noise

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/mnist/data', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor()
                         ])))

        print('Testing with Gaussian Noise')
        eVINet.eval()
        correct = 0
        total = 0
        mu_y_out = np.empty([len(test_loader), 1, 10])
        sigma_y_out = np.empty([len(test_loader), 1, 10, 10])
        predicted_out = np.empty(len(test_loader))
        for idx, (data, targets) in enumerate(test_loader):
            
            data, targets = data.to(device), targets.to(device)
            data =data + torch.randn(data.size()) * sigma_noise #adding noise
            print('data shape', data.shape)
            mu_y, sigma_y = eVINet.forward(data)
            mu_y_out[idx, :] = mu_y.detach().cpu().numpy()
            sigma_y_out[idx, :] = sigma_y.detach().cpu().numpy()
            _, predicted = torch.max(mu_y, 1)
            predicted_out[idx] = predicted.detach().cpu().numpy()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            if ((idx + 1) % 100 == 0):
                print(str(idx + 1) + ' of 10000 test images: ' + str(round(100 * correct / total, 5)) + '%')
    

    

    else:

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/mnist/data', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                         ])))

        eVINet.eval()
        correct = 0
        total = 0
        mu_y_out = np.empty([len(test_loader), 1, 10])
        sigma_y_out = np.empty([len(test_loader), 1, 10, 10])
        predicted_out = np.empty(len(test_loader))
        for idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            mu_y, sigma_y = eVINet.forward(data)
            mu_y_out[idx, :] = mu_y.detach().cpu().numpy()
            sigma_y_out[idx, :] = sigma_y.detach().cpu().numpy()
            _, predicted = torch.max(mu_y, 1)
            predicted_out[idx] = predicted.detach().cpu().numpy()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            if ((idx + 1) % 100 == 0):
                print(str(idx + 1) + ' of 10000 test images: ' + str(round(100 * correct / total, 5)) + '%')

        
    if not os.path.exists(model_path + 'Test_with_{}_noise'.format(g_noise)):
        os.makedirs(model_path + 'Test_with_{}_noise'.format(g_noise))
        model_path = model_path + 'Test_with_{}_noise'.format(g_noise) +'/'

    np.save(model_path+ 'mu_values_noise_{}.npy'.format(g_noise), mu_y_out)
    np.save(model_path+ 'sigma_values_noise_{}.npy'.format(g_noise), sigma_y_out)
    np.save(model_path + 'predicted_values_noise_{}.npy'.format(g_noise), predicted_out)    
        
    textfile = open( model_path + 'Related_hyperparameters.txt','w')    
    textfile.write(' Batch Size : ' +str(batch_size))
    textfile.write('\n No Hidden Nodes : 64')
    textfile.write('\n Output Size : ' +str(num_labels))
    textfile.write('\n No of epochs : ' +str(epochs))
    textfile.write('\n Learning rate : ' +str(lr))     
    textfile.write('\n Momentum term : ' +str(mom))       
    textfile.write("\n---------------------------------")
    
    if args.loadModel == '':
        textfile.write("\n Training  Accuracy : "+ str( epoch_acc))
        textfile.write("\n Final Test Accuracy : "+ str(round(100 * correct / total, 5) ))   
        textfile.write("\n---------------------------------")
        textfile.write('\n Random Noise std: '+ str(sigma_noise )) 
    else:
        textfile.write("\n Final Test Accuracy : "+ str(round(100 * correct / total, 5) ))   
        textfile.write("\n---------------------------------")
        textfile.write('\n Random Noise std: '+ str(sigma_noise ))
        
    textfile.write("\n---------------------------------")    
    textfile.close()
    data=data.detach().cpu().numpy()
    targets=targets.detach().cpu().numpy()
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig.savefig(splits +'testing_images.png')

    