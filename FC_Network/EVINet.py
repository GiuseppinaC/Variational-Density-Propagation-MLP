import torch
import torch.nn as nn
import torch.utils.data
from EVI_Layers import EVI_Relu, EVI_FullyConnected, EVI_Softmax
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
class EVINet(nn.Module):
    def __init__(self):
        super(EVINet, self).__init__()
        self.fullyCon1 = EVI_FullyConnected(784, 64, input_flag=True)
        #self.fullyCon2 = EVI_FullyConnected(800, 800)
        self.fullyCon3 = EVI_FullyConnected(64, 10)
        self.relu = EVI_Relu()

        self.softmax = EVI_Softmax(1)

    def forward(self, x_input):
        mu_flat = torch.flatten(x_input.permute([0, 2, 3, 1]), start_dim=1)

        mu_1, sigma_1 = self.fullyCon1.forward(mu_flat)
        mu_2, sigma_2 = self.relu(mu_1, sigma_1)

        #mu_3, sigma_3 = self.fullyCon2.forward(mu_2, sigma_2)
        #mu_4, sigma_4 = self.relu(mu_3, sigma_3)

        mu_5, sigma_5 = self.fullyCon3.forward(mu_2, sigma_2)
        mu_y, sigma_y = self.softmax.forward(mu_5, sigma_5)

        return mu_y, sigma_y

    def nll_gaussian(self, y_pred_mean, y_pred_sd, y_test):
        thing = torch.tensor(1e-3)
        y_pred_sd_inv = torch.inverse(y_pred_sd + torch.diag(thing.repeat([self.fullyCon3.out_features])))
        mu_ = y_pred_mean - y_test
        mu_sigma = torch.bmm(mu_.unsqueeze(1), y_pred_sd_inv)
        ms = 0.5 * torch.bmm(mu_sigma, mu_.unsqueeze(2)).squeeze(1) + 0.5 * torch.log(
            torch.det(y_pred_sd + torch.diag(thing.repeat([self.fullyCon3.out_features])))).unsqueeze(1)
        ms = ms.mean()
        return ms

    def batch_loss(self, output_mean, output_sigma, label):
        output_sigma_clamp = torch.clamp(output_sigma, 1e-10, 1e+10)
        tau = 0.002
        log_likelihood = self.nll_gaussian(output_mean, output_sigma_clamp, label)
        loss_value = log_likelihood + tau * (self.fullyCon1.kl_loss_term() +  self.fullyCon3.kl_loss_term())
        return loss_value

    def batch_accuracy(self, output_mean, label):
        _, bin = torch.max(output_mean.detach(), dim=1)
        comp = bin == label.detach()
        batch_accuracy = comp.sum().cpu().numpy()/len(label)
        return batch_accuracy