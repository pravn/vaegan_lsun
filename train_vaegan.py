import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
#from torchvision.datasets import MNIST
#from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F


def plot_loss(loss_array,name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_array)
    plt.savefig('loss_'+name)


def weights_init_G(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def KLD(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    #KLD = KLD_element.mul_(-0.5).mean()
    #KLD/=784.0
    batch_size = mu.size(0)

    KLD /= batch_size
    
    return KLD


def run_trainer(train_loader, net_VAE, args):

    LAMBDA = args.LAMBDA
    batch_size = args.batchSize

    optimizer_VAE = optim.Adam(net_VAE.parameters(), lr=args.lr,betas=(args.beta1,0.999))
    VAE_scheduler = StepLR(optimizer_VAE, step_size=1000, gamma=0.8)

    real_label = 1
    fake_label = 0

    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.restart == '':
        net_VAE.apply(weights_init_G)
        #netD.apply(weights_init_D)
        
    else:
        #netD = torch.load('./D_model.pt')
        net_VAE = torch.load('./VAE_model.pt')
        
    criterion_MSE = nn.MSELoss()
    #criterion_cross_entropy = nn.BCELoss()
    
    if args.cuda:
        criterion_MSE = criterion_MSE.cuda()

    for epoch in range(1000):

        data_iter = iter(train_loader)
        i = 0

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            images = images.cuda()
            label = torch.full((batch_size,), real_label, device=device)

            #train netE, netG
            for p in net_VAE.parameters():
                p.requires_grad = True

            net_VAE.zero_grad()

            #reconstruction term 
            recon, z, mu, logvar = net_VAE(images)
            
            recon_loss = criterion_MSE(recon, images)

            KLD_loss = KLD(mu, logvar)
            
            recon_loss.backward(retain_graph=True)
            KLD_loss.backward()
            optimizer_VAE.step()
            
            if  i % 100 == 0 :
                print('saving images for batch', i)
                save_image(recon.squeeze().data.cpu().detach(), './fake.png')
                save_image(images.data.cpu().detach(), './real.png')

            if i % 100 == 0:
                torch.save(net_VAE, './VAE_model.pt')
                #torch.save(netD, './D_model.pt')
                
                print('%d [%d/%d] Loss VAE (recon/kld) [%.4f/%.4f]'%
                      (epoch, i, len(train_loader), 
                       recon_loss, KLD_loss))




     
            
