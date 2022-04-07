""" (WGAN) https://arxiv.org/abs/1701.07875
Wasserstein GAN

The output of WGAN's D is unbounded unless passed through an activation
function. In this implementation, we include a sigmoid activation function
as this empirically improves visualizations for binary MNIST.

WGAN utilizes the Wasserstein distance to produce a value function which has
better theoretical properties than the vanilla GAN. In particular, the authors
prove that there exist distributions for which Jenson-Shannon, Kullback-Leibler,
Reverse Kullback Leibler, and Total Variaton distance metrics where Wasserstein
does. Furthermore, the Wasserstein distance has guarantees of continuity and
differentiability in neural network settings where the previously mentioned
distributions may not. Lastly, they show that that every distribution that
converges under KL, reverse-KL, TV, and JS divergences also converges under the
Wasserstein divergence and that a small Wasserstein distance corresponds to a
small difference in distributions. The downside is that Wasserstein distance
cannot be tractably computed directly. But if we make sure the discriminator
(aka Critic because it is not actually classifying) lies in the space of
1-Lipschitz functions, we can use that to approximate it instead. We crudely
enforce this via a weight clamping parameter C.

Note that this implementation uses RMSprop optimizer instead of Adam, as per
the original paper.
"""
import math
import os
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
import utils as utils
import torch.distributions as tdist

class Generator(nn.Module):
    """ Generator. Input is noise, output is a generated vector.
    data_size:input how much data
    z_dim:output/generate how much data
    """
    def __init__(self, data_size, hidden_dim, z_dim,f):
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.generate = nn.Linear(hidden_dim, data_size)

        self.relu = nn.ReLU()
        self.f = f
    def forward(self, x):
        activated = F.relu(self.linear(x))
        generation = torch.sigmoid(self.generate(activated))
        return generation


class Discriminator(nn.Module):
    """ Critic (not trained to classify). Input is an vector (real or generated),
    output is the approximate Wasserstein Distance between z~P(G(z)) and real.
    """
    def __init__(self, data_size, hidden_dim, output_dim,f,num_classes):
        super().__init__()
        self.num_classes=num_classes
        self.linear = nn.Linear(data_size, hidden_dim)
        self.discriminate = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.f = f

    def forward(self, x):

        activated = F.relu(self.linear(x))
        discrimination = torch.sigmoid(self.discriminate(activated))

        return discrimination

class Classifier(nn.Module):
    def __init__(self, data_size, hidden_dim, out_dim, f, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(data_size, hidden_dim)
        #self.classify = nn.Linear(hidden_dim, output_dim)
        self.classify = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.f = f

    def forward(self, x):
       # /(x)#200,10,17
        activated = F.relu(self.linear(x))#200,10,17

        a=self.classify(activated)
        classification = torch.sigmoid(a)
        #print(classification)
        return classification

class MMGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """
    def __init__(self, data_size, hidden_dim, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G1 = Generator(data_size, hidden_dim, z_dim, torch.sigmoid)
        self.G2 = Generator(data_size, hidden_dim, z_dim, torch.sigmoid)
        self.D = Discriminator(data_size, hidden_dim, output_dim,torch.tanh,5)
        self.C=Classifier(data_size,hidden_dim,output_dim,torch.tanh,5)
        self.shape = int(data_size ** 0.5)


class MMGANTrainer:
    """ Object to hold data iterators, train a GAN variant
    """
    def __init__(self, model, train_iter,  test_iter, train_dataset,test_dataset,label_iter,viz=False):
        self.model = utils.to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.train_dataset=train_dataset

        self.test_iter = test_iter
        self.test_dataset=test_dataset
        self.label_iter=label_iter
        self.Glosses = []
        self.Dlosses = []
        self.Closses = []
        self.count=0
        self.viz = viz
        self.num_epochs = 0

    def getEntropy(self,s):
        # 找到各个不同取值出现的次数
        if not isinstance(s, pd.core.series.Series):
            s = pd.Series(s)
        prt_ary = s.groupby(by=s).count().values / float(len(s))
        return -(np.log2(prt_ary) * prt_ary).sum()

    def train(self, num_epochs, G1_lr=5e-5,G2_lr=5e-5, D_lr=5e-5,C_lr=5e-5, D_steps=5, clip=0.01):
        """ Train a Wasserstein GAN

            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.

        Inputs:
            num_epochs: int, number of epochs to train for
            G_lr: float, learning rate for generator's RMProp optimizer
            D_lr: float, learning rate for discriminator's RMSProp optimizer
            D_steps: int, ratio for how often to train D compared to G
            clip: float, bound for parameters [-c, c] to enforce K-Lipschitz

            a)将从高斯分布 Pz1 和 Pz2 中采样的噪声 z 输入到
            Generator1 和 Generator2 中，得到伪样本 G1(z)和 G2(z)。
            b) 将 G1(z) 、 G2(z) 以 及 真 实 样 本 x 输 入 到 判 别 器
            Discriminator 中，并利用式(6)中定义的损失函数来调整判别
            器的参数。
            c)使用真实样本 x 以及对应的真实标签 y 来训练分类器，
            损失函数被定义在式(7)中。
            d)保持判别器和分类器的参数不变。使用式(8)来同时优
            化生成器 Generator1 和 Generator2 的参数，并且保证两个生
            成器之间共享部分权重参数。
            e)以上四个步骤重复执行，直至设定的 epoch 被达到。
        """
        # Initialize optimizers
        G1_optimizer = optim.Adam(params=[p for p in self.model.G1.parameters()
                                        if p.requires_grad], lr=G1_lr)
        G2_optimizer = optim.Adam(params=[p for p in self.model.G1.parameters()
                                         if p.requires_grad], lr=G2_lr)
        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters()
                                        if p.requires_grad], lr=D_lr)
        C_optimizer = optim.Adam(params=[p for p in self.model.C.parameters()
                                         if p.requires_grad], lr=C_lr)

        # Approximate steps/epoch given D_steps per epoch
        # --> roughly train in the same way as if D_step (1) == G_step (1)
        epoch_steps = int(np.ceil(len(self.train_iter) / (D_steps)))
        # a)将从高斯分布 Pz1 和 Pz2 中采样的噪声 z 输入到 Generator1 和 Generator2 中，得到伪样本 G1(z)和 G2(z)

        # Begin training
        for epoch in tqdm(range(1, num_epochs+1)):

            self.model.train()
            C_losses,G_losses, D_losses = [], [] ,[]

            for _ in range(epoch_steps):

                D_step_loss = []
                C_step_loss=[]
                for _ in tqdm(range(1,D_steps+1)):
                    self.count+=1
                    # Reshape
                    try:

                        data = self.process_batch(self.train_iter)
                        label=self.process_batch(self.label_iter)
                    except StopIteration:
                        # 遇到StopIteration就退出循环
                        print(self.count)
                    # TRAINING D: Zero out gradients for D
                    D_optimizer.zero_grad()

                    # Train the discriminator to approximate the Wasserstein
                    # distance between real, generated distributions
                    D_loss = self.train_D(data)

                    # Update parameters
                    D_loss.backward()
                    D_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    D_step_loss.append(D_loss.item())

                    # Clamp weights (crudely enforces K-Lipschitz)
                    self.clip_D_weights(clip)


                    # Train C
                    C_optimizer.zero_grad()

                    # Train the discriminator to approximate the Wasserstein
                    # distance between real, generated distributions
                    C_loss = self.train_C(data,label)

                    # Update parameters
                    C_loss.backward()
                    C_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    C_step_loss.append(C_loss.item())

                    # Clamp weights (crudely enforces K-Lipschitz)
                    #self.clip_D_weights(clip)
                    print("step[%d/%d],  C Loss: %.4f  D Loss: %.4f"
                          % (self.count, num_epochs, C_loss,D_loss))


                # We report D_loss in this way so that G_loss and D_loss have
                # the same number of entries.
                D_losses.append(np.mean(D_step_loss))
                C_losses.append(np.mean(C_step_loss))

                # TRAINING G: Zero out gradients for G
                G1_optimizer.zero_grad()
                G2_optimizer.zero_grad()
                # Train the generator to (roughly) minimize the approximated
                # Wasserstein distance
                G_loss = self.train_G(data)

                # Log results, update parameters
                G_losses.append(G_loss.item())
                G_loss.backward()
                G1_optimizer.step()
                G2_optimizer.step()
                self.count=0
                print("一轮完成")
            # Save progress
            print("一大轮完成")
            self.Glosses.extend(G_losses)
            self.Dlosses.extend(D_losses)
            self.Closses.extend(C_losses)
            # Progress logging
            print ("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
                   %(epoch, num_epochs, np.mean(G_losses), np.mean(D_losses)))
            self.num_epochs += 1

            # Visualize generator progress
            if self.viz:
                self.generate_images(epoch)
                plt.show()
        self.viz_loss()

    def output(self):







    def train_D(self, data):
        """ Run 1 step of training for discriminator

        Input:
            output of G1&G2
        Output:
            D_loss: E(ln(1-GX))+E[ln(1-D(G1))] +E[ln(1-D(G2))]
        """
        # Sample from the generator
        noise = self.compute_noise(data.shape[0], self.model.z_dim)
        G1_output = self.model.G1(noise)
        G2_output = self.model.G2(noise)
        # Score real, generated images
        DX_score = self.model.D(data.float()) # D(x), "real"
        DG1_score = torch.log(torch.ones(200, 17)-self.model.D(G1_output)) # D(G(x')), "fake"
        DG2_score = torch.log(torch.ones(200, 17)-self.model.D(G2_output))  # D(G(x')), "fake"
        # Compute MMGAN loss for D
        D_loss = torch.mean(DX_score) + torch.mean(DG1_score)+torch.mean(DG2_score)

        return D_loss
    def trans(self,npl):
        res=[]
        for i in range(len(npl)):
            res.append(npl[i][0])
        return np.array(res)
    def train_G(self, data):
        """ Run 1 step of training for generator

        Input:
            data: noise
        Output:

            maxEln[D(G(z1))+Eln[D(G(z1))+H(C(G1))+H(C(G2)))
        """
        # Get noise, classify it using G, then classify the output of G using D.
        noise1 = self.compute_noise(data.shape[0], self.model.z_dim) # z
        noise2 = self.compute_noise(data.shape[0], self.model.z_dim)  # z
        G1_output = self.model.G1(noise1) # G(z1)
        G2_output = self.model.G2(noise2)  # G(z2)
        DG1_score = self.model.D(G1_output)  # D(G(x')), "fake"
        DG2_score = self.model.D(G2_output) # D(G(x')), "fake"
        C1_output=self.getEntropy(self.trans(self.model.C(G1_output).detach().numpy()))
        C2_output=self.getEntropy(self.trans(self.model.C(G2_output).detach().numpy()))
        H=C1_output+C2_output
        # Compute MMGAN loss for G
        G_loss = (torch.mean(torch.log(DG1_score))+torch.mean(torch.log(DG2_score))+H).max()

        return G_loss

    def train_C(self,data,target):
        """ Run 1 step of training for classifier

         Input:
             data: real-data
         Output:

             min (E(D(y,C(x)))
         """
       # print(data.float())
        C_output=self.model.C(data.float()).double() #C(x)
        target= F.log_softmax(target).double()  #y
        criterion = nn.KLDivLoss()
        klloss = criterion(target, C_output)
        C_loss=torch.mean(klloss).min()
        return  C_loss

    def compute_noise(self, batch_size, z_dim):
        """ Compute random noise for input into the Generator G """
        return utils.to_cuda(torch.randn(batch_size, z_dim))

    def process_batch(self, iterator):
        """ Generate a process batch to be input into the Discriminator D """
        data =next(iter(iterator))
            #batch_iterator = iter(data.DataLoader(train_dataset, batch_size=100, shuffle=True))
            #data= next(iter(batch_iterator))
        data = utils.to_cuda(data)
        return data

    def clip_D_weights(self, clip):
        for parameter in self.model.D.parameters():
            parameter.data.clamp_(-clip, clip)

    def generate_images(self, epoch, num_outputs=36, save=True):
        """ Visualize progress of generator learning """
        # Turn off any regularization
        self.model.eval()

        # Sample noise vector
        noise = self.compute_noise(num_outputs, self.model.z_dim)

        # Transform noise to image
        images = self.model.G(noise)

        # Reshape to square image size
        images = images.view(images.shape[0],
                             self.model.shape,
                             self.model.shape,
                             -1).squeeze()

        # Plot
        plt.close()
        grid_size, k = int(num_outputs**0.5), 0
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
        for i, j in product(range(grid_size), range(grid_size)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].imshow(images[k].data.numpy(), cmap='gray')
            k += 1

        # Save images if desired
        if save:
            outname = '../viz/' + self.name + '/'
            if not os.path.exists(outname):
                os.makedirs(outname)
            torchvision.utils.save_image(images.unsqueeze(1).data,
                                         outname + 'reconst_%d.png'
                                         %(epoch), nrow=grid_size)

    def viz_loss(self):
        """ Visualize loss for the generator, discriminator """
        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8,6)

        # Plot Discriminator loss in red
        plt.plot(np.linspace(-2, 1, len(self.Dlosses)),
                 self.Dlosses,
                 'r')

        # Plot Generator loss in green
        plt.plot(np.linspace(-2, 1, len(self.Dlosses)),
                 self.Glosses,
                 'g')

        # Add legend, title
        plt.legend(['Discriminator', 'Generator'])
        plt.title(self.name)
        plt.show()

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)


if __name__ == "__main__":

    # Load in binarized MNIST data, separate into data loaders
    train_iter, test_iter,test_dataset,train_dataset,label_iter = utils.get_data()

    # Init model
    model = MMGAN(data_size=17,
                  hidden_dim=2,
                  z_dim=20)

    # Init trainer
    trainer = MMGANTrainer(model=model,
                          train_iter=train_iter,
                          test_iter=test_iter,
                          train_dataset=train_dataset,
                          test_dataset=test_dataset,
                          label_iter=label_iter,
                          viz=False)

    # Train
    trainer.train(num_epochs=1,
                  G1_lr=5e-5,
                  G2_lr=5e-5,
                  D_lr=5e-5,
                  D_steps=1,
                  clip=0.01)
