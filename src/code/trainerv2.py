from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time

import numpy as np
import torchfile

from miscc.config import cfg
from miscc.utilsv2 import mkdir_p
from miscc.utilsv2 import weights_init
from miscc.utilsv2 import save_img_results, save_model
from miscc.utilsv2 import compute_discriminator_loss, compute_generator_loss

from tensorboard import summary
from tensorboardX import FileWriter


class GANTrainer(object):
    def __init__(self, output_dir):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        #torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        from modelv2 import STAGE1_G, STAGE1_D
        netG = STAGE1_G()
        netG.apply(weights_init)
        print(netG)
        netD = STAGE1_D()
        netD.apply(weights_init)
        print(netD)

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    # ############# For training stageII GAN  #############
    def load_network_stageII(self):
        from modelv2 import STAGE1_G, STAGE2_G, STAGE2_D

        Stage1_G = STAGE1_G()
        netG = STAGE2_G(Stage1_G)
        netG.apply(weights_init)
        print(netG)
        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        elif cfg.STAGE1_G != '':
            state_dict = \
                torch.load(cfg.STAGE1_G,
                           map_location=lambda storage, loc: storage)
            netG.STAGE1_G.load_state_dict(state_dict)
            print('Load from: ', cfg.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = STAGE2_D()
        netD.apply(weights_init)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        print(netD)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def train(self, data_loader, stage=1):
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        lr_decay_factor = 0.5
        
        with torch.no_grad():
            fixed_noise = \
                Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))

        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        optimizerD = \
            optim.Adam(netD.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        count = 0
       
        print("GPUs: " + str(self.gpus))

        epoch_init = cfg.TRAIN.EPOCH_INIT
        print ("Training from epoch {}".format(epoch_init))
               
        #Adjust learning rate for a loaded model
        if (epoch_init > 0):
            num_decays = (epoch_init // lr_decay_step) * 1.
            if epoch_init % lr_decay_step == 0: num_decays -= 1. #Guaranteed to decay on first step
            generator_lr = cfg.TRAIN.GENERATOR_LR * (lr_decay_factor**num_decays)
            discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR * (lr_decay_factor**num_decays)
            
            print("Adjusted G/D learning rates: {}, {}".format(generator_lr, discriminator_lr))
        else:
            print("Initial G/D learning rates: {}, {}".format(generator_lr, discriminator_lr))
           
        g_losses, d_losses, d_accs = [], [], []
        for epoch in range(epoch_init, self.max_epoch + 1):
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= lr_decay_factor
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= lr_decay_factor
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

            for i, data in enumerate(data_loader, 0):
                ######################################################
                # (1) Prepare training data
                ######################################################

                real_img_cpu, txt_embedding = data #txt_embedding is actually context embedding vector
                real_imgs = Variable(real_img_cpu)
                txt_embedding = Variable(txt_embedding).float()
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    txt_embedding = txt_embedding.cuda()

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
                _, fake_imgs = \
                    nn.parallel.data_parallel(netG, inputs, self.gpus)
                
                if stage == 2:
                    fake_imgs = nn.functional.avg_pool2d(fake_imgs, 2)
                    real_imgs = nn.functional.avg_pool2d(real_imgs, 2)
                    
                #real_imgs and fake_imgs should be (64x64) if stage 1, (128x128) if stage 2
                # Note: Model saves non-downsampled (256x256) fakes at test time
                
                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                
                # Run discriminator on real and fake images to generate real and fake classpreds and reconstructions
                clspred_real, recon_real =\
                    nn.parallel.data_parallel(netD, (real_imgs), self.gpus)
                clspred_fake, recon_fake =\
                    nn.parallel.data_parallel(netD, (fake_imgs), self.gpus)
               
                errD = compute_discriminator_loss(real_imgs, fake_imgs, recon_real, clspred_fake, clspred_real, txt_embedding)
                
#                 print(errD, errD.size())
                errD.backward(retain_graph=True)
                optimizerD.step()
                
                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()

                errG = compute_generator_loss(clspred_fake, fake_imgs, recon_fake, txt_embedding)
#                 print(errG, errG.size())
                errG.backward()
                optimizerG.step()           

                count = count + 1
                if i % 10 == 0:
                    print ('Epoch: ' + str(epoch) + ' iteration: ' + str(i), flush=True)
                    print ('D_loss: ' + str(errD.data.item()), flush=True)
                    print ('G_loss: ' + str(errG.data.item()), flush=True)
                    accuracy = np.mean(torch.argmax(clspred_real, 1).cpu().numpy() == torch.argmax(txt_embedding, 1).cpu().numpy())
                    print('Discriminator accuracy: {}'.format(accuracy))
                    g_losses.append(errG)
                    d_losses.append(errD)
                    d_accs.append(accuracy)
                    
            end_t = time.time()
            print('''[%d/%d] Loss_D: %.4f Loss_G: %.4f
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch,
                     errD.data.item(), errG.data.item(),
                     (end_t - start_t)))
            
            inputs = (txt_embedding, fixed_noise)
            lr_fake, fake = \
                nn.parallel.data_parallel(netG, inputs, self.gpus)
            save_img_results(real_img_cpu, fake, epoch, self.image_dir)
            
            if lr_fake is not None:
                print ("Saving generated images for epoch " + str(epoch))
                save_img_results(None, lr_fake, epoch, self.image_dir)
                
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)
        
        g_losses = np.save("../../results/G_losses.npy", np.array(g_losses))
        d_losses = np.save("../../results/D_losses.npy", np.array(d_losses))
        d_accs = np.save("../../results/D_accs.npy", np.array(d_accs))
        save_model(netG, netD, self.max_epoch, self.model_dir)
        

    def sample(self, datapath, stage=1):
        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        # Load text embeddings generated from the encoder
        t_file = torchfile.load(datapath)
        captions_list = t_file.raw_txt
        embeddings = np.concatenate(t_file.fea_txt, axis=0)
        num_embeddings = len(captions_list)
        print('Successfully load sentences from: ', datapath)
        print('Total number of sentences:', num_embeddings)
        print('num_embeddings:', num_embeddings, embeddings.shape)
        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        mkdir_p(save_dir)

        batch_size = np.minimum(num_embeddings, self.batch_size)
        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        if cfg.CUDA:
            noise = noise.cuda()
        count = 0
        while count < num_embeddings:
            if count > 3000:
                break
            iend = count + batch_size
            if iend > num_embeddings:
                iend = num_embeddings
                count = num_embeddings - batch_size
            embeddings_batch = embeddings[count:iend]
            txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
            if cfg.CUDA:
                txt_embedding = txt_embedding.cuda()

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            inputs = (txt_embedding, noise)
            _, fake_imgs = \
                nn.parallel.data_parallel(netG, inputs, self.gpus)
            for i in range(batch_size):
                save_name = '%s/%d.png' % (save_dir, count + i)
                im = fake_imgs[i].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                # print('im', im.shape)
                im = np.transpose(im, (1, 2, 0))
                # print('im', im.shape)
                im = Image.fromarray(im)
                im.save(save_name)
            count += batch_size

