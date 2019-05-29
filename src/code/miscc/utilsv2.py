import os
import errno
import numpy as np

from copy import deepcopy
from miscc.config import cfg

from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils


#############################


# # Define D loss
# lreal = log_sum_exp(Opred_n)
# lfake = log_sum_exp(Opred_g)
# cost_On = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Opred_n, labels=y))
# cost_Dn = - tf.reduce_mean(lreal) + tf.reduce_mean(tf.nn.softplus(lreal))
# cost_Dg_fake = tf.reduce_mean(tf.nn.softplus(lfake))
# cost_msen = tf.reduce_mean(tf.square(recon_n - x_n)) * 0.5
# D_loss = cost_On + cost_Dn + cost_Dg_fake + cost_msen


def compute_discriminator_loss(real_imgs, gen_samples, recon_real, fake_logits, real_logits, embeddings):
    labels = torch.argmax(embeddings, dim=1)
    fake_probs = log_sum_exp(fake_logits)
    real_probs = log_sum_exp(real_logits)
    orig_loss = torch.mean(nn.functional.cross_entropy(real_logits, labels))
    disc_real_loss = -torch.mean(real_probs) + torch.mean(nn.functional.softplus(real_probs))
    disc_gen_loss = torch.mean(nn.functional.softplus(fake_probs))
    recon_loss_real = torch.mean(torch.pow(recon_real - real_imgs, 2.0)) * 0.5
    D_loss = orig_loss + disc_real_loss + disc_gen_loss + recon_loss_real
    return D_loss
    
def compute_generator_loss(fake_logits, gen_samples, recon_fake, embeddings):
    embeddings = embeddings.type(torch.long)
    labels = torch.argmax(embeddings, dim=1)
    fake_probs = log_sum_exp(fake_logits)
    recon_loss_fake = torch.mean(torch.pow(recon_fake - gen_samples, 2.0)) * 0.5
    adv_loss = - torch.mean(fake_probs) + torch.mean(nn.functional.softplus(fake_probs)) + torch.mean(nn.functional.cross_entropy(fake_logits, labels))
    
    G_loss = recon_loss_fake + adv_loss
    return G_loss

# cost_mseg = tf.reduce_mean(tf.square(recon_g - samples)) * 0.5

# # Define G loss
# cost_Dg = - tf.reduce_mean(lfake) + tf.reduce_mean(tf.nn.softplus(lfake))
# cost_Og = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Opred_g, labels=iny))
# G_loss = cost_Dg + cost_Og + cost_mseg

# # Define optimizer
# d_optimizer = tf.train.AdamOptimizer(learning_rate=lr_tf, beta1=0.5).minimize(D_loss, var_list=d_vars)
# g_optimizer = tf.train.AdamOptimizer(learning_rate=lr_tf, beta1=0.5).minimize(G_loss, var_list=g_vars)

# # Evaluate model
# Oaccuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Opred_n, 1), tf.argmax(y, 1)), tf.float32))
# #    

#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples.png' % image_dir,
            normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)


def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pth' % (model_dir))
    print('Save G/D models')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
def log_sum_exp(x, axis=1):
    m,_ = torch.max(x, axis, keepdim=True, out=None) 
    return m + torch.log(torch.sum(torch.exp(x - m), dim=axis))


# def KL_loss(mu, logvar):
#     # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
#     KLD = torch.mean(KLD_element).mul_(-0.5)
#     return KLD
# def compute_generator_loss(netD, fake_imgs, real_labels, conditions, gpus):
#     criterion = nn.BCELoss()
#     cond = conditions.detach()
#     fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)
#     # fake pairs
#     inputs = (fake_features, cond)
#     fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
#     errD_fake = criterion(fake_logits, real_labels)
#     if netD.get_uncond_logits is not None:
#         fake_logits = \
#             nn.parallel.data_parallel(netD.get_uncond_logits,
#                                       (fake_features), gpus)
#         uncond_errD_fake = criterion(fake_logits, real_labels)
#         errD_fake += uncond_errD_fake

# def compute_discriminator_loss(netD, real_imgs, fake_imgs,
#                                real_labels, fake_labels,
#                                conditions, gpus):
#     criterion = nn.BCELoss()
#     batch_size = real_imgs.size(0)
#     cond = conditions.detach()
#     fake = fake_imgs.detach()
#     real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
#     fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
#     # real pairs
#     inputs = (real_features, cond)
#     real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
#     errD_real = criterion(real_logits, real_labels)
#     # wrong pairs
#     inputs = (real_features[:(batch_size-1)], cond[1:])
#     wrong_logits = \
#         nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
#     errD_wrong = criterion(wrong_logits, fake_labels[1:])
#     # fake pairs
#     inputs = (fake_features, cond)
#     fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
#     errD_fake = criterion(fake_logits, fake_labels)

#     if netD.get_uncond_logits is not None:
#         real_logits = \
#             nn.parallel.data_parallel(netD.get_uncond_logits,
#                                       (real_features), gpus)
#         fake_logits = \
#             nn.parallel.data_parallel(netD.get_uncond_logits,
#                                       (fake_features), gpus)
#         uncond_errD_real = criterion(real_logits, real_labels)
#         uncond_errD_fake = criterion(fake_logits, fake_labels)
#         #
#         errD = ((errD_real + uncond_errD_real) / 2. +
#                 (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
#         errD_real = (errD_real + uncond_errD_real) / 2.
#         errD_fake = (errD_fake + uncond_errD_fake) / 2.
#     else:
#         errD = errD_real + (errD_fake + errD_wrong) * 0.5
#     return errD, errD_real.data.item(), errD_wrong.data.item(), errD_fake.data.item()