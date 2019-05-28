import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


# class CA_NET(nn.Module):
#     # some code is modified from vae examples
#     # (https://github.com/pytorch/examples/blob/master/vae/main.py)
#     def __init__(self):
#         super(CA_NET, self).__init__()
#         self.t_dim = cfg.TEXT.DIMENSION
#         self.c_dim = cfg.GAN.CONDITION_DIM
#         self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
#         self.relu = nn.ReLU()

#     def encode(self, text_embedding):
#         x = self.relu(self.fc(text_embedding))
#         mu = x[:, :self.c_dim]
#         logvar = x[:, self.c_dim:]
#         return mu, logvar

#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         if cfg.CUDA:
#             eps = torch.cuda.FloatTensor(std.size()).normal_()
#         else:
#             eps = torch.FloatTensor(std.size()).normal_()
#         eps = Variable(eps)
#         return eps.mul(std).add_(mu)

#     def forward(self, text_embedding):
#         mu, logvar = self.encode(text_embedding)
#         c_code = self.reparametrize(mu, logvar)
#         return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# ############# Networks for stageI GAN #############
class STAGE1_G(nn.Module):
    def __init__(self):
        super(STAGE1_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.define_module()

    def define_module(self):
        ninput = self.z_dim + self.ef_dim
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
#         self.ca_net = CA_NET()

        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())

    def forward(self, text_embedding, noise):
        #c_code, mu, logvar = self.ca_net(text_embedding)


        z_c_code = torch.cat((noise, text_embedding), 1)
        h_code = self.fc(z_c_code)

        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        #return None, fake_img, mu, logvar
        return None, fake_img

def gaussnoise(inp, std):
    noise = Variable(inp.data.new(inp.size()).normal_(0, std))
    return ins + noise
    
class STAGE1_D(nn.Module):
    def __init__(self):
        super(STAGE1_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encoder = nn.Sequential(
            # 64
            nn.Conv2d(3, 128, 3, stride=2),
            nn.LeakyReLU(0.2),
            # 32
            nn.Dropout(p = 0.5),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 16
            nn.Dropout(p = 0.5),
            nn.Conv2d(256, 512, 3, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 8
            nn.Conv2d(512, 512, 3, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(512, 1024, 3, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
            # 4
        )
        # 64
#         inp = gaussnoise(inp, std=0.05)
#         conv1 = conv2d(inp, 128, kernel=3, strides=2, name=dname + 'conv1')
#         conv1 = lrelu(conv1, 0.2)
#         # 32
#         conv2 = tf.nn.dropout(conv1, keep_prob)
#         conv2 = conv2d(conv2, 256, kernel=3, strides=2, name=dname + 'conv2')
#         conv2 = batchnorm(conv2, is_training=is_train, name=dname + 'bn2')
#         conv2 = lrelu(conv2, 0.2)
#         # 16
#         conv3 = tf.nn.dropout(conv2, keep_prob)
#         conv3 = conv2d(conv3, 512, kernel=3, strides=2, name=dname + 'conv3')
#         conv3 = batchnorm(conv3, is_training=is_train, name=dname + 'bn3')
#         conv3 = lrelu(conv3, 0.2)
#         # 8
#         conv3b = conv2d(conv3, 512, kernel=3, strides=1, name=dname + 'conv3b')
#         conv3b = batchnorm(conv3b, is_training=is_train, name=dname + 'bn3b')
#         conv3b = lrelu(conv3b, 0.2)

#         conv4 = tf.nn.dropout(conv3b, keep_prob)
#         conv4 = conv2d(conv4, 1024, kernel=3, strides=2, name=dname + 'conv4')
#         conv4 = batchnorm(conv4, is_training=is_train, name=dname + 'bn4')
#         conv4 = lrelu(conv4, 0.2)
#         # 4

#         flat = flatten(conv4)
#         # Classifier
#         clspred = linear(flat, n_classes, name=dname + 'cpred')
        self.clspred = nn.Linear(4 * 4 * 1024, nef)
    
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, 3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'), # 8x8
            nn.Conv2d(512, 256, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'), # 16x16
            nn.Conv2d(256, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'), # 32x32
            nn.Conv2d(128, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'), # 64x64
            nn.Conv2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, 3),
            nn.Tanh()
        )
#         # Decoder
#         g1 = conv2d(conv4, nout=512, kernel=3, name=dname + 'deconv1')
#         g1 = batchnorm(g1, is_training=tf.constant(True), name=dname + 'bn1g')
#         g1 = lrelu(g1, 0.2)

#         g2 = nnupsampling(g1, [8, 8])
#         g2 = conv2d(g2, nout=256, kernel=3, name=dname + 'deconv2')
#         g2 = batchnorm(g2, is_training=tf.constant(True), name=dname + 'bn2g')
#         g2 = lrelu(g2, 0.2)

#         g3 = nnupsampling(g2, [16, 16])
#         g3 = conv2d(g3, nout=128, kernel=3, name=dname + 'deconv3')
#         g3 = batchnorm(g3, is_training=tf.constant(True), name=dname + 'bn3g')
#         g3 = lrelu(g3, 0.2)

#         g4 = nnupsampling(g3, [32, 32])
#         g4 = conv2d(g4, nout=64, kernel=3, name=dname + 'deconv4')
#         g4 = batchnorm(g4, is_training=tf.constant(True), name=dname + 'bn4g')
#         g4 = lrelu(g4, 0.2)

#         g5 = nnupsampling(g4, [64, 64])
#         g5 = conv2d(g5, nout=32, kernel=3, name=dname + 'deconv5')
#         g5 = batchnorm(g5, is_training=tf.constant(True), name=dname + 'bn5g')
#         g5 = lrelu(g5, 0.2)

#         g5b = conv2d(g5, nout=3, kernel=3, name=dname + 'deconv5b')
#         g5b = tf.nn.tanh(g5b)
#         return clspred, g5b

    def forward(self, image):
#         img_embedding = self.encode_img(image)
        print("Encoding")
        img_embedding = self.encoder(image)
        print("Encoded!")
        clspred = self.clspred(img_embedding.view(img_embedding.size(0), -1)
        print("clspred: " + str(clspred))
        print("Decoding")
        decoded_embedding = self.decoder(img_embedding)
        print("Decoded!")

        return clspred, decoded_embedding


# ############# Networks for stageII GAN #############
class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G):
        super(STAGE2_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.STAGE1_G = STAGE1_G
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        #self.ca_net = CA_NET()
        # --> 4ngf x 16 x 16
        self.encoder = nn.Sequential(
            conv3x3(3, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.hr_joint = nn.Sequential(
            conv3x3(self.ef_dim + ngf * 4, ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 4)
        # --> 2ngf x 32 x 32
        self.upsample1 = upBlock(ngf * 4, ngf * 2)
        # --> ngf x 64 x 64
        self.upsample2 = upBlock(ngf * 2, ngf)
        # --> ngf // 2 x 128 x 128
        self.upsample3 = upBlock(ngf, ngf // 2)
        # --> ngf // 4 x 256 x 256
        self.upsample4 = upBlock(ngf // 2, ngf // 4)
        # --> 3 x 256 x 256
        self.img = nn.Sequential(
            conv3x3(ngf // 4, 3),
            nn.Tanh())

    def forward(self, text_embedding, noise):
        _, stage1_img = self.STAGE1_G(text_embedding, noise)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)

        #c_code, mu, logvar = self.ca_net(text_embedding)
        c_code = text_embedding
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        fake_img = self.img(h_code)
#         return stage1_img, fake_img, mu, logvar
        return stage1_img, fake_img


class STAGE2_D(nn.Module):
    def __init__(self):
        super(STAGE2_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),  # 128 * 128 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 64 * 64 * ndf * 2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
            conv3x3(ndf * 32, ndf * 16),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 16
            conv3x3(ndf * 16, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf * 8
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding
