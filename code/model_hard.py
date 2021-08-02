import torch, math
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ipdb import set_trace

from miscc.config import cfg
from GlobalAttention import GlobalAttentionGeneral as ATT_NET
from GlobalAttention import GlobalAttention_text as ATT_NET_text
from spectral import SpectralNorm

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):  # (N,c_dim*4)
        nc = x.size(1)  # c_dim*4
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)  # c_dim*2
        return x[:, :nc] * torch.sigmoid(x[:, nc:])  # c_dim*2


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)


def conv3x3(in_planes, out_planes, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


# ############## G networks ###################
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM  # 256
        self.c_dim = cfg.GAN.CONDITION_DIM   # 100
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))  # (N,c_dim*2)
        mu = x[:, :self.c_dim]      # (N,c_dim)
        logvar = x[:, self.c_dim:]  # (N,c_dim)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # (N,c_dim)
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # (N,c_dim)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)  # (N,c_dim)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)  # (N,c_dim), (N,c_dim)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar  # (N,c_dim), (N,c_dim), (N,c_dim)


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf  # 64 * 16
        self.in_dim = cfg.GAN.Z_DIM + ncf  # cfg.TEXT.EMBEDDING_DIM

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim  # 200, 1024
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),  # 200, 1024*4*4*2
            nn.BatchNorm1d(ngf * 4 * 4 * 2),  # 1024*4*4
            GLU())  # 1024*4*4

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code, p_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)  # (N,200)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)  # (N,16384)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)    # (N,1024,4,4)

        # prior
        out_code = out_code + p_code  # (N,1024,4,4)

        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)  # (N,512,8,8)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)  # (N,256,16,16)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)    # (N,128,32,32)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)  # (N,64,64,64)

        return out_code64


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context_key, content_value):#
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x idf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context_key.size(0), context_key.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)  # (N,64,h*w)
        targetT = torch.transpose(target, 1, 2).contiguous()  # (N,h*w,64)
        sourceT = context_key  # (N,64,18)

        # Get weight
        # (batch x queryL x idf)(batch x idf x sourceL)-->batch x queryL x sourceL
        weight = torch.bmm(targetT, sourceT)  # (N,h*w,T)

        # --> batch*queryL x sourceL
        weight = weight.view(batch_size * queryL, sourceL)  # (N*h*w,T)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)                  # (N*h*w,T)
            weight.data.masked_fill_(mask.data, -float('inf'))  # (N*h*w,T)
        weight = torch.nn.functional.softmax(weight, dim=1)  

        # --> batch x queryL x sourceL
        weight = weight.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        weight = torch.transpose(weight, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL) --> batch x idf x queryL
        weightedContext = torch.bmm(content_value, weight)  #
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        weight = weight.view(batch_size, -1, ih, iw)

        return weightedContext, weight


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf, size):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf  # 64
        self.ef_dim = nef  # 256
        self.cf_dim = ncf  # 100
        self.num_residual = cfg.GAN.R_NUM  # 2
        self.size = size   # 64 or 128
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):  # 2
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.avg = nn.AvgPool2d(kernel_size=self.size)
        self.A = nn.Linear(self.ef_dim, 1, bias=False)
        self.B = nn.Linear(self.gf_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.M_r = nn.Sequential(
            nn.Conv1d(ngf, ngf * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.M_w = nn.Sequential(
            nn.Conv1d(self.ef_dim, ngf * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.key = nn.Sequential(
            nn.Conv1d(ngf*2, ngf, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Conv1d(ngf*2, ngf, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.memory_operation = Memory()
        self.response_gate = nn.Sequential(
            nn.Conv2d(self.gf_dim * 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask, cap_lens):
        """
            h_code(image features):  batch x idf x ih x iw (queryL=ihxiw)  # (N,64,ih,iw)
            word_embs(word features): batch x cdf x sourceL (sourceL=seq_len)  # (N,256,18)
            c_code: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        # Memory Writing
        word_embs_T = torch.transpose(word_embs, 1, 2).contiguous()  # (N,18,256)
        h_code_avg = self.avg(h_code).detach()  # (N,64,1,1)
        h_code_avg = h_code_avg.squeeze(3)      # (N,64,1)
        h_code_avg_T = torch.transpose(h_code_avg, 1, 2).contiguous()    # (N,1,64)
        gate1 = torch.transpose(self.A(word_embs_T), 1, 2).contiguous()  # (N,1,18)
        gate2 = self.B(h_code_avg_T).repeat(1, 1, word_embs.size(2))     # (N,1,18)
        writing_gate = torch.sigmoid(gate1 + gate2)              # (N,1,18)
        h_code_avg = h_code_avg.repeat(1, 1, word_embs.size(2))  # (N,64,18)
        # (N,128,18)
        memory = self.M_w(word_embs) * writing_gate + self.M_r(h_code_avg) * (1 - writing_gate)

        # Key Addressing and Value Reading
        key = self.key(memory)      # (N,64,18)
        value = self.value(memory)  # (N,64,18)
        self.memory_operation.applyMask(mask)  # mask -> self.memory_operation.mask
        # (N,64,64,64), (N,18,64,64)
        memory_out, att = self.memory_operation(h_code, key, value)

        # Key Response
        response_gate = self.response_gate(torch.cat((h_code, memory_out), 1))  # (N,1,64,64)
        h_code_new = h_code * (1 - response_gate) + response_gate * memory_out  # (N,64,64,64)
        h_code_new = torch.cat((h_code_new, h_code_new), 1)  # (N,128,64,64)

        out_code = self.residual(h_code_new)  # (N,128,64,64)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)    # (N,64,128,128)
        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        ngf = cfg.GAN.GF_DIM          # 64
        nef = cfg.TEXT.EMBEDDING_DIM  # 300
        ncf = cfg.GAN.CONDITION_DIM   # 100
        self.ca_net = CA_NET()
        
        # for priori conv
        self.pri_block = nn.Sequential(
            SpectralNorm(nn.Conv2d(2048, 1024, kernel_size=2, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.query_fc = nn.Linear(nef, 1024, bias=True)
        self.key_fc = nn.Linear(2048, 1024, bias=True)
        self.softmax = nn.Softmax(dim=1)

        if cfg.TREE.BRANCH_NUM > 0:   # 3
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
            self.img_net1 = GET_IMAGE_G(ngf)
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf, 64)
            self.img_net2 = GET_IMAGE_G(ngf)

        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf, 128)
            self.img_net3 = GET_IMAGE_G(ngf)


    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):  # 2
            layers.append(block(channel_num))
        return nn.Sequential(*layers)


    def hard_read(self, query, key, value):
        # query: N,300
        # key:   N,K,2048
        # value: N,K,2048,7,7
        n_batches, K, v_dim, h, w = value.size()
        d_k = query.size(-1)
        
        proj_query = self.query_fc(query).unsqueeze(-1)  # (N,dim,1)
        proj_key = self.key_fc(key.view(-1, 2048)).view(n_batches, K, -1)  # (N,K,dim)
        score = torch.bmm(proj_key, proj_query) / math.sqrt(d_k)  # (N,K,1)
        weights = self.softmax(score).squeeze(-1)  # (N,K)

        # hard reading
        shape = weights.size()
        _, ind = weights.max(dim=-1)  # (N, )
        weights_hard = torch.zeros_like(weights).view(-1, shape[-1])  # (N,K)
        weights_hard.scatter_(1, ind.view(-1, 1), 1)
        weights_hard = weights_hard.view(*shape)
        # Keep the value while 
        # Set gradients w.r.t. weights_hard gradients w.r.t. weights
        weights_hard = (weights_hard - weights).detach() + weights

        value = value.view(n_batches, K, -1).permute(0, 2, 1)           # (N,2048*7*7,K)
        out = torch.bmm(value, weights_hard.unsqueeze(-1)).squeeze(-1)  # (N,2048*7*7)
        out = out.view(n_batches, v_dim, w, h)  # (N,2048,7,7)

        
        return out, ind, weights


    def forward(self, z_code, sent_emb, word_embs, mask, cap_lens, priori):
        """
            :param z_code:    batch x cfg.GAN.Z_DIM
            :param sent_emb:  batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask:      batch x seq_len
            :param priori:    key, value
            :return:
        """
        fake_imgs = []
        att_maps = []
        n_batches = word_embs.size(0)
        c_code, mu, logvar = self.ca_net(sent_emb)   # (N,c_dim), (N,c_dim), (N,c_dim)

        # priori
        kb_key, kb_value = priori  # (N,K,2048), (N,K,2048,7,7)

        # query a priori
        p_code, indices, weights = \
            self.hard_read(sent_emb, kb_key, kb_value)  # (N,2048,7,7)
        p_code = self.pri_block(p_code)  # (N,1024,4,4)

        if cfg.TREE.BRANCH_NUM > 0:  # 3
            h_code1 = self.h_net1(z_code, c_code, p_code)  # -> img_feat R0 (N,64,64,64)
            fake_img1 = self.img_net1(h_code1)  # (N,3,64,64)
            fake_imgs.append(fake_img1)
        
        # Dynamic Memory based Image Refinement
        if cfg.TREE.BRANCH_NUM > 1:
            # h_code2: (N,64,128,128)
            # att1:    (N,18,64,64)
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask, cap_lens)
            fake_img2 = self.img_net2(h_code2)  # (N,3,128,128)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)

        if cfg.TREE.BRANCH_NUM > 2:
            # h_code3: (N,64,256,256)
            # att2:    (N,18,128,128)
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask, cap_lens)
            fake_img3 = self.img_net3(h_code3)  # (N,3,256,256)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)
        
        return fake_imgs, att_maps, mu, logvar, indices, weights



class G_DCGAN(nn.Module):
    def __init__(self):
        super(G_DCGAN, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        # 16gf x 64 x 64 --> gf x 64 x 64 --> 3 x 64 x 64
        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
        self.img_net = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        if cfg.TREE.BRANCH_NUM > 0:
            h_code = self.h_net1(z_code, c_code)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code, att1 = self.h_net2(h_code, c_code, word_embs, mask)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code, att2 = self.h_net3(h_code, c_code, word_embs, mask)
            if att2 is not None:
                att_maps.append(att2)

        fake_imgs = self.img_net(h_code)
        return [fake_imgs], att_maps, mu, logvar


# ############## D networks ##########################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        SpectralNorm(conv3x3(in_planes, out_planes, bias=True)),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downscale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        SpectralNorm(nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=True)),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# Downscale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    layers = []
    layers.append(SpectralNorm(nn.Conv2d(3, ndf, 4, 2, 1, bias=True)))
    layers.append(nn.LeakyReLU(0.2, inplace=True),)
    layers.append(SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True)))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers.append(SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True)))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers.append(SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM  # 32
        nef = cfg.TEXT.EMBEDDING_DIM  # 256
        self.img_code_s16 = encode_image_by_16times(ndf)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code4 = self.img_code_s16(x_var)  # 4 x 4 x 8df
        return x_code4


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        #
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code8 = self.img_code_s16(x_var)   # 8 x 8 x 8df
        x_code4 = self.img_code_s32(x_code8)   # 4 x 4 x 16df
        x_code4 = self.img_code_s32_1(x_code4)  # 4 x 4 x 8df
        return x_code4


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code_s16(x_var)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)
        return x_code4
