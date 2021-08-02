from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from ipdb import set_trace
from collections import OrderedDict

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params

from datasets import prepare_data
from model_base import RNN_ENCODER, CNN_ENCODER
from model_hard import G_DCGAN, G_NET  
from model_hard import D_NET64, D_NET128, D_NET256

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys
from miscc.logger import Logger

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, dataset, ix2vec):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = output_dir
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        #torch.cuda.set_device(cfg.GPU_ID)
        #cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.ix2vec = ix2vec
        self.data_loader = data_loader
        self.dataset = dataset
        self.num_batches = len(self.data_loader)

    def build_models(self):
        def count_parameters(model):
            total_param = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    num_param = np.prod(param.size())
                    if param.dim() > 1:
                        print(name, ':', 'x'.join(str(x) for x in list(param.size())), 
                              '=', num_param)
                    else:
                        print(name, ':', num_param)
                    total_param += num_param
            return total_param

        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        # self.load_model_on_multi_gpus(image_encoder, img_encoder_path)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = RNN_ENCODER(self.ix2vec, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        # self.load_model_on_multi_gpus(text_encoder, cfg.TRAIN.NET_E)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            netG = G_DCGAN()
            if cfg.TREE.BRANCH_NUM ==1:
                netsD = [D_NET64(b_jcu=False)]
            elif cfg.TREE.BRANCH_NUM == 2:
                netsD = [D_NET128(b_jcu=False)]
            else:  # cfg.TREE.BRANCH_NUM == 3:
                netsD = [D_NET256(b_jcu=False)]
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            
            
        else:
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:

        print('number of trainable parameters =', count_parameters(netsD[-1]))
        print('number of trainable parameters =', count_parameters(netG))

        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            # netG.load_state_dict(state_dict)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module.' in k:
                    new_state_dict[k[7:]] = v  # remove 'module.'
                else:
                    new_state_dict[k] = v
            netG.load_state_dict(new_state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)

            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    # netsD[i].load_state_dict(state_dict)
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict[k[7:]] = v  # remove `module.`
                    netsD[i].load_state_dict(new_state_dict)
        
        # ########################################################### #
        if cfg.CUDA:
            if len(cfg.GPU_ID) == 1:
                text_encoder = text_encoder.cuda()
                image_encoder = image_encoder.cuda()
                netG.cuda()
                for i in range(len(netsD)):
                    netsD[i].cuda()
            elif len(cfg.GPU_ID) > 1:
                # text_encoder = nn.DataParallel(text_encoder).cuda()
                # image_encoder = nn.DataParallel(image_encoder).cuda()
                text_encoder = text_encoder.cuda()
                image_encoder = image_encoder.cuda()
                netG = nn.DataParallel(netG).cuda()
                for i in range(len(netsD)):
                    netsD[i] = nn.DataParallel(netsD[i]).cuda()

        return [text_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(filter(lambda p: p.requires_grad, netsD[i].parameters()),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))  # (N,) = [0,1,...,N-1]
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    # def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
    #                      image_encoder, captions, cap_lens,
    #                      gen_iterations, real_image, name='current'):
    #     # Save images
    #     fake_imgs, attention_maps, _, _ = \
    #         netG(noise, sent_emb, words_embs, mask, cap_lens)
    #     for i in range(len(attention_maps)):
    #         if len(fake_imgs) > 1:
    #             img = fake_imgs[i + 1].detach().cpu()
    #             lr_img = fake_imgs[i].detach().cpu()
    #         else:
    #             img = fake_imgs[0].detach().cpu()
    #             lr_img = None
    #         attn_maps = attention_maps[i]
    #         att_sze = attn_maps.size(2)
    #         img_set, _ = \
    #             build_super_images(img, captions, self.ixtoword,
    #                                attn_maps, att_sze, lr_imgs=lr_img)
    #         if img_set is not None:
    #             im = Image.fromarray(img_set)
    #             fullpath = '%s/G_%s_%d_%d.png'% (self.image_dir, name, gen_iterations, i)
    #             im.save(fullpath)

    #     # for i in range(len(netsD)):
    #     i = -1
    #     img = fake_imgs[i].detach()
    #     region_features, _ = image_encoder(img)
    #     att_sze = region_features.size(2)
    #     _, _, att_maps = words_loss(region_features.detach(),
    #                                 words_embs.detach(),
    #                                 None, cap_lens,
    #                                 None, self.batch_size)
    #     img_set, _ = \
    #         build_super_images(fake_imgs[i].detach().cpu(),
    #                            captions, self.ixtoword, att_maps, att_sze)
    #     if img_set is not None:
    #         im = Image.fromarray(img_set)
    #         fullpath = '%s/D_%s_%d.png'\
    #             % (self.image_dir, name, gen_iterations)
    #         im.save(fullpath)
    #     #print(real_image.type)


    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM  # 100
        noise = Variable(torch.FloatTensor(batch_size, nz))  # (N,100)
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))  # (N,100)
        
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        logger = Logger(self.log_dir)
        weights = None
        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)  # __getitem__
            step = 0
            while step < self.num_batches:

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()  # __getitem__
                # imgs: [(N,3,64,64), (N,3,128,128), (N,3,256,256)]
                # captions: (N,seq_len)
                # kb_candi: (K,N)
                imgs, captions, cap_lens, class_ids, keys, \
                priors, kb_candi = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len  # (N,256,18)
                # sent_emb: batch_size x nef  # (N,256)
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)  # (N,18)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                
                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)  # (N,100)
                # imgs:   [(N,3,64,64), (N,3,128,128), (N,3,256,256)]
                # mu:     (N,100)
                # logvar: (N,100)
                fake_imgs, _, mu, logvar, indices, weights = \
                    netG(noise, sent_emb, words_embs, mask, cap_lens, priors)
                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                log_info_D = {}
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD, log, real_acc, fake_acc = \
                        discriminator_loss(netsD[i], imgs[i], fake_imgs[i], sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())
                    D_logs += log

                    log_info_D['errD%d' % i] = '%.4f' % errD.item()
                    log_info_D['real_acc%d' % i] = '%.4f' % real_acc
                    log_info_D['fake_acc%d' % i] = '%.4f' % fake_acc
                log_info_D['D_loss'] = '%.4f' % errD_total
                
                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs, log_info_G = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                log_info_G['kl_loss'] = '%.4f' % kl_loss.item()
                log_info_G['G_loss'] = '%.4f' % errG_total.item()

                for p, avg_p in zip(netG.parameters(), avg_param_G):  # update avg_param_G
                    avg_p.mul_(0.999).add_(0.001, p.data)  # avg_p = 0.999*avg_p + 0.001*p.data
                
                if gen_iterations % 100 == 0:
                    print('Epoch [{}/{}] Step [{}/{}]'.format(
                        epoch, self.max_epoch, step,
                        self.num_batches) + ' ' + D_logs + ' ' + G_logs)
                    
                    for tag, value in log_info_D.items():
                        logger.scalar_summary(tag, float(value), gen_iterations)
                    
                    for tag, value in log_info_G.items():
                        logger.scalar_summary(tag, float(value), gen_iterations)
                    if weights is not None:
                        print(weights[0].data.cpu().numpy())
                    # for item in weights:
                    #     print(item[0].data.cpu().numpy())

                # # save images
                # if gen_iterations % 2000 == 0:  # 10000
                #     backup_para = copy_G_params(netG)
                #     load_params(netG, avg_param_G)
                #     #self.save_img_results(netG, fixed_noise, sent_emb, words_embs, mask, image_encoder,
                #     #                      captions, cap_lens, epoch, imgs[-1], name='average')
                #     load_params(netG, backup_para)
                #     #
                #     # self.save_img_results(netG, fixed_noise, sent_emb,
                #     #                       words_embs, mask, image_encoder,
                #     #                       captions, cap_lens,
                #     #                       epoch, name='current')
                
            end_t = time.time()

            print('''[%d/%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs''' % (
                epoch, self.max_epoch, errD_total.item(), errG_total.item(), end_t - start_t))
            print('-' * 89)
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    # def save_singleimages(self, images, filenames, save_dir,
    #                       split_dir, sentenceID=0):
    #     for i in range(images.size(0)):
    #         s_tmp = '%s/single_samples/%s/%s' %\
    #             (save_dir, split_dir, filenames[i])
    #         folder = s_tmp[:s_tmp.rfind('/')]
    #         if not os.path.isdir(folder):
    #             print('Make a new folder: ', folder)
    #             mkdir_p(folder)

    #         fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
    #         # range from [-1, 1] to [0, 1]
    #         # img = (images[i] + 1.0) / 2
    #         img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
    #         # range from [0, 1] to [0, 255]
    #         ndarr = img.permute(1, 2, 0).data.cpu().numpy()
    #         im = Image.fromarray(ndarr)
    #         im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()

            # load text encoder
            text_encoder = RNN_ENCODER(self.ix2vec, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            #load image encoder
            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            state_dict = \
                torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', img_encoder_path)
            image_encoder = image_encoder.cuda()
            image_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            # netG.load_state_dict(state_dict)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' in k:
                    new_state_dict[k[7:]] = v  # remove `module.`
                else:
                    new_state_dict[k] = v
            netG.load_state_dict(new_state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]  # ../netG_epoch_36
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            R_count = 0
            R = np.zeros(30000)
            cont = True
            # temp_out = set()
            for ii in range(11):  # 11, cfg.TEXT.CAPTIONS_PER_IMAGE
                if (cont == False):
                    break
                for step, data in enumerate(self.data_loader, start=0):
                    cnt += batch_size
                    if (cont == False):
                        break
                    if step % 100 == 0:
                       print('cnt: ', cnt)
                    # if step > 50:
                    #     break

                    imgs, captions, cap_lens, class_ids, keys, \
                    priors, kb_candi = prepare_data(data)
                    
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    # set_trace()
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _, indices, weights = \
                        netG(noise, sent_emb, words_embs, mask, cap_lens, priors)
                    
                    for j in range(batch_size): 
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            #print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii)
                        im.save(fullpath)

                    if cnt >= 30000:  # 30000
                        cont = False

    # def gen_example(self, data_dic):
    #     if cfg.TRAIN.NET_G == '':
    #         print('Error: the path for morels is not found!')
    #     else:
    #         # Build and load the generator
    #         text_encoder = \
    #             RNN_ENCODER(self.ix2vec, nhidden=cfg.TEXT.EMBEDDING_DIM)
    #         state_dict = \
    #             torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    #         text_encoder.load_state_dict(state_dict)
    #         print('Load text encoder from:', cfg.TRAIN.NET_E)
    #         text_encoder = text_encoder.cuda()
    #         text_encoder.eval()

    #         # the path to save generated images
    #         if cfg.GAN.B_DCGAN:
    #             netG = G_DCGAN()
    #         else:
    #             netG = G_NET()
    #         s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
    #         model_dir = cfg.TRAIN.NET_G
    #         state_dict = \
    #             torch.load(model_dir, map_location=lambda storage, loc: storage)
    #         netG.load_state_dict(state_dict)
    #         print('Load G from: ', model_dir)
    #         netG.cuda()
    #         netG.eval()
    #         for key in data_dic:
    #             save_dir = '%s/%s' % (s_tmp, key)
    #             mkdir_p(save_dir)
    #             captions, cap_lens, sorted_indices = data_dic[key]

    #             batch_size = captions.shape[0]
    #             nz = cfg.GAN.Z_DIM
    #             captions = Variable(torch.from_numpy(captions), volatile=True)
    #             cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

    #             captions = captions.cuda()
    #             cap_lens = cap_lens.cuda()
    #             for i in range(1):  # 16
    #                 noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
    #                 noise = noise.cuda()
    #                 #######################################################
    #                 # (1) Extract text embeddings
    #                 ######################################################
    #                 hidden = text_encoder.init_hidden(batch_size)
    #                 # words_embs: batch_size x nef x seq_len
    #                 # sent_emb: batch_size x nef
    #                 words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    #                 mask = (captions == 0)
    #                 #######################################################
    #                 # (2) Generate fake images
    #                 ######################################################
    #                 noise.data.normal_(0, 1)
    #                 fake_imgs, attention_maps, _, _ = \
    #                     netG(noise, sent_emb, words_embs, mask, cap_lens)
    #                 # G attention
    #                 cap_lens_np = cap_lens.cpu().data.numpy()
    #                 for j in range(batch_size):
    #                     save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
    #                     for k in range(len(fake_imgs)):
    #                         im = fake_imgs[k][j].data.cpu().numpy()
    #                         im = (im + 1.0) * 127.5
    #                         im = im.astype(np.uint8)
    #                         # print('im', im.shape)
    #                         im = np.transpose(im, (1, 2, 0))
    #                         # print('im', im.shape)
    #                         im = Image.fromarray(im)
    #                         fullpath = '%s_g%d.png' % (save_name, k)
    #                         im.save(fullpath)

    #                     for k in range(len(attention_maps)):
    #                         if len(fake_imgs) > 1:
    #                             im = fake_imgs[k + 1].detach().cpu()
    #                         else:
    #                             im = fake_imgs[0].detach().cpu()
    #                         attn_maps = attention_maps[k]
    #                         att_sze = attn_maps.size(2)
    #                         img_set, sentences = \
    #                             build_super_images2(im[j].unsqueeze(0),
    #                                                 captions[j].unsqueeze(0),
    #                                                 [cap_lens_np[j]], self.ixtoword,
    #                                                 [attn_maps[j]], att_sze)
    #                         if img_set is not None:
    #                             im = Image.fromarray(img_set)
    #                             fullpath = '%s_a%d.png' % (save_name, k)
    #                             im.save(fullpath)
