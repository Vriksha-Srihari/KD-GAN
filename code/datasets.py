from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchtext.vocab as vocab

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from ipdb import set_trace
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def load_pkl(path):
    with open(path, 'rb') as f:
        kb = pickle.load(f)
    return kb


# def load_prior():
#     kb = load_pkl(cfg.KNOWLEDGE_BASE)
#     kb = kb[cfg.DATASET_NAME]  # {"ckass_id1": [img_name1, ..., img_name10]}
#     kb_keys = []
#     kb_values = []
#     kb_chosen = []

#     for k, v in kb.items():
#         for image in v:
#             kb_keys.append(np.load(os.path.join(cfg.KNOWLEDGE_FEAT, k, image[:-3]+'pool.npy')))
#             kb_values.append(np.load(os.path.join(cfg.KNOWLEDGE_FEAT, k, image[:-3]+'layer4.npy')))
#         kb_chosen.append(['%s/%s' % (k, image) for image in v])
#     if cfg.CUDA:
#         kb_keys = Variable(torch.from_numpy(np.array(kb_keys))).cuda()
#         kb_values = Variable(torch.from_numpy(np.array(kb_values))).cuda()
#     else:
#         kb_keys = Variable(torch.from_numpy(np.array(kb_keys)))
#         kb_values = Variable(torch.from_numpy(np.array(kb_values)))

#     priors = (kb_keys, kb_values)
    
#     return priors, kb_chosen


# def load_prior_batch(data, flag='rand'):
#     imgs, captions, captions_lens, class_ids, keys = data

#     # prepare knowledge from kb
#     kb = load_pkl(cfg.KNOWLEDGE_BASE)
#     if 'coco' in cfg.DATASET_NAME:
#         name2classid = \
#         load_pkl(os.path.join(cfg.DATA_DIR, 'pickles/imgName_to_classID.pkl'))
    
#     kb_keys = []
#     kb_values = []
#     kb_chosen = []
#     for i in range(len(keys)):
#         if 'coco' in cfg.DATASET_NAME:
#             class_name = list(name2classid[keys[i] + '.jpg'])[0]
#             choices = kb[cfg.DATASET_NAME][str(class_name)]
#             _names = []
#             for _ in choices:
#                 split = 'train2014' if 'train' in _ else 'val2014'
#                 _names.append(os.path.join(cfg.KNOWLEDGE_FEAT, split, _))
#         else:
#             class_name = os.path.split(keys[i])[0]
#             choices = kb[cfg.DATASET_NAME][class_name]
#             _names = [os.path.join(cfg.KNOWLEDGE_FEAT, class_name, _) for _ in choices]

#         if 'rand' in flag:
#             # random choose
#             rand_idx = random.randint(0, len(_names))
#             img_name = _names[rand_idx]
#             # print(class_name, img_name)

#             kb_value_feat = np.load(img_name[:-3] + 'layer4.npy')  # (2048,7,7)
#             kb_chosen.append('%s/%s' % (class_name, img_name))
        
#         elif 'sum' in flag:
#             # sum all choices
#             kb_value_feat = [np.load(_[:-3] + 'layer4.npy') for _ in _names]
#             kb_value_feat = np.sum(np.array(kb_value_feat), axis=0)
        
#         elif 'query_pool' in flag:
#             # all keys and values, 10 in total
#             # [(2048,1,1), ...]
#             kb_key_feat = [np.load(_[:-3] + 'pool.npy') for _ in _names]
#             kb_keys.append(kb_key_feat)

#             # [(2048,7,7), ...]  
#             kb_value_feat = [np.load(_[:-3] + 'layer4.npy') for _ in _names]
            
#             kb_chosen.append(['%s/%s' % (class_name, img_name) for img_name in choices])

#         kb_values.append(kb_value_feat)

#     kb_keys = np.squeeze(np.array(kb_keys))  # (N,K,2048)
#     kb_values = np.array(kb_values)          # (N,K,2048,7,7)
    
#     if cfg.CUDA:
#         kb_keys = Variable(torch.from_numpy(kb_keys)).cuda()
#         kb_values = Variable(torch.from_numpy(kb_values)).cuda()
#     else:
#         kb_keys = Variable(torch.from_numpy(kb_keys))
#         kb_values = Variable(torch.from_numpy(kb_values))

#     priors = (kb_keys, kb_values)
    
#     return priors, kb_chosen


def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys, \
    priors, kb_candi = data
    # imgs: [(N,3,64,64), (N,3,128,128), (N,3,256,256)]
    # captions: (N,seq_len)

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):  # 3
        imgs[i] = imgs[i][sorted_cap_indices]  #
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()  # sorted
    class_ids = class_ids[sorted_cap_indices].numpy()  # sorted
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]  # sorted
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
    
    kb_keys, kb_values = priors
    if cfg.CUDA:
            kb_keys = Variable(kb_keys).cuda()
            kb_values = Variable(kb_values).cuda()
    else:
        kb_keys = Variable(torch.kb_keys)
        kb_values = Variable(torch.kb_values)
    priors = [kb_keys, kb_values]

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys, priors, kb_candi]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)  # 64, 128
            else:
                re_img = img  # 256
            ret.append(normalize(re_img))

    return ret  # [(3,64,64), (3,128,128), (3,256,256)]


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE  # 10

        self.imsize = []  # [64,128,256]
        for i in range(cfg.TREE.BRANCH_NUM):  # 3
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir  # ../data/birds
        if data_dir.find('birds') != -1:  # 'birds' in data_dir
            # {'001.Black_footed_Albatross/Black_Footed_Albatross_0046_18': [[60, 27, 325, 304]], ...}
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        self.split = split
        split_dir = os.path.join(data_dir, split)  # ../data/birds/train_or_test
        
        self.glove = vocab.GloVe(name='6B', dim=cfg.TEXT.EMBEDDING_DIM)
        self.word2ix = self.glove.stoi
        self.ix2word = self.glove.itos
        self.ix2vec = self.glove.vectors
        print('Text embedding {}'.format(cfg.TEXT.EMBEDDING_DIM))
        print('Loaded {} words from GloVe'.format(len(self.ix2word)))

        # also update ix2word, word2ix, ix2vec
        self.filenames, self.captions, self.n_words = \
            self.load_text_data(data_dir, split)

        # np.range(length) for coco 
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

        # load knowledge base
        if 'rand' in cfg.CONFIG_NAME:
            if 'all' in cfg.CONFIG_NAME:
                # rand sample from all images
                # self.kb = load_pkl('%s/%s/filenames.pickle' % (cfg.DATA_DIR, 'train'))
                kb_path = '%s/%s/filenames.pickle' % (cfg.DATA_DIR, 'train')
            else:
                # rand sample from fixed random 10 images
                # self.kb = load_pkl(cfg.KNOWLEDGE_BASE.replace('.pickle', '_rand.pickle'))
                kb_path = cfg.KNOWLEDGE_BASE.replace('.pickle', '_rand.pickle')
        else:
            # self.kb = load_pkl(cfg.KNOWLEDGE_BASE)
            kb_path = cfg.KNOWLEDGE_BASE

        self.kb = load_pkl(kb_path)
        print('Load KB from: ', kb_path)
        
        if 'coco' in cfg.DATASET_NAME:
            self.name2classid = \
            load_pkl(os.path.join(cfg.DATA_DIR, 'pickles/imgName_to_classID.pkl'))
        else:
            self.name2classid = None

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()  # 11788 class_id/image_name.jpg
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap, cap_path)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def get_word_idx(self, word):
        try:
            ix = self.word2ix[word]
        except:
            ix = self.word2ix['unk']
        return ix

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        w_threshold = cfg.TEXT.WORD_THRESHOLD
        vocab = [w for w in word_counts if word_counts[w] >= w_threshold]
        vocab.append('<end>')

        first_s = self.ix2word[0]    # 'the'
        new_idx = len(self.ix2word)  # 400000
        self.ix2word.append(first_s) # 'the'
        self.ix2word[0] = u'<end>'
        
        self.word2ix[first_s] = new_idx
        self.word2ix[u'<end>'] = 0

        first_vec = self.ix2vec[0].unsqueeze(0)
        self.ix2vec = torch.cat([self.ix2vec, first_vec], dim=0)
        self.ix2vec[0] = torch.zeros([300])

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                rev.append(self.get_word_idx(w))
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                rev.append(self.get_word_idx(w))
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new, vocab]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        # [002.Laysan_Albatross/Laysan_Albatross_0002_1027, ...],           length=8855
        train_names = self.load_filenames(data_dir, 'train')
        # [001.Black_footed_Albatross/Black_Footed_Albatross_0046_18, ...], length=2933
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)
            train_captions, test_captions, vocab = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions, 
                             self.ix2word, self.word2ix, 
                             vocab], f, protocol=2)
                print('Save to: ', filepath)
            
            ix2vec_path = filepath.replace('captions.pickle', 'ix2vec.pickle')
            if not os.path.isfile(ix2vec_path):
                with open(ix2vec_path, 'wb') as f:
                    pickle.dump(self.ix2vec, f)
                    print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                self.ix2word, self.word2ix = x[2], x[3]
                vocab = x[4]
                del x
                print('Load from: ', filepath)
            
            ix2vec_path = filepath.replace('captions.pickle', 'ix2vec.pickle')
            with open(ix2vec_path, 'rb') as f:
                self.ix2vec = pickle.load(f)
                print('Load from: ', ix2vec_path)
        
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names

        return filenames, captions, len(vocab)

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')  # (18,1)
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)  # randomly remove words from raw sentence
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_knowledge(self, key):
        # get reference images by class
        _names = []
        if self.name2classid:
            cls_name = list(self.name2classid[key+'.jpg'])[0]
            choices = self.kb[cfg.DATASET_NAME][str(cls_name)]
            for _ in choices:
                split = 'train2014' if 'train' in _ else 'val2014'
                _names.append(os.path.join(cfg.KNOWLEDGE_FEAT, split, _))
        else:
            cls_name = os.path.split(key)[0]
            choices = self.kb[cfg.DATASET_NAME][cls_name]
            for _ in choices:
                _names.append(os.path.join(cfg.KNOWLEDGE_FEAT, cls_name, _))
        
        kb_keys = [np.load(_[:-3] + 'pool.npy') for _ in _names]
        kb_values = [np.load(_[:-3] + 'layer4.npy') for _ in _names]
        
        kb_keys = np.squeeze(np.array(kb_keys))  # (K,2048)      # (K,300)
        kb_values = np.array(kb_values)          # (K,2048,7,7)  # (K,300,17,17)

        priors = [kb_keys, kb_values]
        kb_candi = ['%s/%s' % (cls_name, img_name) for img_name in choices]

        return priors, kb_candi
    
    def get_knowledge_new(self, index):
        # get reference images by text
        
        if 'rand' in cfg.CONFIG_NAME and 'all' in cfg.CONFIG_NAME:
            rand_image = self.kb[random.randint(0, len(self.kb))]
            candidates = rand_image

            if 'inception' in cfg.KNOWLEDGE_FEAT:
                kb_path = os.path.join(cfg.KNOWLEDGE_FEAT, rand_image)
                kb_global = np.load(kb_path + '.300g.npy')  # (K,300)
                kb_region = np.load(kb_path + '.300l.npy')  # (K,300,17,17)
            elif 'resnet' in cfg.KNOWLEDGE_FEAT:
                if 'bird' in cfg.DATASET_NAME:
                    kb_path = os.path.join(cfg.KNOWLEDGE_FEAT, rand_image)
                    kb_global = np.load(kb_path + '.pool.npy')    # (K,2048)
                    kb_region = np.load(kb_path + '.layer4.npy')  # (K,2048,7,7)
                else:
                    if 'train' in rand_image:
                        kb_path = os.path.join(cfg.KNOWLEDGE_FEAT, 'train2014', rand_image)
                    else:
                        kb_path = os.path.join(cfg.KNOWLEDGE_FEAT, 'val2014', rand_image)
                    kb_global = np.load(kb_path + '.pool.npy')    # (K,2048)
                    kb_region = np.load(kb_path + '.layer4.npy')  # (K,2048,7,7)

        else:
	        kbase = self.kb[index]  # [[top1_qid, top1_sim, filename], ..., [top10_.. ]]
	        candidates = [_[2] for _ in kbase]

	        if 'inception' in cfg.KNOWLEDGE_FEAT:
	            _names = [os.path.join(cfg.KNOWLEDGE_FEAT, _[2]) for _ in kbase]
	            kb_global = [np.load(_ + '.300g.npy') for _ in _names]  # (K,300)
	            kb_region = [np.load(_ + '.300l.npy') for _ in _names]  # (K,300,17,17)
	        elif 'resnet' in cfg.KNOWLEDGE_FEAT:
	            if 'bird' in cfg.DATASET_NAME:
	                _names = [os.path.join(cfg.KNOWLEDGE_FEAT, _[2]) for _ in kbase]  # 'class_id/filename'
	                kb_global = [np.load(_ + '.pool.npy') for _ in _names]    # (K,2048)
	                kb_region = [np.load(_ + '.layer4.npy') for _ in _names]  # (K,2048,7,7)
	            else:
	                _names = []
	                for _ in kbase:
	                    filename = _[2]
	                    if 'train' in filename:
	                        _names.append(os.path.join(cfg.KNOWLEDGE_FEAT, 'train2014', filename))
	                    else:
	                        _names.append(os.path.join(cfg.KNOWLEDGE_FEAT, 'val2014', filename))
	                kb_global = [np.load(_ + '.pool.npy') for _ in _names]    # (K,2048)
	                kb_region = [np.load(_ + '.layer4.npy') for _ in _names]  # (K,2048,7,7)

        kb_global = np.squeeze(np.array(kb_global))  # (K,2048)
        kb_region = np.array(kb_region)              # (K,2048,7,7)

        priors = [kb_global, kb_region]
        
        return priors, candidates

    def __getitem__(self, index):
        if 'train' in self.split:
            # sent
            cap, cap_len = self.get_caption(index)
            # img
            img_ix = index // cfg.TEXT.CAPTIONS_PER_IMAGE
            key = self.filenames[img_ix]
            cls_id = self.class_id[img_ix]
            q_idx = index
        else:
            # img
            key = self.filenames[index]
            cls_id = self.class_id[index]

            # random select a sentence
            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = index * self.embeddings_num + sent_ix
            cap, cap_len = self.get_caption(new_sent_ix)
            q_idx = new_sent_ix

        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        
        # priors, candidates = self.get_knowledge(key)
        priors, candidates = self.get_knowledge_new(q_idx)

        return imgs, cap, cap_len, cls_id, key, priors, candidates


    def __len__(self):
        if 'train' in self.split:
            return len(self.captions)
        else:
            return len(self.filenames)
