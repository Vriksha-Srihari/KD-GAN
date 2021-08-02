# *KD-GAN*

### Introduction

**Title:** "Knowledge-Driven Generative Adversarial Network for Text-to-Image Synthesis"


### How to use

**Python**

- Python2.7
- Pytorch0.4 (`conda install pytorch=0.4.1 cuda90 torchvision=0.2.1 -c pytorch`)
- tensorflow (`pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp27-none-linux_x86_64.whl`)
- `pip install easydict pathlib`
- `conda install requests nltk pandas scikit-image pyyaml cudatoolkit=9.0`


**Data**
1. Download metadata for [birds](https://drive.google.com) [coco](https://drive.google.com) and save them to `data/`

2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
    - `cd data/birds`
    - `wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz`
    - `tar -xvzf CUB_200_2011.tgz`
    
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`
    - `cd data/coco`
    - `wget http://images.cocodataset.org/zips/train2014.zip`
    - `wget http://images.cocodataset.org/zips/val2014.zip`
    - `unzip train2014.zip`
    - `unzip val2014.zip`
    - `mv train2014 images`
    - `cp val2014/* images`

**Pretrained Models**
- [IS for bird](https://drive.google.com/file/d/0B3y_msrWZaXLMzNMNWhWdW0zVWs)
    - `python google_drive.py 0B3y_msrWZaXLMzNMNWhWdW0zVWs eval/IS/bird/inception_finetuned_models.zip`
- [FID for bird](https://drive.google.com/file/d/1747il5vnY2zNkmQ1x_8hySx537ZAJEtj)
    - `python google_drive.py 1747il5vnY2zNkmQ1x_8hySx537ZAJEtj eval/FID/bird_val.npz`
- [FID for coco](https://drive.google.com/file/d/10NYi4XU3_bLjPEAg5KQal-l8A_d8lnL5)
    - `python google_drive.py 10NYi4XU3_bLjPEAg5KQal-l8A_d8lnL5 eval/FID/coco_val.npz`

**Training**
- go into `code/` folder
- bird: `python main.py --cfg cfg/bird_KDGAN.yml --gpu 0,1`
- coco: `python main.py --cfg cfg/coco_KDGAN.yml --gpu 0,1`

**Validation**
1. Images generation:
    - go into `code/` folder  
    - `python main.py --cfg cfg/eval_bird.yml --gpu 0`
    - `python main.py --cfg cfg/eval_coco.yml --gpu 0`
2. Inception score:
    - go into `eval/IS/bird` folder
    - `python inception_score_bird.py --image_folder ../../../models/bird_KDGAN_hard`
    - or go into `eval/IS/coco` folder
    - `python inception_score_coco.py ../../../models/coco_KDGAN_hard`
3. FID:
    - go into `eval/FID/` folder
    - `python fid_score.py --gpu 0 --batch-size 50 --path1 bird_val.npz --path2 ../../models/bird_KDGAN_hard`
    - `python fid_score.py --gpu 0 --batch-size 50 --path1 coco_val.npz --path2 ../../models/coco_KDGAN_hard`

### Reference
DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis [code](https://github.com/MinfengZhu/DM-GAN)
### License
This code is released under the MIT License (refer to the LICENSE file for details). 
