# Image Inpainting with Masked Self-Attention and Semantic Loss Function
Color inconsistencies are still a problem in image inpainting tasks. To deal with them, we present an attention mechanism named Masked Self-Attention, which takes an alternative approach to deal with masked inputs. In addition to that, we adopt contextual attention idea for our attention module. We also propose a new loss function called Semantic Attention Loss, which is used to establish connection between semantic labels and attention scores.
[Paper](https://github.com/hakansivuk/Image-Inpainting-with-Masked-Self-Attention-and-Semantic-Loss-Function/blob/master/Paper.pdf)


## Installation

Clone this repo.


This code requires PyTorch, python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```


## Dataset Preparation

This code uses [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) and [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset. The prepared dataset can be directly downloaded [here](https://drive.google.com/file/d/1TKhN9kDvJEcpbIarwsd1_fsTR2vGx6LC/view?usp=sharing). After unzipping, put the entire CelebA-HQ folder in the datasets folder. The complete directory should look like `./datasets/CelebA-HQ/train/` and `./datasets/CelebA-HQ/test/`.


## Generating Images Using Pretrained Models

Once the dataset is prepared, the reconstruction results be got using pretrained models.


1. Create `./checkpoints/optim` in the main folder and download the pretrained model from the [Google Drive Folder](https://drive.google.com/file/d/1k0m8-QErsDyaMVAOzwjgOlDsEpLhxJDo/view?usp=sharing). Save the file in `./checkpoints/optim`.


2. Generate the reconstruction results using the pretrained model.
	```bash
   python iter.py --name optim --load_size 128 --crop_size 128 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --batchSize 8 --gpu_ids 3 --iter 0
    ```

Here --iter controls number of iterative optimizations at test time. In the paper we used --iter 20. Note that iterative optimization takes
approximately 4 hours for all evaluation images.  

3. The reconstruction images are saved at `./results/optim/` and the corresponding style codes are stored at `./styles_test/style_codes/`.


## Training New Models

To train the new model, you need to specify the option `--dataset_mode custom`, along with `--label_dir [path_to_labels] --image_dir [path_to_images]`. You also need to specify options such as `--label_nc` for the number of label classes in the dataset, and `--no_instance` to denote the dataset doesn't have instance maps.


```bash
python train.py --name new_experiment --load_size 128 --crop_size 128 --dataset_mode custom --label_dir datasets/CelebA-HQ/train/labels --image_dir datasets/CelebA-HQ/train/images --label_nc 19 --no_instance --batchSize 12 --gpu_ids 0,1,2,3
```

This work highly utilizes SEAN repository. 

## Changes to Existing Network

Attention architectures are added to models/networks/attention.py \
New loss function is implemented in models/pix2pix_model.py:179-202 \
Attention visualizations are implemented in util/util.py:130-151 \
Mask generation code is added to data/pix2pix_dataset.py:112-217 \
The repository did not include any evaluation metrics. Therefore, we added SSIM in metrics/ssim.py and PSNR in metrics/psnr.py. \
Iterative optimization is implemented in iter.py \
General changes to network architecture so that it supports masked input together with attention modules. \
General changes to visualization so that we can visualize masks and attentions.  
