**HistoSeg++: Delving deeper with attention and multiscale feature fusion for biomarker segmentation**
***12th International Conference on Biomedical and Bioinformatics Engineering (ICBBE 2025)***




### Setup Conda Environment
use this command to create a conda environment (all the required packages are listed in `histosegplusplus.yml` file)
```
conda env create -f histosegplusplus.yml
```


### Datasets

#### MoNuSeg - Multi-organ nuclei segmentation from H&E stained histopathological images.
link: https://monuseg.grand-challenge.org/Data/

#### TNBC - Triple-negative breast cancer.
link: https://zenodo.org/records/1175282#.YMisCTZKgow

#### DSB - 2018 Data Science Bowl.
link: https://www.kaggle.com/c/data-science-bowl-2018/data

#### EM - Electron Microscopy.
link: https://www.epfl.ch/labs/cvlab/data/data-em/

### Data Preprocessing
After downloading the dataset you must generate patches of images and their corresponding masks (Ground Truth), & convert it into numpy arrays or you can use dataloaders directly inside the code. Note: The last channel of masks must have black and white (0,1) values not greyscale(0 to 255) values. 
you can generate patches using Image_Patchyfy. Link : https://github.com/saadwazir/Image_Patchyfy

### Offline Data Augmentation
(it requires albumentations library link: https://albumentations.ai)

use `offline_augmentation.py` to generate augmented samples


## Training and Testing

For training and testing use the train-test.ipynb file and update the paths

train_image_dir = ""
train_mask_dir = ""

These are for training images and training masks. The code processes images and masks directly from folders using RGB images and greyscale masks

For testing at patch level

test_images_arg = ""
test_masks_arg = ""

At this level the code accepts arrays so you first need to create arrays from images and masks

For testing at full scale level

image_full_test_directory = ""
mask_full_test_directory = ""

At this level the code accepts full size images and masks directly from disk
