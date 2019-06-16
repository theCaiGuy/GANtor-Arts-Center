# GANtor-Arts-Center

The "GANtor Arts Center" paper can be found in the "cs231n_submissions" folder. Also available in the same folder is a presentation poster summarizing our work.

Authors: Avoy Datta (BSc. Stanford EE, 2020) and Michael Cai (BSc. Stanford CS, 2020). 

Major dependencies:
1. CUDA-compatible GPUs for training
2. pytorch version: 1.0.1.post2
3. torchvison version: 0.2.2

Steps to set up project:

1. Download wikiart dataset from https://drive.google.com/file/d/182-pFiKvXPB25DbTfAYjJ6gDE-ZCRXz0/view, extract files and store in a directory called 'data'.

2. To obtain pre-trained models for GANtor-v2:
    Download pre-trained Stage-1 generators and final discriminator from: https://drive.google.com/a/stanford.edu/file/d/1rxtRLUeU8mVEkxIf-r2dzUI3PYmd0Gp_/view?usp=sharing
    Unzip file and extract .pth files into the following directory path: GANtor-Arts-Center/src/saved_models/v2/s1_genre/
    
    Download pre-trained Stage-2 generators and final discriminator from: https://drive.google.com/a/stanford.edu/file/d/13v8lOt9d6gkVFnPr2fQzuK4Xcdbi-zkA/view?usp=sharing
    Unzip file and extract .pth files into the following directory path: GANtor-Arts-Center/src/saved_models/v2/s2_genre/
    Note: The pre-trained models contain both Stage-II generator weights at the designated epochs, as well as the Stage-I weights (frozen after 90 epochs of training). No need to load S-I weights separately if you only want to generate full 256x256 results.
    
3. To obtain Inception-v3 fine-tuned on wikiart dataset for 15 epochs:
    Download from https://drive.google.com/open?id=1dQaPg2QXOwDv0ocChZ_kgYIsAOfBtQTj
    Extract .pth files into following directory path: GANtor-Arts-Center/src/code/inception/ft_wikiart/ft_genre_15eps.pth
    
4. To generate 256x256 images from GANtor-v2:
    First ensure you have the following directory path set up: "GANtor-Arts-Center/src/code/inception/v2_generated/" and 10 folders labeled 0, 1, ..., 9 set up inside this directory   
    Open  GANtor-Arts-Center/src/code/inception/Generate Images v2.ipynb and make sure category = "genre" and stage = 2. 
    The best stage 2 generator checkpoint was after 50 epochs. This gave the lowest FID scores overall across all genres. Feel free to change this to load any of the other saved pretrained models instead.
    Run cells to load pre-trained models and generate images into the "v2_generated" folder. Each subfolder represents a genre class, as described in the "GANtor Arts Center" paper. 

5. To modify the v2 code (the most recent version), edit the .py files in GANtor-Arts-Center/src/code/ that are suffixed "v2".

Contact co-author at avoy.datta@stanford.edu for further queries.
