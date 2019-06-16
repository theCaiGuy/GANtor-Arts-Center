# GANtor-Arts-Center

Steps to set up directory:

1. Download wikiart dataset from https://drive.google.com/file/d/182-pFiKvXPB25DbTfAYjJ6gDE-ZCRXz0/view, extract files and store in a directory called 'data'.

2. To obtain pre-trained models for GANtor-v2:
    Download pre-trained generators and final discriminator from: 
    Unzip file and extract .pth files into the following directory path: GANtor-Arts-Center/src/saved_models/v2/s2_genre/
    Note: The pre-trained models contain both Stage-II generator weights at the designated epochs, as well as the Stage-I weights (frozen after 90 epochs of training). No need to load S-I weights separately.
    
3. To obtain Inception-v3 fine-tuned on wikiart dataset for 15 epochs:
    Download from https://drive.google.com/open?id=1dQaPg2QXOwDv0ocChZ_kgYIsAOfBtQTj
    Extract .pth files into following directory path: GANtor-Arts-Center/src/code/inception/ft_wikiart/ft_genre_15eps.pth
    
4. To generate 256x256 images from GANtor-v2:
    First ensure you have the following directory path set up: "GANtor-Arts-Center/src/code/inception/v2_generated/" and 10 folders labeled 0, 1, ..., 9 set up inside this directory   
    Open  GANtor-Arts-Center/src/code/inception/Generate Images.ipynb and set category = "genre" and stage = 2. Then run cells to load pre-trained models and generate images into the "v2_generated" folder. Each subfolder represents a genre class, as described in the "GANtor Arts Center" paper. 


Contact co-author at avoy.datta@stanford.edu for further queries.
