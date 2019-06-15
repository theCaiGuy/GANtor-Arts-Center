# GANtor-Arts-Center

Steps to set up directory:

1. Download wikiart dataset from https://drive.google.com/file/d/182-pFiKvXPB25DbTfAYjJ6gDE-ZCRXz0/view, extract files and store in a directory called 'data'.

2. To obtain pre-trained models for GANtor-v2:
    Download pre-trained generators and final discriminator from: 
    Unzip file and extract .pth files into the following directory path: GANtor-Arts-Center/src/saved_models/v2/s2_genre/
    Note: The pre-trained models contain both Stage-II generator weights at the designated epochs, as well as the Stage-I weights (frozen after 90 epochs of training). No need to load S-I weights separately.
    
3. To obtain Inception-v3 fine-tuned on wikiart dataset for 15 epochs:
    Download from 
    Extract .pth files into following directory path: GANtor-Arts-Center/src/code/inception/ft_wikiart/ft_genre_15eps.pth
    
4. To generate
Contact co-author at avoy.datta@stanford.edu for further queries.
