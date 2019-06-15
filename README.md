# GANtor-Arts-Center

Steps to set up directory:

1. Download wikiart dataset from https://drive.google.com/file/d/182-pFiKvXPB25DbTfAYjJ6gDE-ZCRXz0/view, extract files and store in a directory called 'data'.

2. 

3. To obtain pre-trained models for GANtor-v2:
    Download pre-trained generators and final discriminator from: 
    Unzip file and extract .pth files into the following directory path: GANtor-Arts-Center/src/saved_models/v2/s2_genre/
    Note: The pre-trained models contain both Stage-II generator weights at the designated epochs, as well as the Stage-I weights (frozen after 90 epochs of training). No need to load S-I weights separately.
    
4. To generate artwork: 
    
