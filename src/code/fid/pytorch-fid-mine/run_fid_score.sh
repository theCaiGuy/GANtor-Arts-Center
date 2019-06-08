#!/bin/bash

python fid_score.py /home/michaelcai/GANtor-Arts-Center/src/code/fid/Sorted_Images/abstract_painting/ /home/michaelcai/GANtor-Arts-Center/src/code/fid/Generated_Images/abstract_painting/ > abstract_fid_50.txt

python fid_score.py /home/michaelcai/GANtor-Arts-Center/src/code/fid/Sorted_Images/cityscape/ /home/michaelcai/GANtor-Arts-Center/src/code/fid/Generated_Images/cityscape/ > cityscape_fid_50.txt

python fid_score.py /home/michaelcai/GANtor-Arts-Center/src/code/fid/Sorted_Images/genre_painting/ /home/michaelcai/GANtor-Arts-Center/src/code/fid/Generated_Images/genre_painting/ > genre_painting_fid_50.txt

python fid_score.py /home/michaelcai/GANtor-Arts-Center/src/code/fid/Sorted_Images/illustration/ /home/michaelcai/GANtor-Arts-Center/src/code/fid/Generated_Images/illustration/ > illustration_fid_50.txt

python fid_score.py /home/michaelcai/GANtor-Arts-Center/src/code/fid/Sorted_Images/landscape/ /home/michaelcai/GANtor-Arts-Center/src/code/fid/Generated_Images/landscape/ > landscape_fid_50.txt

python fid_score.py /home/michaelcai/GANtor-Arts-Center/src/code/fid/Sorted_Images/nude_painting/ /home/michaelcai/GANtor-Arts-Center/src/code/fid/Generated_Images/nude_painting/ > nude_painting_fid_50.txt

python fid_score.py /home/michaelcai/GANtor-Arts-Center/src/code/fid/Sorted_Images/portrait/ /home/michaelcai/GANtor-Arts-Center/src/code/fid/Generated_Images/portrait/ > portrait_fid_50.txt

python fid_score.py /home/michaelcai/GANtor-Arts-Center/src/code/fid/Sorted_Images/religious_painting/ /home/michaelcai/GANtor-Arts-Center/src/code/fid/Generated_Images/religious_painting/ > religious_painting_fid_50.txt

python fid_score.py /home/michaelcai/GANtor-Arts-Center/src/code/fid/Sorted_Images/sketch_and_study/ /home/michaelcai/GANtor-Arts-Center/src/code/fid/Generated_Images/sketch_and_study/ > sketch_and_study_fid_50.txt

python fid_score.py /home/michaelcai/GANtor-Arts-Center/src/code/fid/Sorted_Images/still_life/ /home/michaelcai/GANtor-Arts-Center/src/code/fid/Generated_Images/still_life/ > still_life_fid_50.txt