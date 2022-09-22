
Faces folder contains 23705 aligned and cropped images of shape [200 x 200 x 3] i.e. [H X W X C].
(Height, Width, Channel)

Three images with a missing label on race have been removed from the original dataset. 

The images are sorted according to filenames (by string). The images have been renamed 0 - 23704 to align sorting in different programming languages. Original filenames are available in 'filenames.txt'. 

Two additional files are available

* labels.csv
* filenames.txt

'labes.csv' contains the labels of the images in the same order as the images in the 'Faces' folder. 
These labels are given in comma separated format as:

age,gender,race

'filenames.txt' contains the original filenames of the images in the same order as the images and the labels.