import sys, os
from PIL import Image

"""
    Use this to resize images for better cnn performance... This may affect
    detail in the photos.

    Provide a directory containing images (sub directories will be resized also)
    and all images and subdirectories with images will be resized
"""

def Resize(img_dir, output_dir, new_length, new_width):
    print(img_dir + " " + output_dir)
    print(os.listdir(img_dir))

    if not os.path.isdir(output_dir): # make dir if needed
        os.mkdir(output_dir)

    for item in os.listdir(img_dir): # go through everything in directory
        if (".jpg" in item) or (".JPG" in item) or (".jpeg" in item) or (".JPEG" in item) or (".png" in item) or (".PNG" in item) or (".bmp" in item) or (".BMP" in item):
            resized_img = Image.open(os.path.join(img_dir,item)).resize((new_length, new_width)) # this is where the resizing happens
            resized_img.save(os.path.join(output_dir, item)) # save in new area
        elif os.path.isdir(os.path.join(img_dir, item)):
            Resize(os.path.join(img_dir, item), os.path.join(output_dir, item), new_length, new_width) # recursive call for recursive nature of directories



if __name__ == "__main__":
    img_dir = str(sys.argv[1])
    output_dir = str(sys.argv[2])
    new_length = int(sys.argv[3])
    new_width = int(sys.argv[4])

    Resize(img_dir, output_dir, new_length, new_width) # resize and save
