# save all images open in GIMP
# to use in GIMP Pyhton console
# https://developer.gimp.org/api/2.0/libgimp/index.html

import os
import gimp

def save_open_images(dir, prefix_name):
    open_images = gimp.image_list()
    for idx, img in enumerate(open_images):
        path = os.path.join(dir, prefix_name + "_" + str(idx) + ".png")
        pdb.gimp_file_save(img, img.layers[0], path, path)

    
