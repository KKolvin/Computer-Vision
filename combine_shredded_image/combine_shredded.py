from PIL import Image
import os, glob
import numpy as np
from IPython.display import display
import math

def load_images_from_folder(folder):
    # first step is to get the list of all the files in the folder
    image_filenames = glob.glob(os.path.join(folder, '*.png'))
    # Now you should load each image into memory as a 
    # numpy array 
    images = []
    # ADD CODE HERE
    for name in image_filenames:
        if "simple_larry-roberts.png" not in name:
            img = Image.open(name)
            images.append(np.asarray(img))
    
    return images

images = load_images_from_folder('shredded-image')
simple_combined = Image.fromarray(np.hstack(images), 'RGB')
display(simple_combined)



# We'll begin by computing similarities between all image pairs
similarities = np.zeros((len(images), len(images)))
for i, ith_image in enumerate(images):
  for j, jth_image in enumerate(images):
    # Now we'll compute similarity by taking the right-most
    # column of the ith image, and the left-most column of the
    # jth image
    diff = np.sum((ith_image[:,-1]-jth_image[:,0])**2)
    similarities[i, j] = math.sqrt(diff)*(-1)# ADD CODE HERE


def greedy_merge(strips, compatibility):
  # ok, we want to merge images in order of compatibility
  # so we can begin by flattening the compatibility
  # array and then using argsort to get the index
  # of the most compatibile strips

  ordering = np.argsort(np.reshape(compatibility, -1))
  ordering = np.flipud(ordering)

  # Now that we have our ordering, we need to keep track of
  # strips so we only select them once.  Let's keep track of 
  # them in the "used_strips" variable
  used_strips = set()

  # Now we should iterate through our ordering and add
  # the most compatible strips until we have a single image
  merged_strips = [] # final image
  merged_left = None # left-most merged strip index
  merged_right = None # right-most merged strip index

  # we'll keep this going until all strips are used
  while len(used_strips) != len(strips):
    # we should always add at least one strip, so let's make sure
    num_used_start = len(used_strips)

    for next_item in ordering:
        # first we get its row and column index
        left_strip = next_item // len(strips)
        right_strip = next_item % len(strips)

        # skip if t hey're the same strip
        if left_strip == right_strip:
            continue

        # base case, no merged strips yet
        if merged_left is None:
            merged_strips = np.hstack((strips[left_strip], strips[right_strip]))
            merged_right = right_strip
            merged_left = left_strip
            used_strips.add(merged_right)
            used_strips.add(merged_left)
            continue

        # Check if you can add this to the left of merged_strips and merge it if
        # so. If you merge, you should update merged_left, used_strips,
        # then break out of the loop.
        if left_strip not in used_strips:
          # ADD CODE HERE
            if merged_left == right_strip:
                merged_strips = np.hstack((strips[left_strip], merged_strips))
                used_strips.add(left_strip)
                merged_left = left_strip
                break

        # Check if you can add this to the left of merged_strips and merge it if
        # so. If you merge, you should update merged_right, used_strips,
        # then break out of the loop.
        if right_strip not in used_strips:
          # ADD CODE HERE
            if merged_right == left_strip:
                merged_strips = np.hstack((merged_strips, strips[right_strip]))
                used_strips.add(right_strip)
                merged_right = right_strip
                break

    assert num_used_start != len(used_strips)
  
  return merged_strips

ssd_images = greedy_merge(images, similarities)
ssd_combined = Image.fromarray(ssd_images, 'RGB')
display(ssd_combined)