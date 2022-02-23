import numpy as np
import cv2, skimage
from scipy.spatial import distance
from PIL import Image
from IPython.display import display
import scipy
import math
import functions


def get_best_matches(img1, img2, num_matches):
    kp1, des1 = functions.get_sift_data(img1)
    kp2, des2 = functions.get_sift_data(img2)

    # Find distance between descriptors in images
    dist = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')

    # get the matches according to dist
    used = set()
    matches = []
    
    ordering = np.argsort(np.reshape(dist, -1))
    
    for pair in ordering:
        point1 = pair // len(dist[0])
        point2 = pair % len(dist[0])
        if (point1 not in used) and (point2 not in used):
            x1, y1 = kp1[point1]
            x2, y2 = kp2[point2]
            #matches.append([x1, y1, x2, y2])
            matches.append([x2, y2, x1, y1])
            used.add(point1)
            used.add(point2)
        if len(matches) >= (num_matches):
            break
    matches = np.array(matches)
    return matches


img1 = functions.imread('./stitch/left.jpg')
img2 = functions.imread('./stitch/right.jpg')
data = get_best_matches(img1, img2, 300)
fig, ax = plt.subplots(figsize=(20,10))
functions.plot_inlier_matches(ax, img1, img2, data)
#fig.savefig('sift_match.pdf', bbox_inches='tight')



def ransac(data, max_iters=5000, min_inliers=10):
    #ransac code to find the best model, inliers, and residuals
    sample = data[0:5]
    H = compute_homography(sample)
    thresh = 2
    
    iter = 0
    avg_res = 0
    match_inliers = []
    
    while iter < max_iters:
        iter += 1
        num_inliers = 0
        inliers = []
        residuals = 0
        
        for match in data:
            x1, y1 = match[2], match[3]
            x2, y2 = match[0], match[1]
            predict = H @ np.array([x2, y2, 1])
            x1_prime = predict[0]/predict[2]
            y1_prime = predict[1]/predict[2]
            dist = math.sqrt((x1_prime - x1)**2 + (y1_prime - y1)**2)
            if dist < thresh:
                num_inliers += 1
                inliers.append(match)
                residuals += dist
                
        if num_inliers > min_inliers:
            H = compute_homography(inliers)
            min_inliers = num_inliers
            avg_res = residuals / num_inliers
            match_inlers = np.array(inliers)
            
    print("Average residuals of inliers:", avg_res)
    return H, num_inliers, match_inlers
        

def compute_homography(matches):
    #compute homography according to the matches
    A = []
    for match in matches:
        x1, y1 = match[2], match[3]
        x2, y2 = match[0], match[1]
        A.append([0,0,0,x2,y2,1,-y1*x2,-y1*y2,-y1])
        A.append([x2,y2,1,0,0,0,-x1*x2,-x1*y2,-x1])
        
    U, S, V = np.linalg.svd(np.array(A))
    H = V[-1,:].reshape((3, 3))
    return H


def avg_residuals(matches, homography):
    dist = 0
    for match in matches:
        x1, y1, x2, y2 = match[1], match[2], match[0], match[1]
        predict = H @ np.array([x2, y2, 1])
        x1_prime, y1_prime = predict[0]/predict[2], predict[1]/predict[2]
        dist += math.sqrt((x1_prime - x1)**2 + (y1_prime - y1)**2)
    dist = dist / len(matches)
    return dist
    
    
# Report the details of the model including inliers, homography, average residuals
H, max_inliers, inliers = ransac(data)
print("Inliers:", max_inliers)
print("Homography:\n", H)

img1 = imread('./stitch/left.jpg')
img2 = imread('./stitch/right.jpg')
fig, ax = plt.subplots(figsize=(20,10))
plot_inlier_matches(ax, img1, img2, inliers)



def warp_images(img1, img2, H):
    # stich the images together using the homography
    dst = cv2.warpPerspective(img2, H, ((img1.shape[1] + img2.shape[1]), img1.shape[0]))
    dst[0:img1.shape[0], 0:img1.shape[1]] = img1
    
    return dst
    

# display the stitching results
img_warped = warp_images(img1, img2, H)
display(Image.fromarray(img_warped))