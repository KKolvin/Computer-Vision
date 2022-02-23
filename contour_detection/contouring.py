from PIL import Image
import numpy as np
import cv2, os
from scipy.io import loadmat
from scipy import signal
import evaluate_boundaries


N_THRESHOLDS = 99

def detect_edges(imlist, fn):
  images, edges = [], []
  for imname in imlist:
    I = cv2.imread(os.path.join('data', str(imname)+'.jpg'))
    images.append(I)

    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = I.astype(np.float32)/255.
    mag = fn(I)
    edges.append(mag)
  return images, edges

def evaluate(imlist, all_predictions):
  count_r_overall = np.zeros((N_THRESHOLDS,))
  sum_r_overall = np.zeros((N_THRESHOLDS,))
  count_p_overall = np.zeros((N_THRESHOLDS,))
  sum_p_overall = np.zeros((N_THRESHOLDS,))
  for imname, predictions in zip(imlist, all_predictions):
    gt = loadmat(os.path.join('data', str(imname)+'.mat'))['groundTruth']
    num_gts = gt.shape[1]
    gt = [gt[0,i]['Boundaries'][0,0] for i in range(num_gts)]
    count_r, sum_r, count_p, sum_p, used_thresholds = \
              evaluate_boundaries.evaluate_boundaries_fast(predictions, gt, 
                                                           thresholds=N_THRESHOLDS,
                                                           apply_thinning=True)
    count_r_overall += count_r
    sum_r_overall += sum_r
    count_p_overall += count_p
    sum_p_overall += sum_p

  rec_overall, prec_overall, f1_overall = evaluate_boundaries.compute_rec_prec_f1(
        count_r_overall, sum_r_overall, count_p_overall, sum_p_overall)
  
  return max(f1_overall)

def compute_edges_dxdy(I):
  """Returns the norm of dx and dy as the edge response function."""
  
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same')
  mag = np.sqrt(dx**2 + dy**2)
  mag = normalize(mag)
  return mag

def normalize(mag):
  mag = mag / 1.5
  mag = mag * 255.
  mag = np.clip(mag, 0, 255)
  mag = mag.astype(np.uint8)
  return mag

imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score:', f1)



# try to remove artifacts on the images' boundaries
def compute_edges_dxdy_warmup(I):
  """Hint: Look at arguments for scipy.signal.convolve2d"""
  # ADD CODE HERE
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary="symm")
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary="symm")
  mag = np.sqrt(dx**2 + dy**2)
  mag = normalize(mag)
  return mag

imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy_warmup
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('After removing artifacts at the image boundaries, overall F1 score:', f1)


# smooth out the image
def compute_edges_dxdy_smoothing(I):
  """ Copy over response from the previous part and alter it
  to include this answer. See cv2.GaussianBlur"""
  # ADD CODE HERE 
  I = cv2.GaussianBlur(I,(5,5),0.5, borderType=cv2.BORDER_REPLICATE)
  # display(Image.fromarray(normalize(I)))
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary="symm")
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary="symm")
  mag = np.sqrt(dx**2 + dy**2)
  mag = normalize(mag)
  return mag

imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy_smoothing
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score after smoothing:', f1)



# Non-max suppression, we want the contour to be thiner
def compute_edges_dxdy_nonmax(I):
  """ Copy over response from the previous part and alter it to include this response"""
  # ADD CODE  HERE
  I = cv2.GaussianBlur(I,(5,5),0.5, borderType=cv2.BORDER_REPLICATE)
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary="symm")
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary="symm")
  mag = np.sqrt(dx**2 + dy**2)
  mag = normalize(mag)
  angle = abs(np.arctan2(dy,dx)*180/np.pi)
  #display(angle)
  thresh = 75.5
  for i in range(1, len(mag)-1):
    for j in range(1, len(mag[0])-1):
            if angle[i][j]<=22.5 or 157.5<=angle[i][j]<=202.5:
                if mag[i][j] < mag[i][j-1] or mag[i][j] < mag[i][j+1]:
                    if mag[i][j] > thresh:
                        mag[i][j] = 0
                    
            """if 22.5<angle[i][j]<=67.5 or 202.5<angle[i][j]<=247.5:
                if mag[i][j] < mag[i-1][j-1] or mag[i][j] < mag[i+1][j+1]:
                    if mag[i][j] > thresh:
                        mag[i][j] = 0"""
                
            if 67.5<angle[i][j]<=112.5 or 247.5<angle[i][j]<=292.5:
                if mag[i][j] < mag[i-1][j] or mag[i][j] < mag[i+1][j]:
                    if mag[i][j] > thresh:
                        mag[i][j] = 0
                
            """if 112.5<angle[i][j]<=157.5 or 292.5<angle[i][j]<=337.5:
                if mag[i][j] < mag[i-1][j+1] or mag[i][j] < mag[i+1][j-1]:
                    if mag[i][j] > thresh:
                        mag[i][j] = 0"""
  return mag


imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy_nonmax
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score:', f1)
