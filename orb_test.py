import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

folder_path = './frames/'

# Load the two images
img1 = cv2.imread(folder_path+'frame0051.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your first image filename
img2 = cv2.imread(folder_path+'frame0052.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your second image filename

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=3000)

# Detect keypoints and compute descriptors
tic = time.time()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match descriptors using Brute-Force matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Apply RANSAC to filter matches geometrically
filtered_matches = []
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    filtered_matches = [m for m, keep in zip(good_matches, matchesMask) if keep]

# Draw matches
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, filtered_matches, None, flags=2)
toc = time.time()

print(toc-tic, 'seconds')

# Show result
plt.figure(figsize=(12, 6))
plt.imshow(matched_img)
plt.title(f'Filtered ORB Matches ({len(filtered_matches)} shown)')
plt.axis('off')
plt.show()
