import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

image_scene = cv2.imread('./asset/ImageCV/kitkat_scene.jpg')

Image_Target_Folder = "ImageCV/ImageCV/targetList/"

# Image_Target_Path = []
Image_Target = []

for path in os.listdir(Image_Target_Folder):
    img_path = Image_Target_Folder + path
    # Image_Target_Path.append(img_path)
    image = cv2.imread(img_path)
    image = cv2.blur(image, (2,2))
    Image_Target.append(image)

# Kita akan menyimpan semua keypoint dan descriptor dari gambar scene dan target untuk semua detektor dan descriptor perlu diubah ke bentuk float32 agar lebih sesuai kecuali detektor orb

SIFT = cv2.SIFT_create()
ORB = cv2.ORB_create()
AKAZE = cv2.AKAZE_create()

SIFT_Keypoint_Target = []
SIFT_Descriptor_Target = []
SIFT_Keypoint_Scene, SIFT_Descriptor_Scene = SIFT.detectAndCompute(image_scene, None)
SIFT_Descriptor_Scene = np.float32(SIFT_Descriptor_Scene)

ORB_Keypoint_Target = []
ORB_Descriptor_Target = []
ORB_Keypoint_Scene, ORB_Descriptor_Scene = ORB.detectAndCompute(image_scene, None)

AKAZE_Keypoint_Target = []
AKAZE_Descriptor_Target = []
AKAZE_Keypoint_Scene, AKAZE_Descriptor_Scene = AKAZE.detectAndCompute(image_scene, None)
AKAZE_Descriptor_Scene = np.float32(AKAZE_Descriptor_Scene)

for image in Image_Target:
    keypoint, descriptor = SIFT.detectAndCompute(image, None)
    descriptor = np.float32(descriptor)
    SIFT_Keypoint_Target.append(keypoint)
    SIFT_Descriptor_Target.append(descriptor)

    keypoint, descriptor = ORB.detectAndCompute(image, None)
    ORB_Keypoint_Target.append(keypoint)
    ORB_Descriptor_Target.append(descriptor)

    keypoint, descriptor = AKAZE.detectAndCompute(image, None)
    descriptor = np.float32(descriptor)
    AKAZE_Keypoint_Target.append(keypoint)
    AKAZE_Descriptor_Target.append(descriptor)

#Matcher
#SIFT & AKAZE == FLANN
#ORB == BFMATCHER
flann = cv2.FlannBasedMatcher(dict(algorithm=1), dict(checks=100))
bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def createMasking(mask, match):
    for i, (fm, sm) in enumerate(match):
        if fm.distance < 0.7 * sm.distance:
            mask[i] = [1, 0]
    return mask

#SIFT
SIFT_Match_Result = []

for descriptor in SIFT_Descriptor_Target:
    result = flann.knnMatch(descriptor, SIFT_Descriptor_Scene, 2)
    SIFT_Match_Result.append(result)

SIFT_Result = max(SIFT_Match_Result, key=len)
SIFT_Image_Index = SIFT_Match_Result.index(SIFT_Result)
SIFT_Matches_Mask = [[0,0] for i in range(len(SIFT_Result))]
SIFT_Matches_Mask = createMasking(SIFT_Matches_Mask, SIFT_Result)

SIFT_Res = cv2.drawMatchesKnn(
    Image_Target[SIFT_Image_Index],
    SIFT_Keypoint_Target[SIFT_Image_Index],
    image_scene,
    SIFT_Keypoint_Scene,
    SIFT_Result, None,
    matchColor=[255,0,0],
    singlePointColor=[0, 255, 0],
    matchesMask=SIFT_Matches_Mask
)

#ORB
ORB_Match_Result = []

for descriptor in ORB_Descriptor_Target:
    result = bfmatcher.match(descriptor, ORB_Descriptor_Scene)
    result = sorted(result, key = lambda x : x.distance)
    ORB_Match_Result.append(result)

ORB_Result = max(ORB_Match_Result, key=len)
ORB_Image_Index = ORB_Match_Result.index(ORB_Result)

ORB_Res = cv2.drawMatches(
    Image_Target[ORB_Image_Index],
    ORB_Keypoint_Target[ORB_Image_Index],
    image_scene,
    ORB_Keypoint_Scene,
    ORB_Result[:30], None,
    matchColor=[255,0,0],
    singlePointColor=[0,255,0],
    flags=2
)

#AKAZE
AKAZE_Match_Result = []

for descriptor in AKAZE_Descriptor_Target:
    result = flann.knnMatch(descriptor, AKAZE_Descriptor_Scene, 2)
    AKAZE_Match_Result.append(result)

AKAZE_Result = max(AKAZE_Match_Result, key=len)
AKAZE_Image_Index = AKAZE_Match_Result.index(AKAZE_Result)
AKAZE_Matches_Mask = [[0,0] for i in range(len(AKAZE_Result))]
AKAZE_Matches_Mask = createMasking(AKAZE_Matches_Mask, AKAZE_Result)

AKAZE_Res = cv2.drawMatchesKnn(
    Image_Target[AKAZE_Image_Index],
    AKAZE_Keypoint_Target[AKAZE_Image_Index],
    image_scene,
    AKAZE_Keypoint_Scene,
    AKAZE_Result, None,
    matchColor=[255,0,0],
    singlePointColor=[0, 255, 0],
    matchesMask=AKAZE_Matches_Mask
)

SIFT_Res = cv2.cvtColor(SIFT_Res, cv2.COLOR_BGR2RGB)
ORB_Res = cv2.cvtColor(ORB_Res, cv2.COLOR_BGR2RGB)
AKAZE_Res = cv2.cvtColor(AKAZE_Res, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.subplot(2,2,1)
plt.imshow(SIFT_Res, cmap='gray')
plt.title("SIFT")
plt.subplot(2,2,2)
plt.imshow(ORB_Res, cmap='gray')
plt.title("ORB")
plt.subplot(2,2,3)
plt.imshow(AKAZE_Res, cmap='gray')
plt.title("AKAZE")
plt.show()