import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt

"Opens the image and store it into a numpy array"
def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    npImage = sitk.GetArrayFromImage(itkimage)

    npOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    npSpacing = np.array(list(reversed(itkimage.GetSpacing())))

    #returns image (npImage), origin(npOrigin) and pixel spacing (npSpacing)
    return npImage, npOrigin, npSpacing

"open and read the list of candidates in the candidates.csv file"
def readCSV(filename):
    lines = []
    with open(filename, 'r') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)    
    return lines

"transform from world to voxel coordinates"
def world2VoxelCoord(worldCoord, origin, spacing):

    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing

    return voxelCoord

"define normalised planes to extract views from the candidates"
def normalisePlanes(npzarray):

    maxHU = 400
    minHU = -1000

    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1
    npzarray[npzarray<0] = 0

    return npzarray

"define image and candidate path"
img_path = 'data\\1.3.6.1.4.1.14519.5.2.1.6279.6001.148447286464082095534651426689.mhd'
cand_path = 'data\\candidates1.csv'

image, origin, spacing = load_itk_image(img_path)
print(image.shape)
print(origin)
print(spacing)

cands = readCSV(cand_path)

for cand in cands[1:]:
    worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
    voxelCoord = world2VoxelCoord(worldCoord, origin, spacing)
    voxelWidth = 65

    patch = image[int(voxelCoord[0]), int(voxelCoord[1]-voxelWidth/2):int(voxelCoord[1]+voxelWidth/2), int(voxelCoord[2]-voxelWidth/2):int(voxelCoord[2]+voxelWidth/2)]
    patch = normalisePlanes(patch)

    print(worldCoord)
    print(voxelCoord)
    print(patch)

    plt.imshow(patch, cmap='gray')
    plt.show()
