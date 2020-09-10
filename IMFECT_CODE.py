import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import numpy
from scipy.stats import kurtosis, skew
from pandas import DataFrame
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import cv2
import sys, os

#READ HERE: THE pip installs you should run

#Click on terminal in the bottom left.

#pip install matplotlib
#pip install scipy
#pip install skimage
#pip install pandas
#pip install numpy
#pip install opencv-python

sys.stdout = open(os.devnull, 'w')  # Suppressing print statements

#Getting the name of the image:
# name='TestImages/testimage_2_ang105width2.tif'

# name ='TestImages/collagen1.tif'

name='TestImages/Picture 7.tif'

# name ='SmallTestImages/C1_0pt4_F_FOV1_REDO_c4_sh.tif'

#Reading the image from a file:
img=cv2.imread(name)

#Converting this image to grayscale, if it isn't already..
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# images = imread_collection( 'TestImages/testimage_1_ang90width2.tif', conserve_memory=True)

#Only want to analyze 1 image at a time,
images=[img]

plt.figure(100)
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

def getCorrelation(input_image,angles,dist):
    """

    Loops through a list of given angles, finding the correlation value at each angle.

    :param input_image: a single grayscale image to analyze
    :param angles: a list of angles in radians (usually from 0 to pi)  to measure correlation at
    :param dist: a single distance parameter to measure correlation at
    :return: An array called "all_corr" containn the correlation values at every angle.
    """

    final_ds = DataFrame(
        columns = [
            'img_id',
            'mean',
            'std',
            'kurtosis',
            'skew',
            'entropy',
            'contrast',
            'dissimilarity',
            'energy',
            'ASM',
            'homogeneity',
            'correlation'
        ]
    )

    img_id = 0
    image= input_image
    all_corr=[]

    for angle in angles:
        glcm = greycomatrix(
            image = image,
            distances = [dist],
            angles = [angle],
            levels = 256,
            symmetric = True,
            normed = True
        )

        t = {
            'img_id':[img_id],
            'mean': [numpy.average(image)],
            'std': [numpy.std(image)],
            'kurtosis': [kurtosis(image.flatten())],
            'skew': [skew(image.flatten())],
            'entropy': [shannon_entropy(glcm, base=numpy.e)],
            'contrast': [greycoprops(glcm, 'contrast')[0,0]],
            'dissimilarity': [greycoprops(glcm, 'dissimilarity')[0,0]],
            'energy': [greycoprops(glcm, 'energy')[0,0]],
            'ASM': [greycoprops(glcm, 'ASM')[0,0]],
            'homogeneity': [greycoprops(glcm, 'homogeneity')[0,0]],
            'correlation': [greycoprops(glcm, 'correlation')[0,0]]
        }

        t = DataFrame(
            data = t,
            columns = [
                'img_id',
                'mean',
                'std',
                'kurtosis',
                'skew',
                'entropy',
                'contrast',
                'dissimilarity',
                'energy',
                'ASM',
                'homogeneity',
                'correlation'
            ]
        )

        img_id += 1
        final_ds = final_ds.append(t)

        corr= greycoprops(glcm, 'correlation')[0,0]
        all_corr.append(corr)


        #Printing a bunch of data from the glcm
        print(final_ds)

        # printing the angle & corresponding correlation:
        print("Angle = ", angle, "Correlation = ", greycoprops(glcm, 'correlation')[0,0])

    return all_corr



## Setting up an array from 0 to 180
ang_mult= np.arange(180)

# converting this array into radians
angles= [np.pi/180*x for x in ang_mult]

# An empty array which will contain 4 sub-arrays of the correlations at each distance
dist_corr=[]

#TODO: PLAY WITH THESE VALUES: all_dist array contains the distances you would like to analyze

## Code only handles 4 distances right now:
all_dist=[16]

#Running the getCorrelation function at different distance parameters
for i in range(0,len(all_dist)):
    correllations= getCorrelation(images[0],angles,all_dist[i])
    dist_corr.append(correllations)

#Finding the dominant angle to draw in red on the original image
#Do this by looking at the line corresponding the last (highest) distance .
reported_angle= np.where(dist_corr[len(dist_corr)-1]==np.max(dist_corr[len(dist_corr)-1]))

#If there are multiple angles with the same max correlation, take the median.
if len(reported_angle)>0:
    reported_angle=int(np.median(reported_angle))

# The GLCM actually uses angles ina different format, so we convert them..
true_angle=180-reported_angle
print("The true angle is:", true_angle)


#Converting the true angle into radians.
tru_angle_rad=true_angle*np.pi/180

#Plotting the dominant angle line on the image.
norm_vector=[np.sin(tru_angle_rad),np.cos(tru_angle_rad)]
norm_vector=np.array([norm_vector[1],norm_vector[0]*-1])

im=images[0]
h=im.shape[0]
w=im.shape[1]
centre=np.array([int(w/2),int(h/2)])
diag_half=int(w/2)

pt1 = np.subtract(centre, np.multiply(norm_vector, diag_half))
pt2 = np.add(centre, np.multiply(norm_vector, diag_half))
print("Point 1", pt1, "Point 2", pt2)

#Now that we have the two points of the line, we draw it on matplotlib
plt.figure(4)
plt.imshow(im,cmap=plt.cm.gray)
plt.title(f'Line direction, angle={true_angle}')

plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],color='red',label='Calculated line',linewidth=2.0)
plt.legend()
plt.draw()








## Running a plot for the correllations at different distances & angles.
plt.figure(6)
plt.title('Correlation vs angles')

angles_adj_for_plt = [180-y for y in ang_mult]
sys.stdout = sys.__stdout__  # Re-allowing print statements

for dist_index,corr in enumerate(dist_corr):
    plt.plot(angles_adj_for_plt,corr, markerfacecolor='black', markersize=3,  linewidth=2, label= f"Distance ={all_dist[dist_index]}")

    corr_set=len(set(corr))
    if len(corr)-corr_set==0:
        print("Unique Correlations found at a distance value of",all_dist[dist_index])


plt.ylabel('Correlation')
plt.xlabel('Angle')
plt.legend()
plt.show()