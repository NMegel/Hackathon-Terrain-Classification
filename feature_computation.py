'''
These 2 functions are designed to be mapped on images of any size of a dataset separately.
The example here is given with 4 canals: Red, Green, Blue, Infrared

The mapping example is computed with pyspark
For each image it returns a list of 62 features
'''

import h5py as h5
import numpy as np
import pywt
from pyspark import Row
from scipy import stats
from skimage.color import rgb2rgbcie

# Import H5 data
PATH_DATA = 'data/YourImageDataset.h5'
f = h5.File(PATH_DATA)

# Number of layers for wavelets

N_LAYER = 4    
    
def wavelet_dec(y):
    # Wavelet definition
    wavelet = pywt.Wavelet('db2')
    # print('Maximum level of decomposition: {}'.format(pywt.dwt_max_level(len(y),wavelet)))
    # Transform
    coeffs = pywt.wavedec2(y, wavelet) #coefs stores the values of d
    A0, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    B0, (B1, B2, B3), (B4, B5, B6) = coeffs
    return coeffs

def features_per_coeffs(coeffs):
    B0, (B1, B2, B3), (B4, B5, B6) = coeffs
    suitable_coeffs = [B0, B1, B2, B4, B5]
    features = np.zeros(8)
    mean = np.zeros(len(suitable_coeffs))
    energy = np.zeros(len(suitable_coeffs))
    Etotal = 0
    for i, coeff in enumerate(suitable_coeffs):
        mean[i] = np.mean(coeff)
        energy[i] = np.sum(np.square(coeff))
        Etotal += energy[i]
    energy = energy*100/Etotal
   
    features[0] = mean[0]*0.1
    features[1] = mean[3]
    features[2] = mean[4]
    features[3:] = energy
    return features

def compute_features(t):
    image, lbl = t
    # Mean
    mean = np.mean(image[:,:,0:4], axis=(0,1))
    mean = mean.tolist()
    mean = [float(m) for m in mean]
    # STD
    std = np.std(image[:,:,0:4], axis= (0,1))
    std = std.tolist()
    std = [float(s) for s in std]
    
    # MODE I
    mode = list()
    for i in range(0,4):
        mode_canal = float(stats.mode(image[:,:,i])[0][0][0])
        mode.append(mode_canal)
    
    # Entropy for the 4 canals
    entropy = list()
    for j in range(0,4):
        hist = np.histogram(image[:,:,j], bins = 100)[0]
        somme = 0
        for i in range(len(hist)):
            if hist[i]!=0:
                prob = float(hist[i])/256**2
                somme = somme- prob * np.log(prob)/np.log(2)
        entropy.append(float(somme))
       
    # FFT
    result = np.zeros((8)) 
    for j in range(4):
            temp = np.fft.fftshift(abs(np.fft.fft2(image[:,:,j])))
            result[2*j] = temp.mean()
            result[2*j+1] = temp.std()
    result = result.tolist()
    fft = [float(r) for r in result]
    
    
    # NDVI, EVI and NDWI
    red = image[:,:,0]
    infra = image[:,:,3]
    blue = image[:,:,2]
    green = image[:,:,1]
    
    NDVI_channel = (infra-red)/(infra+red)
    NDVI_channel[NDVI_channel == -np.inf] = 0
    NDVI_channel[NDVI_channel == np.inf] = 0
    ndvi = [NDVI_channel.mean(), NDVI_channel.std()]
    ndvi = [float(n) for n in ndvi]

    #selon wikipédia:
    EVI_channel = 2.5*(infra-red)/(infra+6*red-7.5*blue+1)
    EVI_channel[EVI_channel == -np.inf] = 0
    EVI_channel[EVI_channel == np.inf] = 0
    evi = [EVI_channel.mean(), EVI_channel.std()]
    evi = [float(n) for n in evi]

    NDWI_channel = (green-infra)/(green+infra)
    NDWI_channel[NDWI_channel == -np.inf] = 0
    NDWI_channel[NDWI_channel == np.inf] = 0
    ndwi = [NDWI_channel.mean(), NDWI_channel.std()]
    ndwi = [float(n) for n in ndwi]
    
    # Wavelets:
    f = np.zeros((N_LAYER,8))
    for i in range(N_LAYER):
        coeffs = wavelet_dec(image[:,:,i])
        f[i,] = features_per_coeffs(coeffs)
    f = f.ravel().tolist()

    return mean + std + mode + entropy + fft + ndvi + evi + ndwi + f + lbl.tolist()

# Parallelization of features computation with Spark mapping

f = h5.File(PATH_DATA, 'r')
X, Y = f["S2"], f["TOP_LANDCOVER"]

ims = list()
lbls = list()
print(len(X))
for i in range(len(X)):
    im=X[i,::]
    lbl=Y[i]
    
    ims.append(im)
    lbls.append(lbl)
    
data = sc.parallelize(zip(ims, lbls),50)
f.close()

names = ["Mean R", "Mean G", "Mean B", "Mean I", "STD R", "STD G", "STD B", "STD I","MOD R", "MOD G", "MOD B" ,"MOD I", "entropy R", "entropy G", "entropy B", "entropy I"] + ["Mean FFT R", "STD FFT R", "Mean FFT G", "STD FFT G", "Mean FFT B", "STD FFT B", "Mean FFT I", "STD FFT I"] + ["MEAN NDVI", " STD NDVI", "MEAN EVI", "STD EVI", "MEAN NDWI", "STD NDWI"] + ["layer1_{}".format(i) for i in range(8)] + ["layer2_{}".format(i) for i in range(8)] + ["layer3_{}".format(i) for i in range(8)] + ["layer4_{}".format(i) for i in range(8)]

df = data.map(compute_features).toDF(names+["Label"])
df.printSchema()