import numpy as np
import scipy
from skimage.color import rgb2gray

def GetAngleGraph(img):
    height, width = img.shape
    aspectRatio = width/height
    maxr = min(height,width)
    angArry = []
    for row in range(height):
        for col in range(width):
            r = np.sqrt((height-1-row)**2 + col**2)
            if r < maxr:
                mag = np.abs(img[row,col])
                theta = np.arctan2(((height-1-row)*aspectRatio),col)
                angArry.append((mag, theta))
    return angArry

def GetBestAngle(angMag, numbins=200):
    
    thetas = [tup[1] for tup in angMag]
    hist, binedges = np.histogram(thetas, numbins, (0.05,(np.pi/2)-0.05))
    
    indexes = np.digitize(thetas, binedges)
    binSum = np.zeros(len(binedges)+1)
    binCount = np.zeros(len(binedges)+1)
    
    for (mag,theta),index in zip(angMag,indexes):
        binSum[index] += mag
        binCount[index] += 1
        
    avgAngle = binSum/binCount
    sigma = 1
    filtered = scipy.ndimage.gaussian_filter1d(avgAngle, sigma)
    avg = (binedges[np.argmax(filtered[1:])] + binedges[np.argmax(filtered[1:])+1])/2
    
    #calculate standard deviation
    centerIndex = np.argmax(filtered[1:])
    thetaDevWidth = 0.1
    stdevWidth = int(np.round(numbins*(binedges[1]-binedges[0])/thetaDevWidth))
    stdevData = filtered[(centerIndex-stdevWidth):(centerIndex+stdevWidth)]
    stdev = np.std(stdevData)
    
    threshold = 28
    if stdev > threshold:
        avg -= np.pi/2
    return avg

def FixRotation(img):
    img1 = np.asarray(img)
    img1 = rgb2gray(img1)
    
    fftimg = np.fft.fft2(img1)
    fftimg2 = np.fft.fftshift(fftimg)
    
    cropimg = fftimg2[:fftimg2.shape[0]//2,fftimg2.shape[1]//2:]
    angMag = GetAngleGraph(cropimg)
    bestang = GetBestAngle(angMag)
    
    bestang = (bestang*180/np.pi)
    rotimage = img.rotate(-bestang)
    return rotimage, bestang