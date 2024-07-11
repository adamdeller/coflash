#!/usr/bin/env python
import os, sys, copy
import argparse
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy import units as u
#from astropy.modeling.models import Ellipse2D
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

class FRBHost:
    def __init__(self, hostimagefile, verbose):
        self.hostimagefile = hostimagefile
        self.verbose = verbose
        self.descriptions = ["Un-modified image", "Smoothed profile", "Residual from smoothed fit", "Mask"]
        self.shortdescriptions = ["Original", "Profile", "Residual", "Mask"]

        # Open up the FITS image of host + model and check it
        self.hostimagehdul = fits.open(self.hostimagefile)
        if len(self.hostimagehdul) != 4:
            print("FITS image file must have 4 HDUs:")
            print(descriptions)
            sys.exit()

        # Estimate and remove the baseline
        self.baseline = self.hostimagehdul[1].data.min()
        if self.verbose:
            print("Minimum from original image is ", self.hostimagehdul[0].data.min(), ", from profile is", self.hostimagehdul[1].data.min())

        # Get the WCS
        self.w = wcs.WCS(self.hostimagehdul[1].header, self.hostimagehdul)

    def growMask(self, expandmaskby):
        self.originalmaskdata = self.hostimagehdul[3].data
        self.maskdata = np.copy(self.originalmaskdata)

        # Expand the mask by one pixel in every direction (one or more times)
        for grow in range(expandmaskby):
            for i in range(self.maskdata.shape[0] - 1):
                for j in range(self.maskdata.shape[1] - 1):
                    if self.originalmaskdata[i+1][j] == 1 or self.originalmaskdata[i][j+1] == 1:
                        self.maskdata[i][j] = 1
                    if self.originalmaskdata[i][j] == 1:
                        self.maskdata[i][j+1] = 1
                        self.maskdata[i+1][j] = 1
            self.originalmaskdata = np.copy(self.maskdata)
        self.originalmaskdata = self.hostimagehdul[3].data

    def addFRBLocalisation(self, frbradeg, febdecdeg, frbsigmaraarcsec, frbsigmadecarcsec):
        # Store the information about the FRB localisation
        self.frbradeg = frbradeg
        self.frbdecdeg = febdecdeg
        self.sigmaraarcsec = frbsigmaraarcsec
        self.sigmadecarcsec = frbsigmadecarcsec

        # Now append a new Image HDU to the end of the HDU list
        self.hostimagehdul.append(self.hostimagehdul[1].copy())

        # Make a deep copy of the data
        self.hostimagehdul[4].data = copy.deepcopy(self.hostimagehdul[4].data)

        # Update this new Image HDU to contain the FRB likelihood
        self.hostimagehdul[4].data -= self.hostimagehdul[4].data # Set it to zero initially
        oversamplefactor = 4
        subpixgrid = np.zeros(oversamplefactor*oversamplefactor*2).reshape(oversamplefactor*oversamplefactor, 2)
        for i in range(oversamplefactor):
            for j in range(oversamplefactor):
                subpixgrid[i*oversamplefactor + j][0] = (i+oversamplefactor/2.0)/float(oversamplefactor)
                subpixgrid[i*oversamplefactor + j][1] = (j+oversamplefactor/2.0)/float(oversamplefactor)
        #subpixgrid = []
        #for i in range(oversamplefactor):
        #    for j in range(oversamplefactor):
        #        subpixgrid.append([(i+oversamplefactor/2.0)/4.0, (j+oversamplefactor/2.0)/4.0])
        for x in range(self.hostimagehdul[4].data.shape[0]):
            for y in range(self.hostimagehdul[4].data.shape[1]):
                pixgrid = [[l[0]+x, l[1]+y] for l in subpixgrid]
                radecgrid = self.w.all_pix2world(pixgrid, 1)
                radecoffsets = [[3600*(g[0]-self.frbradeg)*np.cos(self.frbdecdeg*np.pi/180), 3600*(g[1]-self.frbdecdeg)] for g in radecgrid]
                likelihoods = [np.e**(-(o[0]/self.sigmaraarcsec)**2) * np.e**(-(o[1]/self.sigmadecarcsec)**2) for o in radecoffsets]
                self.hostimagehdul[4].data[y, x] = np.sum(likelihoods)

        # Normalise the FRB localisation probability density
        imsum = self.hostimagehdul[4].data.sum()
        self.hostimagehdul[4].data /= imsum

        # Update the descriptions
        self.descriptions.append("FRB Localisation Likelihood")
        self.shortdescriptions.append("Localisation")

        # Create a space to store likelihoods
        self.nummodels = len(self.hostimagehdul) - 2
        self.modelloglikelihoods = np.zeros(self.nummodels)

    def evaluateModels(self, dostretch):
        for i in range(len(self.descriptions)):
            if self.shortdescriptions[i] == "Mask" or self.shortdescriptions[i] == "Localisation":
                continue # Don't need to evaluate anything for this image
            imagedata = self.hostimagehdul[i].data
            print("Looking at image", i, ", which is the", self.descriptions[i])
            if self.shortdescriptions[i] == 'Original' or self.shortdescriptions[i] == 'Profile':
               imagedata -= self.baseline # Subtract off the baseline for all original and the profile
            imagedata *= (1-self.maskdata)
    
            # Theshold the residual (and the profile) image at zero - no negative values
            imagedata[imagedata < 0] = 0.0

            # If dostretch, then do something here
            if dostretch:
                imagedata = imagedata**0.5

            # Normalise to ensure that the total probability density is unity 
            imsum = imagedata.sum()
            if self.verbose:
                print("Imsum is", imsum)
            imagedata /= imsum

            # Calculate the log likelihood for this model
            self.modelloglikelihoods[i] = (imagedata * self.hostimagehdul[4].data).sum()

    def printReport(self):
        for i in range(len(self.descriptions)):
            if self.shortdescriptions[i] == "Mask" or self.shortdescriptions[i] == "Localisation":
                continue # Don't need to evaluate anything for this image
            print("For model based on {0}, the log likelihood is {1}".format(self.descriptions[i], self.modelloglikelihoods[i]))

    def plotAll(self):
        for i in range(len(self.descriptions)):
            ax = plt.subplot(projection=self.w)
            ax.coords[0].set_axislabel('Right Ascension (J2000)')
            ax.coords[1].set_axislabel('Declination (J2000)')
            ax.imshow(self.hostimagehdul[i].data) #, vmin=-2.e-5, vmax=2.e-4, origin='lower')
            ax.scatter(self.frbradeg, self.frbdecdeg, transform=ax.get_transform('icrs'), s=30, edgecolor='white', facecolor='none')
            plt.savefig(self.shortdescriptions[i] + ".png")
            plt.clf()

def plotAndFit(imagedata, shortdescription, radeg, decdeg, frbrapix, 
               frbdecpix, rauncertaintypix, decuncertaintypix, w, 
               dostretch=False, plotFRBLikelihood=False):
    if dostretch:
        idata = (imagedata + 1e-9)**0.5
        idata -= idata.min()
        idata /= idata.sum()
    else:
        idata = imagedata
    fdata = np.zeros(idata.shape)
    zoomfactor = 2
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(idata[(idata.shape[0]*(zoomfactor-1))//(zoomfactor*2) : (idata.shape[0]*(zoomfactor+1))//(zoomfactor*2),
                        (idata.shape[1]*(zoomfactor-1))//(zoomfactor*2) : (idata.shape[1]*(zoomfactor+1))//(zoomfactor*2)])
    axs[1].hist(idata.flatten(), bins=100, log=True)

    # Also plot the localisation region...
    uncertaintyangle = 0
    # Ellipse takes diameter, not radii, hence the factor of 2 here...
    ell = Ellipse((frbrapix - idata.shape[0]*(zoomfactor-1)//(zoomfactor*2), frbdecpix - idata.shape[1]*(zoomfactor-1)//(zoomfactor*2)), 2*rauncertaintypix, 2*decuncertaintypix, uncertaintyangle, fc='none', ec='red', lw=1)
    axs[0].add_patch(ell)
    #print("Plotting ellipse at {0},{1} with x length {2} x {3}".format(frbrapix, frbdecpix, rauncarcsec/rapixscale.value, decuncarcsec/decpixscale.value))
    print("RA uncertainty pix is {0}, Dec uncertainty pix is {1}".format(rauncertaintypix, decuncertaintypix))
    plt.savefig(shortdescription + ".png")
    plt.clf()

    likelihood = 0

    # Get a grid of +/- 4sigma RA/dec positions around the most likely FRB position, oversampling each optical pixel by 4x
    pixgrid = []
    for j in range(40):
        for k in range(40):
            pixgrid.append([int(frbrapix) - 5 + 0.25 + j*0.25, int(frbdecpix) - 5 + 0.25 + k*0.25])
    radecgrid = w.all_pix2world(pixgrid, 0)
    for j in range(len(pixgrid)):
        xpix, ypix = pixgrid[j]
        ra, dec = radecgrid[j]
        #print(ra - radeg, dec - decdeg, xpix, ypix, idata[int(xpix), int(ypix)], np.e**(-((ra - radeg)/(rauncarcsec/3600.))**2)*np.e**(-((dec - decdeg)/(decuncarcsec/3600.))**2))
        likelihood += 0.25*idata[int(xpix), int(ypix)]*np.e**(-((ra - radeg)/(rauncarcsec/3600.))**2)*np.e**(-((dec - decdeg)/(decuncarcsec/3600.))**2)
        fdata[int(xpix), int(ypix)] += 0.25*np.e**(-((ra - radeg)/(rauncarcsec/3600.))**2)*np.e**(-((dec - decdeg)/(decuncarcsec/3600.))**2)
    print(idata[int(np.round(frbrapix)) - 3:int(np.round(frbrapix))+3, int(np.round(frbdecpix))-3:int(np.round(frbdecpix))+3])
    if plotFRBLikelihood:
        fig, ax = plt.subplots(1,1)
        ax.imshow(fdata[(fdata.shape[0]*(zoomfactor-1))//(zoomfactor*2) : (fdata.shape[0]*(zoomfactor+1))//(zoomfactor*2),
                        (fdata.shape[1]*(zoomfactor-1))//(zoomfactor*2) : (fdata.shape[1]*(zoomfactor+1))//(zoomfactor*2)])
        plt.savefig("frblikelihood.png")
        plt.clf()
    return likelihood

if __name__ == "__main__":
    # Parse arguments
    programdescription = 'Estimate the likelihood of FRB seen at particular location in a host, given a model'
    parser = argparse.ArgumentParser(
                    prog='getFRBLikelihood',
                    description=programdescription,
                    epilog='Models being added all the time')
    parser.add_argument('-i', '--hostimagefile', type=str, default='', 
                        help='The FRBs host, in FITS image format')
    parser.add_argument('-m', '--growmaskpixels', type=int, default=3, 
                        help='Number of pixels to grow the mask by')
    parser.add_argument('--frbra', type=float, help='FRB right ascension in degrees')
    parser.add_argument('--frbdec', type=float, help='FRB declination in degrees')
    parser.add_argument('--rauncarcsec', type=float, help='Uncertainty in FRB RA (arcseconds)')
    parser.add_argument('--decuncarcsec', type=float, help='Uncertainty in FRB Dec (arcseconds)')
    parser.add_argument('--dostretch', default=False, action='store_true',
                        help='Apply a stretch to the host galaxy residual image')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    args = parser.parse_args()

    # Check that we actually got a filename to work with
    if args.hostimagefile == '':
        parser.error('You must supply a FITS filename to hostimagefile')

    # Create FRBHost object
    frbhost = FRBHost(args.hostimagefile, args.verbose)

    # Expand the mask if desired
    frbhost.growMask(args.growmaskpixels)

    # Add the information about the FRB localisation
    frbhost.addFRBLocalisation(args.frbra, args.frbdec, args.rauncarcsec, args.decuncarcsec)

    # Calculate the likelihoods for different models and make plots
    frbhost.evaluateModels(args.dostretch)
    frbhost.printReport()
    frbhost.plotAll()

#
#    if not len(sys.argv) == 6:
#        print("Usage: {0} <FITS image> <RAdeg> <Decdeg> <RAunc_arcsec> <Decunc_arcsec>".format(sys.argv[0]))
#        sys.exit()
#    
#    imagefile = sys.argv[1]
#    radeg     = float(sys.argv[2])
#    decdeg    = float(sys.argv[3])
#    rauncarcsec = float(sys.argv[4])
#    decuncarcsec = float(sys.argv[5])
#    expandmaskby = 3 # pixels
#    dostretch = False
#    descriptions = ["Un-modified image", "Smoothed profile", "Residual from smoothed fit", "Mask"]
#    shortdescriptions = ["Original", "Profile", "Residual", "Mask"]
#    
#    # Open up the FITS file
#    imagehdul = fits.open(imagefile)
#    if len(imagehdul) != 4:
#        print("FITS image file must have 4 HDUs:")
#        print(descriptions)
#        sys.exit()
#    likelihood = np.zeros(3)
#    baseline = imagehdul[1].data.min()
#    print("Minimum from original image is ", imagehdul[0].data.min(), ", from profile is", imagehdul[1].data.min())
#    w = wcs.WCS(imagehdul[1].header, imagehdul)
#    
#    # First get the mask
#    originalmaskdata = imagehdul[3].data
#    maskdata = np.copy(originalmaskdata)
#    
#    # Expand the mask by one pixel in every direction (one or more times)
#    for grow in range(expandmaskby):
#        for i in range(maskdata.shape[0] - 1):
#            for j in range(maskdata.shape[1] - 1):
#                if originalmaskdata[i+1][j] == 1 or originalmaskdata[i][j+1] == 1:
#                    maskdata[i][j] = 1
#                if originalmaskdata[i][j] == 1:
#                    maskdata[i][j+1] = 1
#                    maskdata[i+1][j] = 1
#        originalmaskdata = np.copy(maskdata)
#    originalmaskdata = imagehdul[3].data
#
#    # Now append a new Image HDU to the end
#    imagehdul.append(imagehdul[1].copy())
#
#    # Update this new Image HDU to contain the FRB likelihood
#    imagehdul[4].data -= imagehdul[4].data # Set it to zero initially
#    
#    descriptions.append("FRB Localisation Likelihood")
#    shortdescriptions.append("Localisation")
#    
#    dolog = True
#    # Now loop through the other images, plotting them
#    for i in range(3):
#        imagedata = imagehdul[i].data
#        print("Looking at image", i, ", which is the", descriptions[i])
#        if i< 2:
#            imagedata -= baseline # Subtract off the baseline for all original and the profile
#        imagedata *= (1-maskdata)
#    
#        # Get the pixel scale
#        pixcrd = np.array([[1, 1], [2, 2]])
#        world = w.wcs_pix2world(pixcrd, 0)
#        rapixscale = (world[0,0] - world[1,0])*3600*np.cos(world[1,1]*np.pi/180)*u.arcsec
#        decpixscale = (world[1,1] - world[0,1])*3600*u.arcsec
#        print("Rapixscale {0}, decpixscale {1}".format(rapixscale, rapixscale))
#    
#        # Get pixel position of the most likely FRB position
#        xypix = w.wcs_world2pix(radeg, decdeg, 0)
#        frbrapix = xypix[0]
#        frbdecpix = xypix[1]
#        #print("FRB RA/Dec pix are: ", frbrapix, frbdecpix)
#    
#    
#        ## Theshold the residual (and the profile) image at zero - no negative values
#        #if i>0:
#        #    imagedata[imagedata < 0] = 0.0
#        # Threshold all images at zero - no negative values
#        imagedata[imagedata < 0] = 0.0
#        imsum = imagedata.sum()
#        print("Imsum is", imsum)
#        imagedata /= imsum
#        fig, axs = plt.subplots(1,2)
#        plotfrblikelihood = False
#        if i==0:
#            plotfrblikelihood = True
#    
#        likelihood[i] = plotAndFit(imagedata, shortdescriptions[i], radeg, decdeg, frbrapix, frbdecpix, rauncarcsec/rapixscale.value, decuncarcsec/decpixscale.value, w, dostretch, plotfrblikelihood)
#        #axs[0].imshow(imagedata)
#        #axs[1].hist(imagedata.flatten(), bins=100, log=True)
#    
#        ## Also plot the localisation region...
#        #uncertaintyangle = 0
#        #ell = Ellipse((frbrapix, frbdecpix), rauncarcsec/rapixscale.value, decuncarcsec/decpixscale.value, uncertaintyangle, fc=None, ec='red', lw=1) 
#        #axs[0].add_patch(ell)
#        ##print("Plotting ellipse at {0},{1} with x length {2} x {3}".format(frbrapix, frbdecpix, rauncarcsec/rapixscale.value, decuncarcsec/decpixscale.value))
#        #plt.savefig(shortdescriptions[i] + ".png")
#        #plt.clf()
#    
#        ## Get a grid of +/- 4sigma RA/dec positions around the most likely FRB position, oversampling each optical pixel by 4x
#        #pixgrid = []
#        #for j in range(40):
#        #    for k in range(40):
#        #        pixgrid.append([int(frbrapix) - 5 + 0.25 + j*0.25, int(frbdecpix) - 5 + 0.25 + k*0.25])
#        #radecgrid = w.all_pix2world(pixgrid, 0)
#        #for j in range(len(pixgrid)):
#        #    xpix, ypix = pixgrid[j]
#        #    ra, dec = radecgrid[j]
#        #    #print(ra - radeg, dec - decdeg)
#        #    likelihood[i] += 0.25*imagedata[int(xpix), int(ypix)]*np.e**(-((ra - radeg)/(rauncarcsec/3600.))**2)*np.e**(-((dec - decdeg)/(decuncarcsec/3600.))**2)
#        print("Likelihood for model", descriptions[i], " is ", likelihood[i])
#        print("(FRB pixel value was {0}, brightest pixel in this image was {1})".format(imagedata[int(np.round(frbrapix)), int(np.round(frbdecpix))], np.max(imagedata)))
#    
#    improvement = 2*(np.log(likelihood[2]) - np.log(likelihood[0]))
#    print("Model based on spiral residuals is {0} sigma better than just the starlight itself".format(improvement))
#
#    #plt.imshow(np.log(rawimagedata))
#    #plt.show()
#    #plt.imshow(np.log(rawimagedata * (1-maskdata)))
#    #plt.show()

