#!/usr/bin/env python
import os, sys, copy
import glob
import shutil
import argparse
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy import units as u
#from astropy.modeling.models import Ellipse2D
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize,ZScaleInterval,LinearStretch,simple_norm
from skimage import filters

class FRBHost:
    def __init__(self, hostimagefile, verbose, log, zoompixels):
        self.hostimagefile = hostimagefile
        self.verbose = verbose
        self.log = log
        self.descriptions = ["Un-modified image", "Smoothed profile", "Residual from smoothed fit", "Mask"]
        self.shortdescriptions = ["Original", "Profile", "Residual", "Mask"]

        # Open up the FITS image of host + model and check it
        self.hostimagehdul = fits.open(self.hostimagefile)
        if len(self.hostimagehdul) != 4:
            print("FITS image file must have 4 HDUs:")
            print(descriptions)
            sys.exit()

        # Try and guess a name
        self.name = self.hostimagefile.split('/')[-1].split('_')[0]
        self.filter = self.hostimagefile.split('/')[-1].split('_')[1]

        # Estimate and remove the baseline
        self.baseline = self.hostimagehdul[1].data.min()
        if self.verbose:
            print("Minimum from original image is ", self.hostimagehdul[0].data.min(), ", from profile is", self.hostimagehdul[1].data.min())

        # Get the WCS
        self.w = wcs.WCS(self.hostimagehdul[1].header, self.hostimagehdul)

        # log
        if self.log:
            with open('{0}_{1}_log.txt'.format(self.name, self.filter), 'w') as f:
                f.write('|---------------------------------|\n')
                f.write('| Log of Model Likelihood Testing |\n')
                f.write('|---------------------------------|\n')
                f.write('\n')
                f.write('Doing analysis on '+str(self.name)+' '+str(self.filter)+' image\n')
                f.write('Minimum value of original image is '+str(self.hostimagehdul[0].data.min())+'\n')
                f.write('Minimum value of Sersic profile is '+str(self.hostimagehdul[1].data.min())+'\n')
                f.close()

        # Save the zoompixels
        if not zoompixels == '':
            self.zoompixels = [int(x) for x in zoompixels.split(',')]
            if not len(self.zoompixels) == 4:
                raise ValueError("Zoompixels must have 4 entries")
        else:
            self.zoompixels = []

    def growMask(self, expandmaskby, extramaskpixelstring, galaxymaskstring):
        self.originalmaskdata = self.hostimagehdul[self.shortdescriptions.index('Mask')].data
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

        # Log what we did
        if self.log==True:
            with open('{0}_{1}_log.txt'.format(self.name, self.filter), 'a') as f:
                f.write('Growing mask by '+str(expandmaskby)+' pixels in each direction\n')
                f.close()

        # Add any extra mask pixels
        extramaskpixels = extramaskpixelstring.split(',')
        for e in extramaskpixels:
            splite = e.split(':')
            if not len(splite) == 2:
                print("Mal-formatted extra mask pixel", e, '- ignoring')
                continue
            self.maskdata[int(splite[1])][int(splite[0])] = 1

        # Log what we did
        if self.log==True:
            with open('{0}_{1}_log.txt'.format(self.name, self.filter), 'a') as f:
                f.write('Adding the following extrapixels to the mask: {0}\n'.format(extramaskpixelstring))
                f.close()

        # Mask outside a given ellipse representing the galaxy
        if galaxymaskstring != "":
            galaxymaskparams = galaxymaskstring.split(',')
            if not len(galaxymaskparams) == 3:
                print("Mal-formatted galaxy mask string", galaxymaskstring, '- ignoring')
            else:
                galaxymaj = float(galaxymaskparams[0])
                galaxymin = float(galaxymaskparams[1])
                galaxypa  = float(galaxymaskparams[2])*np.pi/180. # Convert degrees to radians
                pixcrd = np.array([[1, 1], [2, 2]])
                world = self.w.wcs_pix2world(pixcrd, 0)
                rapixscale = (world[0,0] - world[1,0])*3600*np.cos(world[1,1]*np.pi/180)*u.arcsec
                decpixscale = (world[1,1] - world[0,1])*3600*u.arcsec
                centrerapix = self.hostimagehdul[1].data.shape[0]//2
                centredecpix = self.hostimagehdul[1].data.shape[1]//2
                for i in range(self.hostimagehdul[1].data.shape[0]):
                    for j in range(self.hostimagehdul[1].data.shape[1]):
                        raoffset = (i - centrerapix)*rapixscale.value
                        decoffset = (j - centredecpix)*decpixscale.value
                        majoffset = decoffset*np.cos(-galaxypa) + raoffset*np.sin(-galaxypa)
                        minoffset = raoffset*np.cos(-galaxypa) - decoffset*np.sin(-galaxypa)
                        if (majoffset/galaxymaj)**2 + (minoffset/galaxymin)**2 >= 1:
                            self.maskdata[j,i] = 1

        # Log what we did
        if self.log==True:
            with open('{0}_{1}_log.txt'.format(self.name, self.filter), 'a') as f:
                f.write('Masking region outside {0}\n'.format(galaxymaskstring))
                f.close()                

        # Save a copy of the final mask again (not sure a copy of the final mask again (not sure why...))
        self.originalmaskdata = self.hostimagehdul[3].data

    def addEdgeImage(self, parentdescription):
        # Get the index of the parent image and mask image
        parentindex = self.shortdescriptions.index(parentdescription)
        maskindex = self.shortdescriptions.index('Mask')

        # Append a new Image HDU to the end of the HDU list
        self.hostimagehdul.append(self.hostimagehdul[1].copy())

        # Replace the data with an edge filtered view of the residual
        self.hostimagehdul[-1].data = np.abs(filters.roberts(self.hostimagehdul[parentindex].data * (1 - self.hostimagehdul[maskindex].data)))

        # Update the descriptions
        self.descriptions.append("Residual Edge Image")
        self.shortdescriptions.append("ResidualEdge")

    def addFRBLocalisation(self, frbradeg, febdecdeg, frbmajoraxisarcsec, frbminoraxisarcsec, frbpadeg):
        # Store the information about the FRB localisation
        self.frbradeg = frbradeg
        self.frbdecdeg = febdecdeg
        #self.sigmaraarcsec = frbsigmaraarcsec
        #self.sigmadecarcsec = frbsigmadecarcsec
        #self.frbtheta = 0 # placeholder for plotting FRB ellipse
        self.frbmajoraxisarcsec = frbmajoraxisarcsec
        self.frbminoraxisarcsec = frbminoraxisarcsec
        self.frbparad = frbpadeg*np.pi/180.0

        if self.log==True:
            with open('{0}_{1}_log.txt'.format(self.name, self.filter), 'a') as f:
                f.write('-----------------------------------------\n')
                f.write('User-input FRB information:\n')
                f.write('FRB R.A.: '+str(self.frbradeg)+' degrees\n')
                f.write('FRB Decl.: '+str(self.frbdecdeg)+' degrees\n')
                #f.write('FRB R.A. uncertainty: '+str(self.sigmaraarcsec)+' arcsecs\n')
                #f.write('FRB Decl. uncertainty: '+str(self.sigmadecarcsec)+' arcsecs\n')
                #f.write('FRB uncertainty angle: '+str(self.frbtheta)+' deg E of N\n')
                f.write('FRB Uncertainty semi major axis: ' + str(self.frbmajoraxisarcsec)+' arcsecs\n')
                f.write('FRB Uncertainty semi minor axis: ' + str(self.frbminoraxisarcsec)+' arcsecs\n')
                f.write('FRB uncertainty angle: '+str(frbpadeg)+' deg E of N\n')
                f.write('-----------------------------------------\n')
                f.close()

        # Now append a new Image HDU to the end of the HDU list
        self.hostimagehdul.append(self.hostimagehdul[1].copy())

        # Make a deep copy of the data
        self.hostimagehdul[-1].data = copy.deepcopy(self.hostimagehdul[-1].data)

        # Update this new Image HDU to contain the FRB likelihood
        self.hostimagehdul[-1].data -= self.hostimagehdul[-1].data # Set it to zero initially
        oversamplefactor = 5
        subpixgrid = np.zeros(oversamplefactor*oversamplefactor*2).reshape(oversamplefactor*oversamplefactor, 2)
        for i in range(oversamplefactor):
            for j in range(oversamplefactor):
                subpixgrid[i*oversamplefactor + j][0] = (i-oversamplefactor//2)/float(oversamplefactor)
                subpixgrid[i*oversamplefactor + j][1] = (j-oversamplefactor//2)/float(oversamplefactor)
        #subpixgrid = []
        #for i in range(oversamplefactor):
        #    for j in range(oversamplefactor):
        #        subpixgrid.append([(i+oversamplefactor/2.0)/4.0, (j+oversamplefactor/2.0)/4.0])
        for x in range(self.hostimagehdul[-1].data.shape[0]):
            for y in range(self.hostimagehdul[-1].data.shape[1]):
                pixgrid = [[l[0]+x, l[1]+y] for l in subpixgrid]
                radecgrid = self.w.all_pix2world(pixgrid, 0)
                radecoffsets = [[3600*(g[0]-self.frbradeg)*np.cos(self.frbdecdeg*np.pi/180), 3600*(g[1]-self.frbdecdeg)] for g in radecgrid]
                majminoffsets = [[o[0]*np.sin(self.frbparad) + o[1]*np.cos(self.frbparad), -o[0]*np.cos(self.frbparad) + o[1]*np.sin(self.frbparad)] for o in radecoffsets]
                #norm = 1/(2*np.pi*self.sigmaraarcsec*self.sigmadecarcsec)
                norm = 1/(2*np.pi*self.frbmajoraxisarcsec*self.frbminoraxisarcsec)
                #likelihoods = [norm*np.e**(-(o[0])**2/(2*self.sigmaraarcsec**2)) * np.e**(-(o[1])**2/(2*self.sigmadecarcsec**2)) for o in radecoffsets]
                likelihoods = [norm*np.e**(-(o[0])**2/(2*self.frbmajoraxisarcsec**2)) * np.e**(-(o[1])**2/(2*self.frbminoraxisarcsec**2)) for o in majminoffsets]
                self.hostimagehdul[-1].data[y, x] = np.mean(likelihoods)

        # Normalise the FRB localisation probability density
        imsum = self.hostimagehdul[-1].data.sum()
        self.hostimagehdul[-1].data /= imsum

        # Update the descriptions
        self.descriptions.append("FRB Localisation Likelihood")
        self.shortdescriptions.append("Localisation")

        # Create a space to store likelihoods
        #self.nummodels = len(self.hostimagehdul) - 2
        self.modellikelihoods = np.zeros(len(self.hostimagehdul))

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

            if self.log==True:
                with open('{0}_{1}_log.txt'.format(self.name, self.filter), 'a') as f:
                    f.write('Sum of '+str(self.shortdescriptions[i])+' image: '+str(imsum)+'\n')
                    f.close()

            # Calculate the log likelihood for this model
            self.modellikelihoods[i] = (imagedata * self.hostimagehdul[self.shortdescriptions.index('Localisation')].data).sum()

    def printReport(self):
        for i in range(len(self.descriptions)):
            if self.shortdescriptions[i] == "Mask" or self.shortdescriptions[i] == "Localisation":
                continue # Don't need to evaluate anything for this image
            print("For model based on {0}, the likelihood is {1}".format(self.descriptions[i], self.modellikelihoods[i]))

            if self.log==True:
                with open('{0}_{1}_log.txt'.format(self.name, self.filter), 'a') as f:
                    f.write('For model based on '+str(self.descriptions[i])+', the likelihood is '+str(self.modellikelihoods[i])+'\n')
                    f.close()

    def plotAll(self):
        # Get an appropriate wcs (zoomed in if necessary)
        if len(self.zoompixels) > 1:
            zoomwcs = self.w[self.zoompixels[0]:self.zoompixels[2], self.zoompixels[1]:self.zoompixels[3]]
        else:
            zoomwcs = self.w
        for i in range(len(self.descriptions)):
            # get the appropriate subset of data
            if len(self.zoompixels) > 1:
                zoomdata = self.hostimagehdul[i].data[self.zoompixels[0]:self.zoompixels[2], self.zoompixels[1]:self.zoompixels[3]]
            else:
                zoomdata = self.hostimagehdul[i].data

            # Set up the axes
            ax = plt.subplot(projection=zoomwcs)
            ax.coords[0].set_axislabel('Right Ascension (J2000)')
            ax.coords[1].set_axislabel('Declination (J2000)')

            # Normalise and plot
            #norm = simple_norm(zoomdata, 'linear', percent=99.0)
            asinh_a_factor = 0.15
            percent_factor = 100.0
            if self.shortdescriptions[i] == "Original" or self.shortdescriptions[i] == "Profile":
                asinh_a_factor = 0.015
                percent_factor = 99.99
            norm = simple_norm(zoomdata, 'asinh', asinh_a=asinh_a_factor, percent=percent_factor)
            ax.imshow(zoomdata, norm=norm) #, vmin=-2.e-5, vmax=2.e-4, origin='lower')

            # Plot the FRB localisation
            frb = Ellipse((self.frbradeg, self.frbdecdeg), width=(self.frbmajoraxisarcsec/3600)*2, height=(self.frbminoraxisarcsec/3600)*2, angle=90-self.frbparad*180/np.pi, edgecolor='white', facecolor='none', lw=2, transform=ax.get_transform('icrs'))
            ax.add_patch(frb)

            # Save figure
            plt.savefig(self.name + "-" + self.shortdescriptions[i] + ".png")
            plt.clf()

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
    parser.add_argument('--rauncarcsec', type=float, default=-1.0, help='Uncertainty in FRB RA (arcseconds)')
    parser.add_argument('--decuncarcsec', type=float, default=-1.0, help='Uncertainty in FRB Dec (arcseconds)')
    parser.add_argument('--frbuncmajorarcsec', type=float, default=-1.0, help='Major axis of FRB uncertainty (arcseconds)')
    parser.add_argument('--frbuncminorarcsec', type=float, default=-1.0, help='Minor axis of FRB uncertainty (arcseconds)')
    parser.add_argument('--frbuncpa', type=float, default=-1.0, help='Position angle of FRB uncertainty (east of north, degrees)')
    parser.add_argument('--zoompixels', type=str, default="", help='Pixels to zoom in on in form blc,brc,tlc,trc')
    parser.add_argument('--extramaskpixels', type=str, default="", help='Colon-separated list of pixels to mask, e.g. 50,50:50,51')
    parser.add_argument('--galaxymask', type=str, default="", help="Major,minor,PAdegrees - mask outside this region (centred on centre of image)")
    parser.add_argument('--dostretch', default=False, action='store_true',
                        help='Apply a stretch to the host galaxy residual image')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--log', default=True, action='store_true')
    args = parser.parse_args()

    # Check that only one type of error ellipse was provided
    if args.rauncarcsec > 0:
        if args.decuncarcsec < 0.0:
            parser.error("If RA uncertainty is provided, Dec uncertainty must also be provided (and error ellipse cannot be provided")
        elif args.frbuncmajorarcsec >= 0 or args.frbuncminorarcsec >= 0:
            parser.error("Cannot supply both RA/dec uncertainties and an uncertainty ellipse via frbuncmajorarcsec/frbuncminorarcsec/frbuncpa!")
        else: # Uncertainty is correctly specified via RA/Dec - convert to maj/min/pa
            if args.rauncarcsec > args.decuncarcsec:
                args.frbuncmajorarcsec = args.rauncarcsec
                args.frbuncminorarcsec = args.decuncarcsec
                args.frbuncpa = 90.0
            else:
                args.frbuncmajorarcsec = args.decuncarcsec
                args.frbuncminorarcsec = args.rauncarcsec
                args.frbuncpa = 0.0
    elif args.frbuncmajorarcsec <= 0 or args.frbuncminorarcsec < 0:
        parser.error("Must supply either RA/Dec uncertainty, or an uncertainty ellipse via frbuncmajorarcsec/frbuncminorarcsec/frbuncpa")

    # Check that we actually got a filename to work with
    if args.hostimagefile == '':
        parser.error('You must supply a FITS filename to hostimagefile')

    # Create FRBHost object
    frbhost = FRBHost(args.hostimagefile, args.verbose, args.log, args.zoompixels)

    # Expand the mask if desired
    frbhost.growMask(args.growmaskpixels, args.extramaskpixels, args.galaxymask)

    # Add an edge image
    frbhost.addEdgeImage("Residual")

    # Add the information about the FRB localisation
    frbhost.addFRBLocalisation(args.frbra, args.frbdec, args.frbuncmajorarcsec, args.frbuncminorarcsec, args.frbuncpa)

    # Calculate the likelihoods for different models and make plots
    frbhost.evaluateModels(args.dostretch)
    frbhost.printReport()
    frbhost.plotAll()

    # clean up working directory
    cwd = os.getcwd()

    if not os.path.exists('Log_Files'):
        os.makedirs('Log_Files')
    
    if not os.path.exists('Figures'):
        os.makedirs('Figures')

    for file in glob.glob('*.txt'):
        shutil.move(os.path.join(cwd, file), os.path.join(cwd+'/Log_Files', file))
    
    for file in glob.glob('*.png'):
        shutil.move(os.path.join(cwd, file), os.path.join(cwd+'/Figures', file))

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

