"""
Analyze the results of an MCMC run.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from funcsRV import *
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import csv

def analyze_rv_chains(infile):    
    """
    Analyze the results of an MCMC run.
    """
    labels = ['$P$ [days]', '$t_{tran}$ [BJD - 2454833]', '$\sqrt{e} cos\omega$', '$\sqrt{e} sin\omega$', 
    '$K$ [km/s]', '$\gamma [km/s]$', '$\sigma_{j} [km/s]$']

    # after the burn in, only use every thin amount for speed
    nthin = 1


    foldername = './plots/'
    cornerFigname = 'corner_' + infile[17:-4] + '.png'
    chainFigname = 'chainPlot_'+ infile[17:-4] + '.png'


    # iteration where burn-in stops
    burnin = 2000
    # make the triangle plot
    maketriangle = True

    #########################

    nparams = 7

    x = np.loadtxt(infile)
    print('File loaded')

    # split the metadata from the chain results
    iteration = x[:, 0]
    walkers = x[:, 1]
    uwalkers = np.unique(walkers)
    loglike = x[:, 2]
    x = x[:, 3:]

    # thin the file if we want to speed things up
    thin = np.arange(0, iteration.max(), nthin)
    good = np.in1d(iteration, thin)
    x = x[good, :]
    iteration = iteration[good]
    walkers = walkers[good]
    loglike = loglike[good]




    # plot the value of each chain for each parameter as well as its log likelihood
    plt.figure(figsize = (24, 18))
    plt.clf()
    for ii in np.arange(nparams+1):
        # use 3 columns of plots
        ax = plt.subplot(int(np.ceil((nparams+1)/3.)), 3, ii+1)
        for jj in uwalkers:
            this = np.where(walkers == jj)[0]
            if ii < nparams:
                # if this chain is really long, cut down on plotting time by only
                # plotting every tenth element
                if len(iteration[this]) > 5000:
                    plt.plot(iteration[this][::10],
                             x[this, ii].reshape((-1,))[::10])
                else:
                    plt.plot(iteration[this], x[this, ii].reshape((-1,)))
            # plot the likelihood
            else:
                if len(iteration[this]) > 5000:
                    plt.plot(iteration[this][::10], loglike[this][::10])
                else:
                    plt.plot(iteration[this], loglike[this])
        # show the burnin location
        plt.plot([burnin, burnin], plt.ylim(), lw=2)
        # add the labels
        if ii < nparams:
            plt.ylabel(labels[ii])
        else:
            plt.ylabel('Log Likelihood')
            plt.xlabel('Iterations')
        ax.ticklabel_format(useOffset=False)
    plt.savefig(foldername + chainFigname)

    # now remove the burnin phase
    pastburn = np.where(iteration > burnin)[0]
    iteration = iteration[pastburn]
    walkers = walkers[pastburn]
    loglike = loglike[pastburn]
    x = x[pastburn, :]

    # sort the results by likelihood for the triangle plot
    lsort = np.argsort(loglike)
    lsort = lsort[::-1]
    iteration = iteration[lsort]
    walkers = walkers[lsort]
    loglike = loglike[lsort]
    x = x[lsort, :]

            

    if maketriangle:
        plt.figure(figsize = (18, 18))
        plt.clf()
        # set unrealistic default mins and maxes
        maxes = np.zeros(len(x[0, :])) - 9e9
        mins = np.zeros(len(x[0, :])) + 9e9
        nbins = 1
        # go through each combination of parameters
        for jj in np.arange(len(x[0, :])):
            for kk in np.arange(len(x[0, :])):
                # only handle each combination once
                if kk < jj:
                    # pick the right subplot
                    ax = plt.subplot(len(x[0, :]), len(x[0, :]),
                                     jj * len(x[0, :]) + kk + 1)
                    # 3, 2, and 1 sigma levels
                    sigmas = np.array([0.9973002, 0.9544997, 0.6826895])
                    # put each sample into 2D bins
                    hist2d, xedge, yedge = np.histogram2d(x[:, jj], x[:, kk],
                                                          bins=[nbins, nbins],
                                                          normed=False)
                    # convert the bins to frequency
                    hist2d /= len(x[:, jj])
                    flat = hist2d.flatten()
                    # get descending bin frequency
                    fargs = flat.argsort()[::-1]
                    flat = flat[fargs]
                    # cumulative fraction up to each bin
                    cums = np.cumsum(flat)
                    levels = []
                    # figure out where each sigma cutoff bin is
                    for ii in np.arange(len(sigmas)):
                            above = np.where(cums > sigmas[ii])[0][0]
                            levels.append(flat[above])
                    levels.append(1.)
                    # figure out the min and max range needed for this plot
                    # then see if this is beyond the range of previous plots.
                    # this is necessary so that we can have a common axis
                    # range for each row/column
                    above = np.where(hist2d > levels[0])
                    thismin = xedge[above[0]].min()
                    if thismin < mins[jj]:
                        mins[jj] = thismin
                    thismax = xedge[above[0]].max()
                    if thismax > maxes[jj]:
                        maxes[jj] = thismax
                    thismin = yedge[above[1]].min()
                    if thismin < mins[kk]:
                        mins[kk] = thismin
                    thismax = yedge[above[1]].max()
                    if thismax > maxes[kk]:
                        maxes[kk] = thismax
                    # make the contour plot for these two parameters
                    plt.contourf(yedge[1:]-np.diff(yedge)/2.,
                                 xedge[1:]-np.diff(xedge)/2., hist2d,
                                 levels=levels,
                                 colors=('k', '#444444', '#888888'))
                # plot the distribution of each parameter
                if jj == kk:
                    ax = plt.subplot(len(x[0, :]), len(x[0, :]),
                                     jj*len(x[0, :]) + kk + 1)
                    plt.hist(x[:, jj], bins=nbins, facecolor='k')

        # allow for some empty space on the sides
        diffs = maxes - mins
        mins -= 0.05*diffs
        maxes += 0.05*diffs
        # go back through each figure and clean it up
        for jj in np.arange(len(x[0, :])):
            for kk in np.arange(len(x[0, :])):
                if kk < jj or jj == kk:
                    ax = plt.subplot(len(x[0, :]), len(x[0, :]),
                                     jj*len(x[0, :]) + kk + 1)
                    # set the proper limits
                    if kk < jj:
                        ax.set_ylim(mins[jj], maxes[jj])
                    ax.set_xlim(mins[kk], maxes[kk])
                    # make sure tick labels don't overlap between subplots
                    ax.yaxis.set_major_locator(plticker.MaxNLocator(nbins=4,
                                                                    prune='both'))
                    # only show tick labels on the edges
                    if kk != 0 or jj == 0:
                        ax.set_yticklabels([])
                    else:
                        # tweak the formatting
                        plt.ylabel(labels[jj])
                        locs, labs = plt.yticks()
                        plt.setp(labs, rotation=0, va='center')
                        yformatter = plticker.ScalarFormatter(useOffset=False)
                        ax.yaxis.set_major_formatter(yformatter)
                    # do the same with the x-axis ticks
                    ax.xaxis.set_major_locator(plticker.MaxNLocator(nbins=4,
                                                                    prune='both'))
                    if jj != len(x[0, :])-1:
                        ax.set_xticklabels([])
                    else:
                        plt.xlabel(labels[kk])
                        locs, labs = plt.xticks()
                        plt.setp(labs, rotation=90, ha='center')
                        yformatter = plticker.ScalarFormatter(useOffset=False)
                        ax.xaxis.set_major_formatter(yformatter)
        # remove the space between plots
        plt.subplots_adjust(hspace=0.0, wspace=0.0)

        #save corner plot
        plt.savefig(foldername + cornerFigname)
        plt.show()


    # the best, median, and standard deviation of the input parameters
    # used to feed back to model_funcs for initrange, and plotting the best fit
    # model for publication figures in mcmc_run
    best = x[0, :]
    meds = np.median(x, axis=0)
    devs = np.std(x, axis=0)
    print('Best model parameters: ')
    print(best)

    print('Median model parameters: ')
    print(meds)
    
    return(best,meds,x)




def plot_RV(p, t, rv, rvErr, filename):
    '''
    Plot the RV data against RV model

    '''
    # Define all parameters except gamma and jitters
    (period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma, jitter) = p


    colors = [
    '#800000', '#9A5324', '#808000', '#469990', '#000075', '#e6194B', '#f58231', 
    '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6'
    ]

    plt.figure(figsize=(15,10))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.03)
    gs.update(hspace=0.)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    print(filename)
    print(filename[30:-11])

    ax0.errorbar(t, rv, yerr = np.sqrt(rvErr**2. + jitter**2.), fmt = 'o', color = colors[0],  markersize = 10, label = filename[30:-11])
    

    t_plot = np.arange(np.min(t)*0.95, np.max(t)*1.05)
    model = RV_model(t_plot, period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma)
    ax0.plot(t_plot, model, color = 'k')


    rv_model = RV_model(t, period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma)
    
    ax1.plot([np.min(t)*0.95, np.max(t)*1.05], [0., 0.], color = 'k')

    ax1.errorbar(t, rv - rv_model, yerr = np.sqrt(rvErr**2. + jitter**2.), fmt = 'o', markersize = 10,  color = colors[0])

   
    ax1.set_xlabel("Time [BJD - 2,454,833]", fontsize = 18)
    ax0.set_ylabel("Radial Velocity [m/s]", fontsize = 18)
    ax1.set_ylabel("Residuals [m/s]", fontsize = 18)
    yticks = ax0.yaxis.get_major_ticks()
    xticks = ax0.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    yticks[0].label1.set_visible(False)
    ax0.legend(numpoints = 1, loc = 2, fontsize = 18)


    plt.savefig(filename)
    plt.show()


def plot_foldedRV(p, t, rv, rvErr, filename):
    '''
    Plot the RV data against RV model folded

    '''
    # Define all parameters except gamma and jitters
    (period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma, jitter) = p


    colors = [
    '#800000', '#9A5324', '#808000', '#469990', '#000075', '#e6194B', '#f58231', 
    '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6'
    ]

    plt.figure(figsize=(15,10))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.03)
    gs.update(hspace=0.)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    phase_rv = ((t-p[1]) % p[0])/p[0]
    ax0.errorbar(phase_rv, rv, yerr=(np.sqrt(rvErr**2. + jitter**2.)), fmt='o', color = colors[0],  markersize = 10, label = filename[19:-11])

    
    tMod = np.arange(p[1], p[0] + p[1])
    model = RV_model(tMod, period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma)
    phase = ((tMod-p[1]) % p[0]) / p[0]
    lsort = np.argsort(phase)
    ax0.plot(phase[lsort], model[lsort], color = 'k')


    rv_model = RV_model(t, period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma)

    
    ax1.plot([0., 1.], [0., 0.], color = 'k')
    ax1.errorbar(phase_rv, rv - rv_model, yerr = np.sqrt(rvErr**2. + jitter**2.), fmt = 'o', markersize = 10,  color = colors[0])

   
    ax1.set_xlabel("Phase", fontsize = 18)
    ax0.set_ylabel("Radial Velocity [km/s]", fontsize = 18)
    ax1.set_ylabel("Residuals [km/s]", fontsize = 18)
    yticks = ax0.yaxis.get_major_ticks()
    xticks = ax0.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    yticks[0].label1.set_visible(False)
    ax0.legend(numpoints = 1, loc = 2, fontsize = 18)

    plt.savefig(filename)
    plt.show()





def get_RMS_residuals(p, t, rv, rvErr):
    '''
    p: input parameters
    the rest are observations
    '''
    # Define all parameters except gamma and jitters
    (period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma, jitter) = p


    rv_model = RV_model(t, period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma)

    n = len(t)

    rms = ( np.sum((rv - rv_model)**2) / n)

    rms = np.sqrt(rms)


    return rms










