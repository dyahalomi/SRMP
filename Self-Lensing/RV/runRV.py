import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from funcsRV import *
import csv




def runMCMC(p, t, rv, rvErr, outfile, niter=10000, nwalkers=50):

	"""
	Run the MCMC Orbital fit to Spectroscopic RV Observations

	Input
	-----
	p : ndarray
		Orbital parameters. See RV model in funcs.py for order
	t, rv, rvErr : MxNdarray
		times, RV, and RV errors of the data.
		arranged as a list of lists
		len(array) = number of observing devices
	outfile : string
		name of output file where MCMC chain is stored
	niter : int, optional
        number of MCMC iterations to run. Default = 10,000
    nwalkers : int, optional
        number of MCMC walkers in modeling. Default = 50
		
	Returns
	------
	String stating "MCMC complete"

	(Outputs MCMC chain into file labeled whatever input into variable: outfile)


	"""

	ndim = len(p)



	#start walkers in a ball near the optimal solution
	startlocs = [p + initrange(p) * np.random.randn(ndim) for i in np.arange(nwalkers)]

	#run emcee MCMC code
	sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args = [t, rv, rvErr])

	#clear output file
	ofile = open(outfile, 'w')
	ofile.close()

	iternum = 1
	#run the MCMC...record parameters for every walker at every step
	for result in sampler.sample(startlocs, iterations = niter, store = False):
		pos = result.coords
		ofile = open(outfile, 'a')

		#write iteration number, walker number, and log likelihood
		#and value of parameters for the step
		for walker in np.arange(pos.shape[0]):
			ofile.write('{0} {1} {2} {3}\n'.format(iternum, walker, str(result.log_prob[walker]), " ".join([str(x) for x in pos[walker]])))

		ofile.close()


		#keep track of step number
		mod = iternum % 1000
		if mod == 0:
			print(iternum)
			print(pos[0])

		iternum += 1
	return "MCMC complete"










#set first guess of parameters for modeling
#p = (period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma, jitter)



'''
t_Outlier, rv_Outlier, rvErr_Outlier = readObservations('./KOI315_Outlier.txt', False)
t_Outlier = [t_Outlier]
rv_Outlier = [rv_Outlier/1000]
rvErr_Outlier = [rvErr_Outlier/1000]
print runMCMC(p, t_Outlier, rv_Outlier, rvErr_Outlier, './chain_100000_Outlier_FixedEcc.txt', niter = 100000, nwalkers = 50)
'''






#t, rv, rvErr = readObservations('./KOI315_noOutlier.txt', False)
#t = t - bjd_t0




#print runMCMC(p, t, rv, rvErr, './chain_10000_nojitter+Outlier_sep20.txt', niter = 10000, nwalkers = 50)




