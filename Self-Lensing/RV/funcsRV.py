import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import csv


def initrange(p):
	"""
	Return initial error estimates in each parameter.
	Used to start the MCMC chains in a small ball near an estimated solution.

	Input
	-----
	p : ndarray
		Model parameters. See light_curve_model for the order.


	Returns
	-------
	errs : ndarray
		The standard deviation to use in each parameter
		for MCMC walker initialization.
	"""

	#period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma, jitter
	errorEst = [0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001]
	errorEst = np.array(errorEst)

	return errorEst



def kepler(M, e):
	"""
	Simple Kepler solver.
	Iterative solution via Newton's method. Could likely be sped up,
	but this works for now; it's not the major roadblock in the code.

	Input
	-----
	M : ndarray
	e : float or ndarray of same size as M

	Returns
	-------
	E : ndarray
	"""

	M = np.array(M)
	E = M * 1.
	err = M * 0. + 1.

	while err.max() > 1e-8:
		#solve using Newton's method
		guess = E - (E - e * np.sin(E) - M) / (1. - e * np.cos(E))
		err = np.abs(guess - E)
		E = guess

	return E



def RV_model(t, period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma):
	"""
	Given the orbital parameters compute the RV at times t, without gamma

	Input
	-----
	t : ndarray
		Times to return the model RV.
	period : float [days]
	ttran : float [days]
	sqrte_cosomega : float
	sqrte_sinomega : float 
	K : float [km/s]
	gamma : float [km/s]

	Returns
	-------
	RV_model : ndarray
		RV corresponding to the times in t [km/s].

	"""




	e = (sqrte_cosomega**2.) + (sqrte_sinomega**2.)
	omega = np.arctan2(sqrte_sinomega, sqrte_cosomega)

	#mean motion: n = 2pi/period
	n = 2. * np.pi / period

	# Sudarsky 2005 Eq. 9 to convert between center of transit
	# and pericenter passage (tau)
	edif = 1. - e**2.
	fcen = np.pi/2. - omega
	tau = (ttran + np.sqrt(edif) * period / (2 * np.pi) * 
		  (e * np.sin(fcen) / (1. + e * np.cos(fcen)) - 2. / np.sqrt(edif) * 
		  np.arctan(np.sqrt(edif) * np.tan(fcen / 2.) / (1. + e))))


	#Define mean anomaly: M
	M = (n * (t - tau)) % (2. * np.pi)



	#Determine the Eccentric Anomaly: E
	E = kepler(M, e)

	#Solve for fanom (measure of location on orbit)
	tanf2 = np.sqrt((1. + e) / (1. - e)) * np.tan(E / 2.)
	fanom = (np.arctan(tanf2) * 2.) % (2. * np.pi)

	#Calculate RV at given location on orbit
	RV = K * (e * np.cos(omega) + np.cos(fanom + omega)) + gamma

	return RV



def loglikelihood(p, t, RV, RVerr, chisQ=False):
	"""
	Compute the log likelihood of a RV signal with these orbital
	parameters given the data. 
	
	Input
	-----
	p : ndarray
		Orbital parameters. See RV model for order
	t, RV, RVerr : MxNdarray
		times, RV, and RV errors of the data.
		arranged as a list of lists
		len(array) = number of observing devices

	chisQ : boolean, optional
        If True, we are trying to minimize the chi-square rather than
        maximize the likelihood. Default False.


		
	Returns
	------
	likeli : float
		Log likelihood that the model fits the data.
	"""

	# Define all parameters
	(period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma, jitter) = p



	# Check observations are consistent
	if ( len(t) != len(RV) or len(t) != len(RVerr) or len(RV) != len(RVerr) ):
		print("Error! Mismatched number of spectra!")

	# Compute RV model light curve for the first all spectra without gamma added
	model = RV_model(t, period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma)


	# compute loglikelihood for model
	# Eastman et al., 2013 equation 
	# Christiansen et al., 2017 sec. 3.2 eq. 1
	totchisq = np.sum((RV-model)**2. / ( RVerr**2. + jitter**2.) )	
	loglikelihood = -np.sum( 
			(RV-model)**2. / ( 2. * (RVerr**2. + jitter**2.) ) +
			np.log(np.sqrt(2. * np.pi * (RVerr**2. + jitter**2.)))
			)




	# If we want to minimize chisQ, return it now
	if chisQ:
		return totchisq
	
	# Else return log likelihood
	return loglikelihood



def logprior(p):
	"""
	Priors on the input parameters.

	Input
	-----
	p : ndarray
		Orbital parameters. RV_model for the order.
		
	Returns
	-------
	prior : float
		Log likelihood of this set of input parameters based on the
		priors.
	"""

	(period, ttran, sqrte_cosomega, sqrte_sinomega, K, gamma, jitter) = p

	e = (sqrte_cosomega**2.) + (sqrte_sinomega**2.)
	omega = np.arctan2(sqrte_sinomega, sqrte_cosomega)

	




	#If any parameters not physically possible, return negative infinity.
	if (period < 0. or K < 0. or e < 0. or e >= 1.):
		return -np.inf

	# Uniform prior on jitter between 0 and 1 km/s
	if (jitter < 0. or jitter > 1.):
		return -np.inf



	# otherwise return a uniform prior
	#uniform chisq
	totchisq = 0 
	return totchisq





def logprob(p, t, RV, RVerr):
	"""
	Get the log probability of the data given the priors and the model.
	See loglikeli for the input parameters.
	
	Returns
	-------
	prob : float
		Log likelihood of the model given the data and priors, up to a
		constant.
	"""
	lp = logprior(p)
	llike = loglikelihood(p, t, RV, RVerr)


	if not np.isfinite(lp):
		return -np.inf

	if not np.isfinite(llike):
		return -np.inf

	return lp + llike 


