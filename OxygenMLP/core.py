import joblib
import numpy as np
import sklearn
from os import path

here = path.abspath(path.dirname(__file__))

class OxygenMLP:
	'''
	This is a class for using the neural network model to predict oxygen abundance.
	The class should first be created by 
	
	> import OxygenMLP
	> oxygenClass = OxygenMLP.OxygenMLP()

	The line flux ratios are then fed to the class via 
	> 	oxygenClass.ingestLines(o2,o3,n2,s2)
	
	where o2, o3, n2, s2 are numpy arrays of the line fluxes normalized to H-beta
	o2 = ([OII]3726+[OII]3729) / H-beta
	o3 = [OIII]5007 / H-beta
	n2 = [OIII]6583 / H-beta
	s2 = ([SII]6716+[SII]6731) / H-beta	

	All the line ratios need to be there. NaN is not accepted (will return NaN for oxygen abundance). 

	The prediction of oxygen abundance can be done by simply:
	> z, z_err = oxygenClass.predictZ()

	
	'''
	def __init__(self):
		self.pipes = joblib.load(here + '/bestModels.pkl')
		self.featureList = ['self.o2','self.o3','self.n2','self.s2',
		                     'self.o3/self.o2','self.n2/self.o2','self.s2/self.o2',
		                     'self.n2/self.o3','self.s2/self.o3','self.s2/self.n2',
		                     'self.o2+self.o3','self.o2+self.n2','self.o2+self.s2','self.o3+self.n2','self.o3+self.s2','self.n2+self.s2']
		
	def ingestLines(self,o2, o3, n2, s2):
		self.o2 = o2
		self.o3 = o3
		self.n2 = n2
		self.s2 = s2
		self.X = self.getX()
		self.y, self.y_err = self.predictZ()


	def getX(self):
		X  = np.zeros((self.o2.shape[0],len(self.featureList)))
		cmd = 'X[:] = np.stack((np.log10(' + "),np.log10(".join(self.featureList) + ')),axis=-1)[:]'
		exec(cmd)
		return X

	def predictZ(self):
		# Dig out NaNs
		XFlat = np.sum(self.X,axis=1)
		indGood = np.where(np.isfinite(XFlat))
		# make output array
		predict = np.zeros((len(self.pipes),self.X.shape[0])) + np.nan

		# fill in predictions from different pipe
		for i,pipe in enumerate(self.pipes):
			predict[i,indGood] = pipe.predict(self.X[indGood])
		# return median and std
		return np.nanmedian(predict,axis=0),np.nanstd(predict,axis=0)
