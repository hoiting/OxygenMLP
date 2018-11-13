from sklearn.externals import joblib
import numpy as np
import sklearn

class OxygenMLP:
	def __init__(self):
		self.pipes = joblib.load('best_models.pkl')
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
