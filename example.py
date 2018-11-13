import numpy as np
import OxygenMLP

if __name__ == '__main__':
	# Read in test data
	r, o2, o3, n2, s2 = np.loadtxt('data.txt',usecols=(0,1,2,3,4),unpack=True)

	# Apply MLP model	
	# Step 1: Load class
	oxygenClass = OxygenMLP.OxygenMLP()
	# Step 2: Send data into class
	oxygenClass.ingestLines(o2,o3,n2,s2)
	# Step 3: get predictions
	z, z_err = oxygenClass.predictZ()
	'''
	z is 12 + log(O/H) (
	z_err is the standard deviation of the 100 predictions (from the 100 models; see paper)
	z is the median of the 100 predictionsx
	'''
	