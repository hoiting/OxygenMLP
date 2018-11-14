# OxygenMLP
A neural network, multi-layer perceptron model to predict oxygen abundance from strong lines. The model is calibrated to ~ 950 literature HII region spectra with auroral line detections. Details can be found in this paper [link]. 

### Dependencies
* Python 3
* scikit-learn
* numpy 


### Usage:
Check out the example script example.py that does the left panel of Fig. 10 in the paper


OxygenMLP is a class for using the neural network model to predict oxygen abundance.
The class should first be initialised by 
```python	
import OxygenMLP
oxygenClass = OxygenMLP.OxygenMLP()
```

The line flux ratios are then feed to the class via 
```python
oxygenClass.ingestLines(o2,o3,n2,s2)
```
where o2, o3, n2, s2 are numpy arrays of the line fluxes normalized to H-beta
o2 = (3727+3729) / H-beta
o3 = 5007 / H-beta
n2 = 6584 / H-beta
s2 = (6716+6731) / H-beta	

Note that all the line ratios need to be there. NaN is not accepted (will return NaN for oxygen abundance). 

The prediction of oxygen abundance can be done by simply:
```python
z, z_err = oxygenClass.predictZ()
```
	
