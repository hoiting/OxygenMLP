## OxygenMLP
*OxygenMLP* is a neural network, multi-layer perceptron model that can predict oxygen abundance using strong emission lines. The model is calibrated using ~ 950 literature HII region spectra with auroral line detections. The network is trained using [scikit-learn](http://scikit-learn.org/). Details can be found in this [paper](http://adsabs.harvard.edu/abs/2019MNRAS.485.3569H). 

#### Dependencies
* Python 3
* scikit-learn
* numpy 


#### What is what?
| Filename     |   Content    |
|--------------|-------|
| `OxygenMLP.py` | main python class |
| `bestModels.pkl` | neural network models | 
| `example.py`    | example script | 
| `data.txt`      | data used by example.py | 
| `inputdata.data`      | input catalog used for training and testing the model| 



#### Usage:
*Check out the example script example.py that does the left panel of Fig. 10 in the paper*

OxygenMLP is a class for using the neural network model to predict oxygen abundance.
The class should first be initialised by 
```python	
import OxygenMLP
oxygenClass = OxygenMLP.OxygenMLP()
```

The line flux ratios are then fed to the class via 
```python
oxygenClass.ingestLines(o2,o3,n2,s2)
```
where o2, o3, n2, s2 are numpy arrays of the extinction-corrected line fluxes normalized to H-beta

* o2 = ([OII]3726+[OII]3729) / H-beta
* o3 = [OIII]5007 / H-beta
* n2 = [NII]6583 / H-beta
* s2 = ([SII]6716+[SII]6731) / H-beta	

Note that all the line ratios need to be there. NaN is not accepted (will return NaN for oxygen abundance). 

The prediction of oxygen abundance can be done by simply:
```python
z, z_err = oxygenClass.predictZ()
```

Note: z_err is the standard deviation of the 100 predictions. If you want to propagate your measurement errors, you should do Monte Carlo simulations, i.e. create realizations of line fluxes, pass them through the model, and get the final z distribution (see Appendix in the paper). 

#### Contact:
For bug report and question, please write to [I-Ting Ho](iting@mpia.de)
