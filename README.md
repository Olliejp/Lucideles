# Lucideles

This is a working repo for the Lucideles project undertaken at Idiap. The objective of this project is to use machine learning as a surrogate model for physics based models in the application of autonomous blind and lighting control to optimise visual comfort and energy efficiency in smart buildings. 

## Structure

```
├── Project 1/ # Initial experiment at the Idiap building in Martigny, single office with a single window
├── Project 2/ # More complex experiment for a test cell at the University of Firbourg with two windows
```

## Project 1
A collection of jupyter notebooks highlighing model training and data analysis for a single room case. Research was presented as a conference paper at BS21 in Bruges. [Link to paper.](https://olliejp.github.io/ML_building_control_idiap.pdf)

## Project 2
Project 2 is a more complex case with a larger room and two windows, orientated north and south
```
├── Room 1/
	├── inputs/ # data inputs to ML models. Most data has been removed for privacy and size
	├── outputs/ # outputs of machine learning models, inclusing indexes for reversing shuffling
	├── project_notebooks/ # any notebooks used outside of main src folder
	├── src/ # the source directory for the project machine learning pipeline
```
## SRC files description 
- Dockerfile 
	- A dockerfile with image directory for rasberry pi and python 3.6
- config.py
	- File directories and config data for model training
- deployment.py
	- python class to be deployed to rasberry pi for WPI and DGP predictions in real time
- engine.py
	- pytorch training enigine class. Handles training and evaluation and defines linear and MLP models. 
- make_data.py
	- a collection of modules to handle all data processing tasks, including preparing raw RADIANCE simulation data to clean data for model training, random shuffling and cyclical feature transformation
- make_folds.py
	- function to compute bins and create folds for stratified K-folds
- modules.py
	- helper functions, incliding standardisation and metric calculations
- train_linear.py
	- script to run training for linear regression in pytorch
- train_mlp.py
	- script to run training for neural network in pytorch
- train_xgboost.py
	- script to run training for xgboost
