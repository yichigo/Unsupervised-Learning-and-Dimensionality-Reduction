# Unsupervised Learning and Dimensionality Reduction


##  Program Language:
	python3
	
## Packages:
	pandas,
	numpy,
	matplotlib,
	sklearn,
	pickle,
	plots.py (to plot roc curve)


## Input and Output Path
the input data and preprocessed data are saved in ./data/

the output plots are exported to ./plots/


## Running

To run the codes, go to the folder ./src/,


### 1. Gestures Sensor Data Clustering and Dimensionality Reduction

Data preprocessing, in terminal, run:

	python gestures_data_preprocessing.py

Clustering, open Jupyter Notebook, run:

	gestures.ipynb


### 2. Heart Disease Data Clustering and Dimensionality Reduction

Data preprocessing, in terminal, run:

	python heart_data_preprocessing.py

Clustering, open Jupyter Notebook, run:

	heart.ipynb


### 3. Heart Disease Prediction

#### 3.1 Original Neural Network

In terminal, run:

	python NN.py


#### 3.2 Neural Network after Dimensionality Reduction ( without Cluster Label )

In terminal, run:

	python NN.py PCA

	python NN.py ICA

	python NN.py RP

	python NN.py TSNE


#### 3.2 Neural Network after Dimensionality Reduction ( add Cluster Label )

In terminal, run:

	python NN.py PCA KMeans

	python NN.py PCA EM

	python NN.py ICA KMeans

	python NN.py ICA EM

	python NN.py RP KMeans

	python NN.py RP EM

	python NN.py TSNE KMeans

	python NN.py TSNE EM

