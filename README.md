This is the source code used for dimensionality reduction and forecasting tasks described in Brunelleschi's Dome structural analysis paper. 
This code works only with the compact dataset (dome_data_compact.csv) which is attached in this folder.
The dome_data_compact.csv dataset contains 122 daily measures sampled from a single year (2014).
For the forecasting task we provide the fitted models trained on the whole time series so you can use them to make predictions on the reduced dataset.
The accuracy of the forecasting is lower than what reported in the paper, because the provided compact dataset is not composed of contiguous temporal points.
You can run the code following these instructions:

INSTALLATION:
install the required packages via pip: 
$ pip install -r requirements.txt


USAGE:
- $ python main.py --dimred kpca -> Perform KPCA dimensionality reduction on the compact dataset with plot
- $ python main.py --dimred isomap -> Perform Isomap dimensionality reduction on the compact dataset with plot
- $ python main.py --dimred tsne -> Perform tsne dimensionality reduction on the compact dataset with plot

- $ python main.py --forecast 1 -> forecasting task on the web number 1 - the models are trained on the whole dataset but the inference is on the compact dataset attached
- $ python main.py --forecast 2 -> forecasting task on the web number 2 - the models are trained on the whole dataset but the inference is on the compact dataset attached
...


The animation_kpca_isomap.zip file contains the animated graph videos related to the dimensionality reduction task
