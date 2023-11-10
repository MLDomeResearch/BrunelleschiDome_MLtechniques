This is a reduced version of the source code used for dimensionality reduction task in Brunelleschi's Dome structural analysis. 
This code works only with the compact dataset (dome_data_compact.csv) which is attached in this folder.
The dome_data_compact.csv dataset contains 122 daily measures sampled from a single year (2014).
For the forecasting task we provide the fitted models trained on the whole time series so you can use them to make predictions on the reduced dataset.
You can run the code following these instructions:

INSTALLATION:
install the required packages via pip: 
$ pip install -r requirements.txt


USAGE:
- $ python main.py --dimred kpca -> Perform KPCA dimensionality reduction on the compact dataset with plot
- $ python main.py --dimred isomap -> Perform Isomap dimensionality reduction on the compact dataset with plot
- $ python main.py --dimred tsne -> Perform tsne dimensionality reduction on the compact dataset with plot

- $ python main.py --forecast 1 -> forecasting task on the web mnumber 1 - the models are trained on the whole dataset but the inference is on the compact dataset attached
- $ python main.py --forecast 2 -> forecasting task on the web mnumber 2 - the models are trained on the whole dataset but the inference is on the compact dataset attached
...
