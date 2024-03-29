import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
    
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
    
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
    
        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
              [labels[:, :, self.column_indices[name]] for name in self.label_columns],
              axis=-1)
        
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def plot_train(self, plot_col, model=None, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                   label='Inputs', marker='.', zorder=-10)
        
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
        
            if label_col_index is None:
                continue
        
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                      edgecolors='orange', label='Labels', c='orange', s=50)
            if model is not None:
                predictions = model(inputs)
                plt.plot(self.label_indices, predictions[n, :, label_col_index],
                        marker='*', label='Predictions',
                        c='red')
        
            if n == 0:
                plt.legend()
        
        plt.xlabel('Time [d]')
        plt.show()
        

    def plot_test(self, plot_col, input_data, indices, model=None):
        inputs, labels = input_data
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(indices[-(self.input_width+self.shift):-self.shift], inputs[0, :, plot_col_index],
               label='Inputs', marker='.', zorder=-10)
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index
        plt.plot(indices[-self.label_width:], labels[0, :, label_col_index],
                      marker='*', label='Labels', c='orange')
        if model is not None:
            predictions = model(inputs)
            plt.plot(indices[-self.label_width:], predictions[0, :, label_col_index],
                    marker='*', label='Predictions',
                    c='red')
        plt.xlabel('Time [d]')
        return plt
    
    def plot_prediction(self, plot_col, input_data, indices, y_interval):
        inputs, predictions = input_data
        plt.rcParams['font.size'] = 20
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        plt.ylabel(f'{plot_col}')
        plt.plot(indices[-(self.input_width+self.shift):-self.shift], inputs[0, :, plot_col_index],
               label='Inputs', marker='.', c='green', zorder=-10)
        plt.ylim(y_interval)
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index
        plt.plot(indices[-self.label_width:], predictions[0, :, label_col_index],
                      marker='*', label='Predictions', c='red')
        plt.legend()
        plt.xlabel('Time [d]')
        return plt
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        
        ds = ds.map(self.split_window)
        
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    