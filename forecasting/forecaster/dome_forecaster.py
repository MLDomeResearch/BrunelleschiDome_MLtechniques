
import tensorflow as tf
from forecasting.forecaster.window_generator import WindowGenerator
import pandas as pd
import numpy as np

class DomeForecaster():
    def __init__(self, df, col_label, model_name="lstm_classic", OUT_STEPS=100, INPUT_WIDTH=300, MAX_EPOCHS=60):
        self.df=df
        self.df.set_index('date_time', inplace = True)
        self.col_label=col_label
        self.MAX_EPOCHS=MAX_EPOCHS
        self.OUT_STEPS=OUT_STEPS
        self.INPUT_WIDTH=INPUT_WIDTH
        self.n = len(df)
        self.num_features = df.shape[1]
        self.split_dataframe()
        self.standardScale_splitted_dataframe()
        self.create_multi_window()
        self.model=self.select_model(model_name)
        
    def __repr__(self):
        return '\n'.join([
            'DOMEFORECASTER',
            f'INPUT_WIDTH: {self.INPUT_WIDTH}',
            f'OUT_STEPS: {self.OUT_STEPS}',
            f'Dataframe: {self.df}',
            f'Train_set: {self.train_df}',
            f'Validation_set: {self.val_df}',
            f'Test_set: {self.test_df}',
            f'Dataframe length: {self.n}',
            f'Number of Features: {self.num_features}',
            f'Model: {self.model}',
            f'Max epochs: {self.MAX_EPOCHS}',
            f'WINDOW: {self.multi_window}'])

    #splitting in 70% train - 20% validation - 10% test
    def split_dataframe(self):
        self.train_df = self.df[0:int(self.n * 0.7)]
        self.val_df = self.df[int(self.n*0.7):int(self.n*0.9)]
        self.test_df = self.df[int(self.n*0.9):]

    #std dataframe
    def standardScale_splitted_dataframe(self):
        self.train_mean = self.train_df.mean()
        self.train_std = self.train_df.std()
        
        self.train_df = (self.train_df - self.train_mean) / self.train_std
        self.val_df = (self.val_df - self.train_mean) / self.train_std
        self.test_df = (self.test_df - self.train_mean) / self.train_std
        
    #sliding multi_window creation
    def create_multi_window(self):
        self.multi_window = WindowGenerator(input_width=self.INPUT_WIDTH,
                               label_width=self.OUT_STEPS,
                               shift=self.OUT_STEPS,
                               train_df=self.train_df, val_df=self.val_df, test_df=self.test_df)

    #select the model to train/evaluate
    def select_model(self, model_name):
        if (model_name == "lstm_bidirectional"):
            multi_lstm_model_bidirectional = tf.keras.Sequential([
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
                tf.keras.layers.Dense(self.OUT_STEPS*self.num_features,
                                    kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features])
            ])
            return multi_lstm_model_bidirectional
        elif (model_name == "GRU_bidirectional"):    
            multi_GRU_model_bidirectional = tf.keras.Sequential([
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=False)),
                tf.keras.layers.Dense(self.OUT_STEPS*self.num_features,
                                    kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features])
            ])
            return multi_GRU_model_bidirectional
        elif (model_name == "CONV_GRU_bidirectional"):
            multi_conv_gru_model_bidirectional = tf.keras.Sequential([
                tf.keras.layers.Reshape([self.INPUT_WIDTH, self.num_features]),
                tf.keras.layers.Conv1D(filters=40, kernel_size=6, strides=2, padding="valid", input_shape=[None, 1]),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(20)),
                tf.keras.layers.Dense(120, activation="relu"),
                tf.keras.layers.Dense(self.OUT_STEPS*self.num_features),
                tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features])
            ])
            return multi_conv_gru_model_bidirectional

        else:
            multi_lstm_model_classic = tf.keras.Sequential([
                tf.keras.layers.LSTM(16, return_sequences=False),
                tf.keras.layers.Dense(self.OUT_STEPS*self.num_features,
                                    kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features])
            ])
            return multi_lstm_model_classic
        
    def model_summary(self):
        return self.model.summary()

    def get_train_mean(self):
        return self.train_mean

    def get_train_std(self):
        return self.train_std

    #Compile and build the selected model
    def compile_and_build(self):
        self.model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        self.model.build((None, self.INPUT_WIDTH, self.num_features))
        
                
    #Compile and fite the selected model
    def compile_and_fit(self, patience=5):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')
        my_optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=0.001,
            rho=0.5,
            momentum=0.1,
            epsilon=1e-07,
            centered=True,
            name="RMSprop"
        )
        self.model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        # self.model.compile(loss=tf.losses.MeanSquaredError(),
        #               optimizer=tf.optimizers.Adam(),
        #               metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.model.fit(self.multi_window.train, epochs=self.MAX_EPOCHS,
                            validation_data=self.multi_window.val,
                            callbacks=[early_stopping])
        return history

    def evaluate_on_test_set (self, plot=True):
        input=self.test_df[-(self.INPUT_WIDTH+self.OUT_STEPS):-self.OUT_STEPS].to_numpy()
        labels=self.test_df.tail(self.OUT_STEPS).to_numpy()
        x = np.expand_dims(input, axis=0)
        y = np.expand_dims(labels, axis=0)
        indices =  self.test_df[-(self.INPUT_WIDTH+self.OUT_STEPS):].index
        if plot :
            figure = self.multi_window.plot_test(plot_col=self.col_label, input_data=(x,y), indices = indices, model=self.model)
            figure.show()
        
        return self.model.evaluate(x,y)
    
    def predict(self, start_day, plot=True, days_shift=1):
        input_df = self.df.loc[start_day - pd.Timedelta(self.INPUT_WIDTH*days_shift, unit="d"): start_day - pd.Timedelta(1*days_shift, unit="d")]
        input_df_std = (input_df - self.train_mean) / self.train_std 
        input_df_std = input_df_std.fillna(0)
        input = input_df_std.to_numpy()
        x = np.expand_dims(input, axis=0)
        predictions = self.model(x)
        predictions = predictions * self.train_std + self.train_mean
        x = np.expand_dims(input_df.to_numpy(), axis=0)
        indices =  self.df.loc[start_day - pd.Timedelta(self.INPUT_WIDTH*days_shift, unit="d"): start_day + pd.Timedelta(self.OUT_STEPS*days_shift-1, unit="d")].index
        if plot :
            y_interval = [-1,1]
            if int(self.col_label[4:5]) in {2,4,6,8}:
                 y_interval = [-3.5,2.5]      
            figure = self.multi_window.plot_prediction(plot_col=self.col_label, input_data=(x,predictions), indices = indices.to_numpy(), y_interval=y_interval)
            return predictions, indices, figure
        return predictions, indices 

    def save_forecaster(self, name):
        self.model.save('models/{mname}'.format(mname=name))

    def load_forecaster(self, name):
        self.model = tf.keras.models.load_model('models/{mname}'.format(mname=name))
