
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_absolute_error

from forecasting.forecaster.dome_forecaster import DomeForecaster

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
pd.set_option('display.max_columns', None)

#function to load the compact dataset and enrich it with seasons feature
def load_dataset() :
    df = pd.read_csv("dome_data_compact.csv", sep=",")

    df["datetime"] = pd.to_datetime(df["aaaammgg"], format='%Y%m%d')

    column_indices = {name: i for i, name in enumerate(df.columns)}

    conditions = [
        df.datetime.dt.dayofyear.isin(np.arange(1,81)),
        df.datetime.dt.dayofyear.isin(np.arange(81,173)),
        df.datetime.dt.dayofyear.isin(np.arange(173, 265)),
        df.datetime.dt.dayofyear.isin(np.arange(265,356)),
        df.datetime.dt.dayofyear.isin(np.arange(356,367))
    ]
    seasons = [1,2,3,4,1]
    df['seasons'] = np.select(conditions, seasons)

    #Seasons one hot encoding
    season_dummies = pd.get_dummies(df['seasons'], prefix='season')
    df=df.join(season_dummies)
    return df

#FORECASTING ON WEBS
OUT_STEPS = 20
INPUT_WIDTH = 100
MODEL_NAME = "GRU_bidirectional"

#For this compact version the parameters are different 
kpca_best_parameters = {'web1': {'gamma': 1.0, 'kernel': 'linear'}, 
                        'web2': {'gamma': 0.7, 'kernel': 'linear'}, 
                        'web3': {'gamma': 0.5, 'kernel': 'linear'}, 
                        'web4': {'gamma': 0.5, 'kernel': 'linear'}, 
                        'web5': {'gamma': 0.8, 'kernel': 'linear'}, 
                        'web6': {'gamma': 0.3, 'kernel': 'linear'}, 
                        'web7': {'gamma': 0.5, 'kernel': 'linear'}, 
                        'web8': {'gamma': 0.8, 'kernel': 'linear'}}

#creation of the dataframe for the forecasting
def create_forecast_df (df, web_reduced, web_number, df_t_mean, date_time):
    web_name = 'Web_{number}_first_two_PCS'.format(number=web_number)
    day_of_the_year= df['datetime'].dt.dayofyear
    month= df['datetime'].dt.month
    week = df['settimanay']
    season_1 = df['season_1']
    season_2 = df['season_2']
    season_3 = df['season_3']
    season_4 = df['season_4']
    web_df = pd.DataFrame({"date_time": date_time, 
                           "month": month, 
                           "week": week, 
                           "temp" : df_t_mean,
                           "season_1" : season_1,
                           "season_2" : season_2,
                           "season_3" : season_3,
                           "season_4" : season_4,
                           web_name: web_reduced.ravel()})
    return web_df


#Forecating execution
def forecast (web_forecaster, df_reduced, start_point, title, date_time):

    web_predictions, web_indices, web_figure = web_forecaster.predict(date_time[start_point], days_shift=3)
    
    mae = mean_absolute_error(df_reduced[start_point:start_point + OUT_STEPS].flatten(), web_predictions[0,:,-1])
    print("mae:", mae)

    plt.rcParams['font.size'] = 20
    web_figure.plot(web_indices[-OUT_STEPS:], df_reduced[start_point:start_point + OUT_STEPS],
                    marker='*', label='Labels',
                    c='blue')
    web_figure.title(title)

    # Set the font size of the x-axis and y-axis tick labels
    web_figure.tick_params(axis='x', labelsize=12)
    web_figure.tick_params(axis='y', labelsize=12)
    web_figure.legend()
    web_figure.show()
    #web_figure.close('all')
    return web_predictions, web_indices, web_figure, mae

def web_1_forecast():
    df = load_dataset()
    date_time = pd.to_datetime(df.pop('aaaammgg'), format='%Y%m%d')
    df_t_mean = df[["TM101", "TM102", "TM103"]].mean(axis=1)
    df_web1 = df[["DF101", "DF102", "DF103", "DF104", "DF105", "DF106"]]
    web1_kpca = KernelPCA(n_components=2, kernel=kpca_best_parameters["web1"]["kernel"], gamma=kpca_best_parameters["web1"]["gamma"])
    Web1_kpca_reduced=web1_kpca.fit_transform(df_web1).sum(axis=1)
    web_1_Forecaster = DomeForecaster(df=create_forecast_df(df, -Web1_kpca_reduced, 1, df_t_mean=df_t_mean, date_time=date_time), col_label="Web_1_first_two_PCS", model_name=MODEL_NAME, OUT_STEPS=OUT_STEPS, INPUT_WIDTH=INPUT_WIDTH, MAX_EPOCHS=60)
    web_1_Forecaster.load_forecaster("compact_web_1_forecaster_2_PCS") 
    web_1_predictions_winter, web_1_indices, web_1_figure, mae = forecast(web_1_Forecaster, -Web1_kpca_reduced, 101, title='Web 1 forecast', date_time=date_time)

def web_2_forecast():
    df = load_dataset()
    date_time = pd.to_datetime(df.pop('aaaammgg'), format='%Y%m%d')
    df_t_mean = df[["TM201", "TM202", "TM203", "TM204", "TM205", "TM206", "TM207", "TM208", "TM209", "TM210", "TM211", "TM212"]].mean(axis=1)
    df_web2 = df[["DF201", "DF202", "DF203", "DF204", "DF205", "DF206", "DF207", "DF208", "DF209", "DF210"]]
    web2_kpca = KernelPCA(n_components=2, kernel=kpca_best_parameters["web2"]["kernel"], gamma=kpca_best_parameters["web2"]["gamma"])
    Web2_kpca_reduced=web2_kpca.fit_transform(df_web2).sum(axis=1)
    web_2_Forecaster = DomeForecaster(df=create_forecast_df(df, -Web2_kpca_reduced, 2, df_t_mean=df_t_mean, date_time=date_time), col_label="Web_2_first_two_PCS", model_name=MODEL_NAME, OUT_STEPS=OUT_STEPS, INPUT_WIDTH=INPUT_WIDTH, MAX_EPOCHS=60)
    web_2_Forecaster.load_forecaster("compact_web_2_forecaster_2_PCS") 
    web_2_predictions_winter, web_2_indices, web_2_figure, mae = forecast(web_2_Forecaster, -Web2_kpca_reduced, 101, title='Web 2 forecast', date_time=date_time)

def web_3_forecast():
    df = load_dataset()
    date_time = pd.to_datetime(df.pop('aaaammgg'), format='%Y%m%d')
    df_t_mean = df[["TM301", "TM302", "TM303", "TM304", "TM305", "TM306", "TM307", "TM308"]].mean(axis=1)
    df_web3 = df[["DF301", "DF302", "DF303"]]
    web3_kpca = KernelPCA(n_components=2, kernel=kpca_best_parameters["web3"]["kernel"], gamma=kpca_best_parameters["web3"]["gamma"])
    Web3_kpca_reduced=web3_kpca.fit_transform(df_web3).sum(axis=1)
    web_3_Forecaster = DomeForecaster(df=create_forecast_df(df, -Web3_kpca_reduced, 3, df_t_mean=df_t_mean, date_time=date_time), col_label="Web_3_first_two_PCS", model_name=MODEL_NAME, OUT_STEPS=OUT_STEPS, INPUT_WIDTH=INPUT_WIDTH, MAX_EPOCHS=60)
    web_3_Forecaster.load_forecaster("compact_web_3_forecaster_2_PCS") 
    web_3_predictions_winter, web_3_indices, web_3_figure, mae = forecast(web_3_Forecaster, -Web3_kpca_reduced, 101, title='Web 3 forecast', date_time=date_time)

def web_4_forecast():
    df = load_dataset()
    date_time = pd.to_datetime(df.pop('aaaammgg'), format='%Y%m%d')
    df_t_mean = df[["TM401", "TM402", "TM403"]].mean(axis=1)
    df_web4 = df[["DF401", "DF402", "DF404", "DF405", "DF406", "DF407", "DF408", "DF409", "DF410", "DF411", "DF412", "DF413"]]
    web4_kpca = KernelPCA(n_components=2, kernel=kpca_best_parameters["web4"]["kernel"], gamma=kpca_best_parameters["web4"]["gamma"])
    Web4_kpca_reduced=web4_kpca.fit_transform(df_web4).sum(axis=1)
    web_4_Forecaster = DomeForecaster(df=create_forecast_df(df, -Web4_kpca_reduced, 4, df_t_mean=df_t_mean, date_time=date_time), col_label="Web_4_first_two_PCS", model_name=MODEL_NAME, OUT_STEPS=OUT_STEPS, INPUT_WIDTH=INPUT_WIDTH, MAX_EPOCHS=60)
    web_4_Forecaster.load_forecaster("compact_web_4_forecaster_2_PCS") 
    web_4_predictions_winter, web_4_indices, web_4_figure, mae = forecast(web_4_Forecaster, -Web4_kpca_reduced, 101, title='Web 4 forecast', date_time=date_time)

def web_5_forecast():
    df = load_dataset()
    date_time = pd.to_datetime(df.pop('aaaammgg'), format='%Y%m%d')
    df_t_mean = df[["TM501", "TM502", "TM503"]].mean(axis=1)
    df_web5 = df[["DF502", "DF503", "DF504"]]
    web5_kpca = KernelPCA(n_components=2, kernel=kpca_best_parameters["web5"]["kernel"], gamma=kpca_best_parameters["web5"]["gamma"])
    Web5_kpca_reduced=web5_kpca.fit_transform(df_web5).sum(axis=1)
    web_5_Forecaster = DomeForecaster(df=create_forecast_df(df, -Web5_kpca_reduced, 5, df_t_mean=df_t_mean, date_time=date_time), col_label="Web_5_first_two_PCS", model_name=MODEL_NAME, OUT_STEPS=OUT_STEPS, INPUT_WIDTH=INPUT_WIDTH, MAX_EPOCHS=60)
    web_5_Forecaster.load_forecaster("compact_web_5_forecaster_2_PCS") 
    web_5_predictions_winter, web_5_indices, web_5_figure, mae = forecast(web_5_Forecaster, -Web5_kpca_reduced, 101, title='Web 5 forecast', date_time=date_time)

def web_6_forecast():
    df = load_dataset()
    date_time = pd.to_datetime(df.pop('aaaammgg'), format='%Y%m%d')
    df_t_mean = df[["TM601", "TM602", "TM603"]].mean(axis=1)
    df_web6 = df[["DF601", "DF604", "DF605", "DF606", "DF607", "DF608", "DF609", "DF610", "DF611", "DF612"]]
    web6_kpca = KernelPCA(n_components=2, kernel=kpca_best_parameters["web6"]["kernel"], gamma=kpca_best_parameters["web6"]["gamma"])
    Web6_kpca_reduced=web6_kpca.fit_transform(df_web6).sum(axis=1)
    web_6_Forecaster = DomeForecaster(df=create_forecast_df(df, -Web6_kpca_reduced, 6, df_t_mean=df_t_mean, date_time=date_time), col_label="Web_6_first_two_PCS", model_name=MODEL_NAME, OUT_STEPS=OUT_STEPS, INPUT_WIDTH=INPUT_WIDTH, MAX_EPOCHS=60)
    web_6_Forecaster.load_forecaster("compact_web_6_forecaster_2_PCS") 
    web_6_predictions_winter, web_6_indices, web_6_figure, mae = forecast(web_6_Forecaster, -Web6_kpca_reduced, 101, title='Web 6 forecast', date_time=date_time)

def web_7_forecast():
    df = load_dataset()
    date_time = pd.to_datetime(df.pop('aaaammgg'), format='%Y%m%d')
    df_t_mean = df[["TM101", "TM102", "TM103"]].mean(axis=1)
    df_web7 = df[["DF701", "DF702", "DF703"]]
    web7_kpca = KernelPCA(n_components=2, kernel=kpca_best_parameters["web7"]["kernel"], gamma=kpca_best_parameters["web7"]["gamma"])
    Web7_kpca_reduced=web7_kpca.fit_transform(df_web7).sum(axis=1)
    web_7_Forecaster = DomeForecaster(df=create_forecast_df(df, -Web7_kpca_reduced, 7, df_t_mean=df_t_mean, date_time=date_time), col_label="Web_7_first_two_PCS", model_name=MODEL_NAME, OUT_STEPS=OUT_STEPS, INPUT_WIDTH=INPUT_WIDTH, MAX_EPOCHS=60)
    web_7_Forecaster.load_forecaster("compact_web_7_forecaster_2_PCS") 
    web_7_predictions_winter, web_7_indices, web_7_figure, mae = forecast(web_7_Forecaster, -Web7_kpca_reduced, 101, title='Web 7 forecast', date_time=date_time)

def web_8_forecast():
    df = load_dataset()
    date_time = pd.to_datetime(df.pop('aaaammgg'), format='%Y%m%d')
    df_t_mean = df[["TM801", "TM802", "TM803"]].mean(axis=1)
    df_web8 = df[["DF801", "DF802", "DF803", "DF804", "DF805", "DF806", "DF807", "DF808", "DF809", "DF810"]]
    web8_kpca = KernelPCA(n_components=2, kernel=kpca_best_parameters["web8"]["kernel"], gamma=kpca_best_parameters["web8"]["gamma"])
    Web8_kpca_reduced=web8_kpca.fit_transform(df_web8).sum(axis=1)
    web_8_Forecaster = DomeForecaster(df=create_forecast_df(df, -Web8_kpca_reduced, 8, df_t_mean=df_t_mean, date_time=date_time), col_label="Web_8_first_two_PCS", model_name=MODEL_NAME, OUT_STEPS=OUT_STEPS, INPUT_WIDTH=INPUT_WIDTH, MAX_EPOCHS=60)
    web_8_Forecaster.load_forecaster("compact_web_8_forecaster_2_PCS") 
    web_8_predictions_winter, web_8_indices, web_8_figure, mae = forecast(web_8_Forecaster, -Web8_kpca_reduced, 101, title='Web 8 forecast', date_time=date_time)