import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["#0000ff","#fff040","#ff0000","#008000"])

#Function to search for best KPCA hyperparameters
def kpca_hyperparameters_search(df, n_components, gamma_range=np.linspace(0.1, 0.5, 100), kernel_types=["linear", "poly", "rbf", "sigmoid"]):
    
    def reconstruction_error(kpca, X):
        X_reduced = kpca.transform(X)
        X_preimage = kpca.inverse_transform(X_reduced)
        return -1 * mean_squared_error(X, X_preimage)

    param_grid = [{
        "gamma": gamma_range,
        "kernel": kernel_types
    }]

    kpca=KernelPCA(n_components=n_components, fit_inverse_transform=True) 
    grid_search = GridSearchCV(kpca, param_grid, cv=3, scoring=reconstruction_error)
    grid_search.fit(df)

    return grid_search.best_params_


#Function to search for best isomap hyperparameters
def isomap_hyperparameters_search(df, n_components, n_neighbors=[30,40]):
    
    def reconstruction_error(isomap, X):
        return -1 * isomap.reconstruction_error()

    param_grid = [{
        "n_neighbors": n_neighbors
    }]

    isomap=Isomap(n_components=n_components) 
    grid_search = GridSearchCV(isomap, param_grid, cv=3, scoring=reconstruction_error)
    grid_search.fit(df)

    return grid_search.best_params_

df = pd.read_csv("dome_data_compact.csv", sep=",")
df["datetime"] = pd.to_datetime(df["aaaammgg"], format='%Y%m%d')

#Seasons
season_conditions = [
    df.datetime.dt.dayofyear.isin(np.arange(1,81)),
    df.datetime.dt.dayofyear.isin(np.arange(81,173)),
    df.datetime.dt.dayofyear.isin(np.arange(173, 265)),
    df.datetime.dt.dayofyear.isin(np.arange(265,356)),
    df.datetime.dt.dayofyear.isin(np.arange(356,367))
]
seasons = [1,2,3,4,1]
df['seasons'] = np.select(season_conditions, seasons)

#Seasons One hot encoding
season_dummies = pd.get_dummies(df['seasons'], prefix='season')
df=df.join(season_dummies)

# We get only deformometers measures
df_webs = df[["DF101", "DF102", "DF103", "DF104", "DF105", "DF106",
              "DF201", "DF202", "DF203", "DF204", "DF205", "DF206", "DF207", "DF208", "DF209", "DF210",
              "DF301", "DF302", "DF303",
              "DF401", "DF402", "DF404", "DF405", "DF406", "DF407", "DF408", "DF409", "DF410", "DF411", "DF412", "DF413",
              "DF502", "DF503", "DF504",
              "DF601", "DF604", "DF605", "DF606", "DF607", "DF608", "DF609", "DF610", "DF611", "DF612",
              "DF701", "DF702", "DF703",
              "DF801", "DF802", "DF803", "DF804", "DF805", "DF806", "DF807", "DF808", "DF809", "DF810"]]



#KPCA dimensionality reduction - #KPCA hyperparameters are different wrt the papaer because here we are using a reduced dataset
def kpca_dim_reduction():
    kpca_best_parameters = {'gamma': 0.1, 'kernel': 'linear'} #kpca_hyperparameters_search(df_webs, n_components=2) 

    web_kpca = KernelPCA(n_components=2, kernel=kpca_best_parameters['kernel'], gamma=kpca_best_parameters['gamma'])
    Web_kpca_reduced=web_kpca.fit_transform(df_webs)

    fig, ax = plt.subplots()
    fig.suptitle('KPCA - seasonal clusters')
    scatter = ax.scatter(-Web_kpca_reduced[:,0], -Web_kpca_reduced[:,1], c=df["seasons"], cmap=cmap, s=16)
    plt.xlabel("First mapped dimension")
    plt.ylabel("Second mapped dimension")
    legend = ax.legend(handles=scatter.legend_elements()[0], labels=['Winter','Spring','Summer','Autumn'],
                        loc="upper right", title="Seasons")
    ax.add_artist(legend)
    plt.show()


#ISOMAP dimensionality reduction - hyperparameters are different wrt the papaer because here we are using a reduced dataset
def isomap_dim_reduction():
    isomap_best_parameters = {'n_neighbors': 15} #isomap_hyperparameters_search(df_webs, n_components=2)

    web_isomap = Isomap(n_components=2, n_neighbors=isomap_best_parameters['n_neighbors'])
    web_isomap_reduced = web_isomap.fit_transform(df_webs)

    dist = euclidean_distances(web_isomap.embedding_)

    corr, _ = pearsonr (web_isomap.dist_matrix_.ravel(), dist.ravel())

    print ('Isomap Pearson Correlation: %.3f' % corr)
    corr_2 = corr**2
    print ('Isomap Squared Pearson Correlation: %.3f' % corr_2)
    var_res = 1-corr_2
    print ('Isomap Residual Variance: %.3f' % var_res)

    fig, ax = plt.subplots()
    fig.suptitle('Isomap - seasonal clusters')
    scatter = ax.scatter(-web_isomap_reduced[:,0], -web_isomap_reduced[:,1], c=df["seasons"], cmap=cmap, s=16)
    plt.xlabel("First mapped dimension")
    plt.ylabel("Second mapped dimension")
    legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=['Winter','Spring','Summer','Autumn'],
                        loc="lower left", title="Seasons")
    ax.add_artist(legend1)
    plt.show() 



#TSNE dimensionality reduction - hyperparameters are different wrt the paper because here we are using a reduced dataset
def tsne_dim_reduction():
    orig_space_dist = euclidean_distances(df_webs)

    web_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=12)
    web_tsne_reduced = web_tsne.fit_transform(df_webs)

    dist = euclidean_distances(web_tsne.embedding_[:,1].reshape(-1,1))

    corr, _ = pearsonr (orig_space_dist.ravel(), dist.ravel())

    print ('Tsne Pearson Correlation: %.3f' % corr)
    corr_2 = corr**2
    print ('Tsne Squared Pearson Correlation: %.3f' % corr_2)
    var_res = 1-corr_2
    print ('Tsne Residual Variance: %.3f' % var_res)

    fig, ax = plt.subplots()
    fig.suptitle('Tsne - seasonal clusters')
    scatter = ax.scatter(web_tsne_reduced[:,0], web_tsne_reduced[:,1], c=df["seasons"], cmap=cmap, s=16)
    plt.xlabel("First mapped dimension")
    plt.ylabel("Second mapped dimension")
    legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=['Winter','Spring','Summer','Autumn'],
                        loc="lower left", title="Seasons")
    ax.add_artist(legend1)
    plt.show() 

def plot_odd_and_even_webs_vs_TM():
    #TM AVG
    T_MEAN = df[["TM101", "TM102", "TM103","TM201", "TM202", "TM203", "TM204", "TM205", "TM206", "TM207", "TM208", "TM209", "TM210", "TM211", "TM212",
                "TM301", "TM302", "TM303", "TM304", "TM305", "TM306", "TM307", "TM308", "TM401", "TM402", "TM403", "TM501", "TM502", "TM503", "TM601", 
                "TM602", "TM603", "TM701", "TM702", "TM703", "TM704", "TM705", "TM706", "TM707", "TM708", "TM709", "TM710", "TM711", "TM712", "TM801", "TM802", "TM803"]].mean(axis=1)

    #DF measures for each web separately
    df_web1 = df[["DF101", "DF102", "DF103", "DF104", "DF105", "DF106"]]
    df_web2 = df[["DF201", "DF202", "DF203", "DF204", "DF205", "DF206", "DF207", "DF208", "DF209", "DF210"]]
    df_web3 = df[["DF301", "DF302", "DF303"]]
    df_web4 = df[["DF401", "DF402", "DF404", "DF405", "DF406", "DF407", "DF408", "DF409", "DF410", "DF411", "DF412", "DF413"]]
    df_web5 = df[["DF502", "DF503", "DF504"]]
    df_web6 = df[["DF601", "DF604", "DF605", "DF606", "DF607", "DF608", "DF609", "DF610", "DF611", "DF612"]]
    df_web7 = df[["DF701", "DF702", "DF703"]]
    df_web8 = df[["DF801", "DF802", "DF803", "DF804", "DF805", "DF806", "DF807", "DF808", "DF809", "DF810"]]

    #STD 
    df_web1 = (df_web1 - df_web1.mean())/df_web1.std()
    df_web2 = (df_web2 - df_web2.mean())/df_web2.std()
    df_web3 = (df_web3 - df_web3.mean())/df_web3.std()
    df_web4 = (df_web4 - df_web4.mean())/df_web4.std()
    df_web5 = (df_web5 - df_web5.mean())/df_web5.std()
    df_web6 = (df_web6 - df_web6.mean())/df_web6.std()
    df_web7 = (df_web7 - df_web7.mean())/df_web7.std()
    df_web8 = (df_web8 - df_web8.mean())/df_web8.std()


    #Best KPCA hyperparameters are different wrt the paper because here we are using a reduced dataset
    kpca_best_parameters = {'web1': {'gamma': 1.0, 'kernel': 'linear'}, 
                            'web2': {'gamma': 0.7, 'kernel': 'linear'}, 
                            'web3': {'gamma': 0.5, 'kernel': 'linear'}, 
                            'web4': {'gamma': 0.5, 'kernel': 'linear'}, 
                            'web5': {'gamma': 0.8, 'kernel': 'linear'}, 
                            'web6': {'gamma': 0.3, 'kernel': 'linear'}, 
                            'web7': {'gamma': 0.5, 'kernel': 'linear'}, 
                            'web8': {'gamma': 0.8, 'kernel': 'linear'}}

    web1_kpca = KernelPCA(n_components=1, kernel=kpca_best_parameters["web1"]["kernel"], gamma=kpca_best_parameters["web1"]["gamma"])
    web2_kpca = KernelPCA(n_components=1, kernel=kpca_best_parameters["web2"]["kernel"], gamma=kpca_best_parameters["web2"]["gamma"])
    web3_kpca = KernelPCA(n_components=1, kernel=kpca_best_parameters["web3"]["kernel"], gamma=kpca_best_parameters["web3"]["gamma"])
    web4_kpca = KernelPCA(n_components=1, kernel=kpca_best_parameters["web4"]["kernel"], gamma=kpca_best_parameters["web4"]["gamma"])
    web5_kpca = KernelPCA(n_components=1, kernel=kpca_best_parameters["web5"]["kernel"], gamma=kpca_best_parameters["web5"]["gamma"])
    web6_kpca = KernelPCA(n_components=1, kernel=kpca_best_parameters["web6"]["kernel"], gamma=kpca_best_parameters["web6"]["gamma"])
    web7_kpca = KernelPCA(n_components=1, kernel=kpca_best_parameters["web7"]["kernel"], gamma=kpca_best_parameters["web7"]["gamma"])
    web8_kpca = KernelPCA(n_components=1, kernel=kpca_best_parameters["web8"]["kernel"], gamma=kpca_best_parameters["web8"]["gamma"])

    Web1_kpca_reduced=web1_kpca.fit_transform(df_web1).sum(axis=1)
    Web2_kpca_reduced=web2_kpca.fit_transform(df_web2).sum(axis=1)
    Web3_kpca_reduced=web3_kpca.fit_transform(df_web3).sum(axis=1)
    Web4_kpca_reduced=web4_kpca.fit_transform(df_web4).sum(axis=1)
    Web5_kpca_reduced=web5_kpca.fit_transform(df_web5).sum(axis=1)
    Web6_kpca_reduced=web6_kpca.fit_transform(df_web6).sum(axis=1)
    Web7_kpca_reduced=web7_kpca.fit_transform(df_web7).sum(axis=1)
    Web8_kpca_reduced=web8_kpca.fit_transform(df_web8).sum(axis=1)

    date_time = pd.to_datetime(df.pop('aaaammgg'), format='%Y%m%d')

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Odd and even webs trend - Sum of first and second PCs vs avg masonry temp.')

    ax1.plot(date_time, -Web1_kpca_reduced, label='Web1', zorder=-10, color="orange")
    ax1.plot(date_time, -Web3_kpca_reduced, label='Web3', zorder=-10, color="grey")
    ax1.plot(date_time, Web5_kpca_reduced, label='Web5', zorder=-10, color="blue")
    ax1.plot(date_time, -Web7_kpca_reduced, label='Web7', zorder=-10, color="green")

    ax2.plot(date_time, T_MEAN, label='Avg. masonry temp.', zorder=-10, color="black")

    ax3.plot(date_time, -Web2_kpca_reduced, label='Web2', zorder=-10, color="red")
    ax3.plot(date_time, -Web4_kpca_reduced, label='Web4', zorder=-10, color="orange")
    ax3.plot(date_time, -Web6_kpca_reduced, label='Web6', zorder=-10, color="brown")
    ax3.plot(date_time, -Web8_kpca_reduced, label='Web8', zorder=-10, color="blue")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_ylim([-5,3])
    ax2.set_ylabel("Celsius degrees")
    ax3.set_ylim([-6,5])
    plt.show()
