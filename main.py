import argparse

from dim_reduction.dim_reduction import kpca_dim_reduction, isomap_dim_reduction, tsne_dim_reduction, plot_odd_and_even_webs_vs_TM
from forecasting.compact_forecasting import web_1_forecast, web_2_forecast, web_3_forecast, web_4_forecast, web_5_forecast, web_6_forecast, web_7_forecast, web_8_forecast

ap = argparse.ArgumentParser()
ap.add_argument("--dimred", required=False, help="run and plot dimensionality reduction task with specified algorithm (for example --dimred kpca or --dimred isomap or --dimred tsne or --dimred year)")
ap.add_argument("--forecast", required=False, help="run and plot forecasting task on the specified web (for example --forecast 1 executes prediction on web 1)")
args = ap.parse_args()

def main(args):
    if args.dimred == 'kpca':
        kpca_dim_reduction()
    elif args.dimred == 'isomap':
        isomap_dim_reduction()
    elif args.dimred == 'tsne':
        tsne_dim_reduction()
    elif args.dimred == 'year':
        plot_odd_and_even_webs_vs_TM()
    elif args.forecast == '1':
        web_1_forecast()
    elif args.forecast == '2':
        web_2_forecast()
    elif args.forecast == '3':
        web_3_forecast()
    elif args.forecast == '4':
        web_4_forecast()
    elif args.forecast == '5':
        web_5_forecast()
    elif args.forecast == '6':
        web_6_forecast()
    elif args.forecast == '7':
        web_7_forecast()
    elif args.forecast == '8':
        web_8_forecast()
    else:
        print("Error reading arguments. Please follow the rules in README")

if __name__ == "__main__":
    main(args)