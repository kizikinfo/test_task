import streamlit as st
import pandas as pd
from etna.datasets import TSDataset
from etna.pipeline import Pipeline



# Read the data
url="https://raw.githubusercontent.com/tinkoff-ai/etna/master/examples/data/example_dataset.csv"
original_df = pd.read_csv(url)
#st.write(original_df)
df = TSDataset.to_dataset(original_df)
ts = TSDataset(df, freq="D")


st.set_option('deprecation.showPyplotGlobalUse', False)
nav = st.sidebar.radio("Navigation",["Forecast","Backtest"])

if nav == "Forecast":
    
    from etna.analysis import plot_forecast
    from etna.models.catboost import CatBoostMultiSegmentModel
    import warnings
    from etna.transforms import (
        MeanTransform,
        LagTransform,
        LogTransform,
        SegmentEncoderTransform,
        DateFlagsTransform,
        LinearTrendTransform,
        FourierTransform,
        TimeSeriesImputerTransform
    )
    warnings.filterwarnings("ignore")

    log = LogTransform(in_column="target")
    trend = LinearTrendTransform(in_column="target")
    seg = SegmentEncoderTransform()
    lags = LagTransform(in_column="target", lags=list(range(30, 96, 1)))
    d_flags = DateFlagsTransform(
        day_number_in_week=True,
        day_number_in_month=True,
        week_number_in_month=True,
        week_number_in_year=True,
        month_number_in_year=True,
        year_number=True,
        special_days_in_week=[5, 6],
    )
    mean30 = MeanTransform(in_column="target", window=30)
    fourier = FourierTransform(period=360.25, order=6, out_column="fourier")
    timeseries = TimeSeriesImputerTransform(in_column="target", strategy="forward_fill")

    # Select transform
    desired_transform = st.selectbox('Choose transforms:', ['MeanTransform', 'FourierTransform', 'TimeSeriesImputerTransform'])

    if desired_transform == 'MeanTransform':
        transforms = transforms = [log, trend, lags, d_flags, seg, mean30]
        
    if desired_transform == 'TimeSeriesImputerTransform':
        transforms = transforms = [log, trend, lags, d_flags, seg, timeseries]
        
    if desired_transform == 'FourierTransform':
        transforms = transforms = [log, trend, lags, d_flags, seg, fourier]
        
    HORIZON = 30
    train_ts, test_ts = ts.train_test_split(
        train_start="2019-01-01",
        train_end="2019-10-31",
        test_start="2019-11-01",
        test_end="2019-11-30",
    )

    model = Pipeline(
        model=CatBoostMultiSegmentModel(),
        transforms=transforms,
        horizon=HORIZON,
    )
    model.fit(train_ts)
    
    forecast_ts = model.forecast()
    
    st.write('Forecast plot:')
    st.pyplot(plot_forecast(forecast_ts, test_ts, train_ts, n_train_samples=20))



if nav == "Backtest":

    from etna.models import ProphetModel
    from etna.metrics import SMAPE
    from etna.metrics import MAE
    from etna.metrics import MSE
    from etna.analysis import plot_backtest


    horizon = 31  # Set the horizon for predictions
    model = ProphetModel()  # Create a model
    transforms = []  # A list of transforms -  we will not use any of them

    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
    
    # Running validation
    
    metrics_df, forecast_df, fold_info_df = pipeline.backtest(ts=ts, metrics=[MAE(), MSE(), SMAPE()])
    st.write('Metrics by folds:')
    st.write(metrics_df.head())

    
    metrics_df, forecast_df, fold_info_df = pipeline.backtest(
        ts=ts, metrics=[MAE(), MSE(), SMAPE()], aggregate_metrics=True
    )
    st.write('Metrics averaged over folds:')
    st.write(metrics_df.head())

    # Validation visualisation
    st.write('Validation visualisation:')
    st.pyplot(plot_backtest(forecast_df, ts))
    
    
    # Metrics visualization
    from etna.analysis import (
        metric_per_segment_distribution_plot,
        plot_residuals,
        plot_metric_per_segment,
        prediction_actual_scatter_plot,
    )
    
    
    ts_all = TSDataset(df, freq="D")
    metrics_df, forecast_df, fold_info_df = pipeline.backtest(ts=ts_all, metrics=[MAE(), MSE(), SMAPE()])
    
    st.write('SMAPE metric by folds:')
    st.pyplot(metric_per_segment_distribution_plot(metrics_df=metrics_df, metric_name="SMAPE", plot_type="box"))
    
    st.write('SMAPE metric by segments:')
    st.pyplot(plot_metric_per_segment(metrics_df=metrics_df, metric_name="SMAPE", ascending=True))

    st.write('Residuals of the model predictions from the backtest. Visualization can be done with any feature:')
    st.pyplot(plot_residuals(forecast_df=forecast_df, ts=ts_all))





