# calendar.py

import pandas as pd
import holidays
import plotly.graph_objects as go

def calendar_df(df: pd.DataFrame, country: str, nationality: str) -> pd.DataFrame:
    """
    Adds calendar-based features to the dataframe, including hour, weekday, month,
    year, day, and a flag for holidays.
    """
    df = df.assign(
        hour=df['time'].dt.hour,
        weekday=df['time'].dt.dayofweek + 1,  # Monday=1 ... Sunday=7
        month=df['time'].dt.month,
        year=df['time'].dt.year,
        day=df['time'].dt.day
    )

    # Map country names to their 2-letter holiday codes
    country_code_map = {
        'brazil': 'BR',
        'sweden': 'SE',
        'switzerland': 'CH',
        'portugal': 'PT',
        'spain': 'ES'
    }
    country_code = country_code_map.get(country.lower())

    if not country_code:
        raise ValueError(f"Country '{country}' is not supported for holiday mapping.")

    # Get holidays for all years present in the dataframe
    years = sorted(df['year'].unique().tolist())
    country_hols = holidays.country_holidays(country_code, years=years)

    print(f"{nationality} holidays loaded:")
    for date, name in sorted(country_hols.items()):
        if date.year in years:
            print(f"  {date}: {name}")

    # Flag each row if its date is a holiday
    df['is_holiday'] = df['time'].dt.date.isin(country_hols).astype(int)

    return df

def plot_train_test_features(df_train: pd.DataFrame, df_test: pd.DataFrame, nationality: str):
    """
    Plots the features for both training and testing sets on the same graph,
    with a vertical line indicating the split point.
    """
    fig = go.Figure()
    feature_cols = [col for col in df_train.columns if col not in {'time', 'value'}]

    # Plot train features
    for col in feature_cols:
        fig.add_trace(go.Scatter(
            x=df_train['time'], y=df_train[col], mode='lines', name=f'Train: {col}'
        ))

    # Plot test features
    for col in feature_cols:
        fig.add_trace(go.Scatter(
            x=df_test['time'], y=df_test[col], mode='lines', name=f'Test: {col}',
            line=dict(dash='dot')
        ))

    # Add vertical line at the division point
    if not df_test.empty:
        division_time = df_test['time'].iloc[0]
        fig.add_shape(
            type='line', x0=division_time, x1=division_time, y0=0, y1=1,
            yref='paper', line=dict(color='black', width=2)
        )

    # Customize the ranges here:
    # For specific date range in training data, use: df_train[df_train['time'] >= 'YYYY-MM-DD']['time'].min()
    x_min = df_train['time'][df_train['time']== pd.Timestamp('10-01-2019 00:00:00')].min()  # Change this to filter specific range
    x_max = df_train['time'][df_train['time'] == pd.Timestamp('11-30-2019 23:00:00')].max()   # Or use df_train['time'].max() for train only
    
    fig.update_layout(
        # title=f'{nationality} Train and Test Features',
        xaxis_title='Time', yaxis_title='Feature Values', template='plotly_white',
        xaxis=dict(range=[x_min, x_max], title_font=dict(size=24), tickfont=dict(size=20)),
        yaxis=dict(range=[0, 32], title_font=dict(size=24), tickfont=dict(size=20)),  # Set y-axis limits here
        legend=dict(
                font=dict(size=20),
                x=0.95,  # horizontal position (0-1)
                y=1,  # vertical position (0-1)
                bgcolor="rgba(255,255,255,0.8)",  # semi-transparent white background
                bordercolor="white",
                borderwidth=1
            ),
        # height=750,
        # width=950
    )
    
    fig.show()


    print('df_train[\'time\'] type:', type(df_train['time'].iloc[0]))
    try:
        fig.write_image(f"Not_results_plots/Calendar_plot.pdf", 
                        width=1400, height=800, scale=2)
        print(f"✅ PDF saved to plots/Calendar_plot_{nationality.lower()}.pdf")
    except Exception as e:
        print(f"⚠️ Could not save PDF: {e}")
        print("Install kaleido with: pip install kaleido")