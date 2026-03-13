import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
from typing import Tuple

def filter_similar_periods(periods, mags, th_hours, th_week, th_months, th_years):
    drop = set()
    def domain(idxs, convert_fn, thresh, unit):
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                a, b = idxs[i], idxs[j]
                da, db = convert_fn(periods[a]), convert_fn(periods[b])
                if abs(da - db) <= thresh:
                    drop.add(a if mags[a] < mags[b] else b)
                    warnings.warn(f"Dropping {unit}-close: {periods[a]:.1f}h vs {periods[b]:.1f}h")

    hrs = [i for i,p in enumerate(periods) if p < 24*7*1.01]
    wks = [i for i,p in enumerate(periods) if 24*7*0.99 <= p < 24*30*1.01]
    mos = [i for i,p in enumerate(periods) if 24*30*0.99 <= p < 24*365*1.01]
    yrs = [i for i,p in enumerate(periods) if p >= 24*365*0.99]
    domain(hrs, lambda x: x, th_hours, 'hour')
    domain(wks, lambda x: x/24/7, th_week, 'week')
    domain(mos, lambda x: x/24/30, th_months, 'month')
    domain(yrs, lambda x: x/24/365, th_years, 'year')
    return [i for i in range(len(periods)) if i not in drop]


def deos(
    df_train: pd.DataFrame,
    n_peaks: int, 
    th_hours: float, 
    th_week: float, 
    th_months: float, 
    th_years: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs FFT analysis on a time series to detect dominant seasonalities (Deos).

    Args:
        df_train (pd.DataFrame): DataFrame containing the training data with a 'value' column.
        trunc_factor (int): Factor to truncate the frequency spectrum for visualization.
        n_peaks (int): The number of top peaks to identify in the spectrum.
        th_hours (float): Threshold in hours for filtering similar daily periods.
        th_week (float): Threshold in hours for filtering similar weekly periods.
        th_months (float): Threshold in hours for filtering similar monthly periods.
        th_years (float): Threshold in hours for filtering similar yearly periods.

    Returns:
        A tuple containing six numpy arrays:
        - f_plot: Frequencies for the truncated spectrum plot.
        - m_plot: Magnitudes for the truncated spectrum plot.
        - peak_freqs: Frequencies of the identified top peaks.
        - peak_mags: Magnitudes of the identified top peaks.
        - keep_freqs: Frequencies of the filtered (final) peaks.
        - keep_mags: Magnitudes of the filtered (final) peaks.
    """
    trunc_factor = 2 # Always truncate by factor of 2 for cleaner plots
    x = df_train['value'].to_numpy()
    N = len(x)
    fftc = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1.0)
    mask = freqs > 0
    fpos = freqs[mask]
    mpos = np.abs(fftc[mask]) / (N / 2)

    # Truncate for cleaner plotting
    if N > 8770*5:
        f_plot = 4*fpos[1:len(fpos)//trunc_factor]
    else:
        f_plot = fpos[1:len(fpos)//trunc_factor]
    m_plot = mpos[1:len(mpos)//trunc_factor]

    # Find the top N peaks in the truncated spectrum
    top_idxs = np.argpartition(m_plot, -n_peaks)[-n_peaks:]
    top_idxs = top_idxs[np.argsort(m_plot[top_idxs])[::-1]]
    peak_freqs = f_plot[top_idxs]
    peak_mags = m_plot[top_idxs]
    peak_periods = 1 / peak_freqs

    # Filter out peaks that are too close to each other to avoid redundancy
    # Assumes the existence of a `filter_similar_periods` function
    keep = filter_similar_periods(
        periods=peak_periods, 
        mags=peak_mags, 
        th_hours=th_hours, 
        th_week=th_week, 
        th_months=th_months, 
        th_years=th_years
    )
    
    keep_freqs = peak_freqs[keep]
    keep_mags = peak_mags[keep]

    return f_plot, m_plot, peak_freqs, peak_mags, keep_freqs, keep_mags

def stairway_to_heaven_signal(df_train: pd.DataFrame, df_test: pd.DataFrame, keep_freqs: np.ndarray, keep_mags: np.ndarray):
    """
    Generate stair-like signals repeating within each cycle defined by the FFT filtered frequencies.
    Each signal will increase from 1 up to the period (in time steps), then restart from 1 again.

    Returns a DataFrame with:
    - 'value' column (original values)
    - 'fft_{period}h_signal' columns for each frequency
    - Also returns a single interactive plot showing all signals
    """

    
    df_out = pd.concat([df_train, df_test]).sort_values('time').reset_index(drop=True)

    N = len(df_out)
    t = np.arange(N)  # sample index as time base (assuming uniform sampling)
    
    df_out[f"fft_{N}h_signal"] = np.arange(0, N)  # add a simple increasing signal for reference
    fig = go.Figure()

    for f, mag in zip(keep_freqs, keep_mags):
        period = int(round(1 / f))
        if period < 1:
            continue  # skip invalid periods
        if N > 8770*6:
            signal = (t % (4*period)) + 1
        else:
            signal = (t % period) + 1

        colname = f"fft_{period}h_signal"
        df_out[colname] = signal

        fig.add_trace(go.Scatter(
            x=df_out['time'],
            y=signal,
            mode='lines',
            name=colname,
            line=dict(width=1)
        ))

        if f == keep_freqs[-1]:
            
            # Add a final trace for the last frequency
            fig.add_trace(go.Scatter(
                x=df_out['time'],
                y=df_out[f"fft_{N}h_signal"],
                mode='lines',
                name=f'fft_{N}h_signal',
                line=dict(width=2, color='purple')  # you can adjust color/width here
            ))

    ## for f, mag in zip(keep_freqs, keep_mags):
    ##     if f <= 0:
    ##         continue  # skip invalid frequencies

    ##     # Sine wave signal: A * sin(2πft)
    ##     signal = mag * np.sin(2 * np.pi * f * t)

    ##     period = int(round(1 / f))  # still useful for naming
    ##     colname = f"FFT {period}-h signal"
    ##     df_out[colname] = signal

    ##     fig.add_trace(go.Scatter(
    ##         x=df_out['time'],
    ##         y=signal,
    ##         mode='lines',
    ##         name=colname,
    ##         line=dict(width=1)
    ##     ))

    # combined_signal = np.zeros_like(t, dtype=float)

    # for f, mag in zip(keep_freqs, keep_mags):
    #     if f <= 0:
    #         continue  # skip invalid frequencies

    #     # Accumulate sine components: A * sin(2πft)
    #     combined_signal += mag * np.sin(2 * np.pi * f * t)

    # # Add only one combined signal trace
    # colname = "fft_combined_signal"
    # df_out[colname] = combined_signal

    # fig.add_trace(go.Scatter(
    #     x=df_out['time'],
    #     y=combined_signal,
    #     mode='lines',
    #     name='Combined FFT Signal',
    #     line=dict(width=2, color='purple')  # you can adjust color/width here
    # ))

    # Add vertical black line at division point
    if not df_test.empty:
        division_time = df_test['time'].iloc[0]
        fig.add_shape(
            type='line',
            x0=division_time, x1=division_time,
            y0=0, y1=1,
            yref='paper',  # makes y0=0 and y1=1 span the full plot height
            line=dict(color='black', width=2, dash='solid')
        )
        
        three_am = pd.Timestamp(f"2019-10-08 03:00:00")
        nine_am = pd.Timestamp(f"2019-10-08 09:00:00")
        
        # fig.add_shape(
        #     type='line',
        #     x0=three_am, x1=three_am,
        #     y0=0, y1=1,
        #     yref='paper',  # makes y0=0 and y1=1 span the full plot height
        #     line=dict(color='magenta', width=2, dash='dash')
        # )

        # fig.add_shape(
        #     type='line',
        #     x0=nine_am, x1=nine_am,
        #     y0=0, y1=1,
        #     yref='paper',  # makes y0=0 and y1=1 span the full plot height
        #     line=dict(color='green', width=2, dash='dash')
        # )

    x_min = df_train['time'][df_train['time']== pd.Timestamp('01-01-2019 00:00:00')].min()  # Change this to filter specific range
    x_max = df_train['time'][df_train['time'] == pd.Timestamp('12-31-2024 23:00:00')].max()   # Or use df_train['time'].max() for train only
    
    fig.update_layout(
        title="Ramps generated from the FFT High-Magnitude Cyclic Periods",
        xaxis_title="Time",
        yaxis_title="Signal Value",
        legend_title="FFT Ramp Signals",
        title_font=dict(size=32),
        template='plotly_white',
        xaxis=dict(range=[x_min, x_max], title_font=dict(size=24), tickfont=dict(size=20)),
        yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
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

    # try:
    #     fig.write_image(f"Not_results_plots/Ramps_BR.pdf", 
    #                     width=1400, height=800, scale=2)
    #     print(f"✅ PDF saved to plots/Ramps_BR.pdf")
    # except Exception as e:
    #     print(f"⚠️ Could not save PDF: {e}")
    #     print("Install kaleido with: pip install kaleido")
    


    # fig.update_layout(
    #     title="Sinusoidal Signal generated from the FFT High-Magnitude Cyclic Periods Separated",
    #     xaxis_title="Time",
    #     yaxis_title="Signal Value",
    #     legend_title="FFT Sinusoidal Signals",
    #     template='plotly_white',
    #     height=550,
    #     # width=950
    # )
    # fig.show()


    mask = df_out['time'] <= '2023-12-31 23:00:00'
    df_train_out = df_out[mask].reset_index(drop=True)
    df_test_out = df_out[~mask].reset_index(drop=True)

    return df_train_out, df_test_out


def add_hatched_region(fig, x0, x1, n_lines=30, color="red", width=1, opacity=0.4):
    """
    Adds a manually drawn, semi-transparent, hatched red rectangle to a figure
    by drawing multiple diagonal lines.
    """
    weekly = 1/ (24 * 7)          # 168 hours
    monthly = 1/ (24 * 30)        # 720 hours
    yearly = 1/ (24 * 365 * 0.99)        # 8760 hours
    five_yearly = 1/ (24 * 365 * 5) # 43800 hours

    # Determine the color based on the region's location
    # This uses a proper if/elif/else chain
    color = 'red' # Default color
    if x0 <= weekly and x1 >= monthly:
        color = 'cyan'  # Your image shows a brown/orange color
    elif x0 <= monthly and x1 >= yearly:
        color = 'brown'
    elif x0 <= yearly and x1 >= five_yearly:
        color = 'gold'
    # The default 'red' will be used for all other cases, like the 24h-168h range.

    # --- The hatching logic remains the same ---
    x_range = x1 - x0
    hatch_width = x_range * 0.5
    line_starts = np.linspace(x0 - hatch_width, x1, n_lines)

    for start_x in line_starts:
        end_x = start_x + hatch_width
        line_draw_start_x = max(start_x, x0)
        line_draw_end_x = min(end_x, x1)

        if line_draw_start_x < line_draw_end_x:
            start_y = (line_draw_start_x - start_x) / hatch_width
            end_y = (line_draw_end_x - start_x) / hatch_width

            fig.add_shape(
                type='line',
                xref='x', yref='paper',
                x0=line_draw_start_x, y0=start_y,
                x1=line_draw_end_x, y1=end_y,
                line=dict(color=color, width=width), # Use the determined color
                opacity=opacity
            )

def plot_fft_light(
    country: str,
    n_peaks: int,
    f_plot: np.ndarray,
    m_plot: np.ndarray,
    peak_freqs: np.ndarray,
    peak_mags: np.ndarray,
    keep_freqs: np.ndarray,
    keep_mags: np.ndarray
):
    """
    Creates and displays a customized FFT spectrum plot with country-specific colors.

    Args:
        country (str): The name of the country to determine the color palette.
        n_peaks (int): The number of peaks identified, used for the plot title.
        f_plot (np.ndarray): Frequencies for the main spectrum plot.
        m_plot (np.ndarray): Magnitudes for the main spectrum plot.
        peak_freqs (np.ndarray): Frequencies of all identified peaks.
        peak_mags (np.ndarray): Magnitudes of all identified peaks.
        keep_freqs (np.ndarray): Frequencies of the filtered peaks to keep.
        keep_mags (np.ndarray): Magnitudes of the filtered peaks to keep.

    Returns:
        go.Figure: The configured Plotly figure object.
    """
    # 1. Define the color palettes for each country
    COLOR_PALETTES = {
        'brazil':      {'spectrum': '#009C3B', 'peaks': '#002776', 'filtered_peaks': '#B9CA1D'},
        'sweden':      {'spectrum': '#006AA7', 'peaks': 'gray',    'filtered_peaks': '#FFCD00'},
        'switzerland': {'spectrum': '#DA291C', 'peaks': 'gray',    'filtered_peaks': 'black'},
        'portugal':    {'spectrum': '#006600', 'peaks': '#FFDA00', 'filtered_peaks': '#DA291C'},
        'spain':       {'spectrum': '#C60B1E', 'peaks': '#751380', 'filtered_peaks': '#FABD00'},
    }
    
    # 2. Normalize country name and select the correct palette
    country_lower = country.lower().strip()
    if country_lower in ['br', 'brasil']:
        country_lower = 'brazil'
    
    # Use Brazil's palette as a safe default if the country isn't found
    palette = COLOR_PALETTES.get(country_lower, COLOR_PALETTES['brazil'])

    # 3. Create the plot with the selected colors
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_plot, y=m_plot, line=dict(color=palette['spectrum']), name='FFT Spectrum'))
    fig.add_trace(go.Scatter(x=peak_freqs, y=peak_mags, mode='markers', marker=dict(color=palette['peaks'], size=8, symbol='circle-open'), name='Top Peaks'))
    fig.add_trace(go.Scatter(x=keep_freqs, y=keep_mags, mode='markers', marker=dict(color=palette['filtered_peaks'], size=10, symbol='circle'), name='Filtered Peaks'))
    
    # --- Vertical Lines and Layout (preserved from your original code) ---
    
    # fig.add_shape(
    #     type='line',
    #     x0=1/(24+2), x1=1/(24+2),
    #     y0=0, y1=1,
    #     yref='paper',
    #     line=dict(color='black', width=2, dash='dash')
    # )
    # fig.add_shape(
    #     type='line',
    #     x0=1/(24-2), x1=1/(24-2),
    #     y0=0, y1=1,
    #     yref='paper',
    #     line=dict(color='black', width=2, dash='dash')
    # )

    # fig.add_shape(type='line', x0=1/(24*7*0.99), x1=1/(24*7*0.99), y0=0, y1=1, yref='paper', line=dict(color='cyan', width=2, dash='dash'))
    # fig.add_shape(type='line', x0=1/(24*7*1.01), x1=1/(24*7*1.01), y0=0, y1=1, yref='paper', line=dict(color='red', width=2, dash='dash'))
    # fig.add_shape(type='line', x0=1/(24*30*0.99), x1=1/(24*30*0.99), y0=0, y1=1, yref='paper', line=dict(color='brown', width=2, dash='dash'))
    # fig.add_shape(type='line', x0=1/(24*30*1.01), x1=1/(24*30*1.01), y0=0, y1=1, yref='paper', line=dict(color='cyan', width=2, dash='dash'))
    # fig.add_shape(type='line', x0=1/(24*365*0.99), x1=1/(24*365*0.99), y0=0, y1=1, yref='paper', line=dict(color='gold', width=2, dash='dash'))
    # fig.add_shape(type='line', x0=1/(24*365*1.01), x1=1/(24*365*1.01), y0=0, y1=1, yref='paper', line=dict(color='brown', width=2, dash='dash'))

    regions_to_highlight = [
        # (1/(24+2), 1/(24-2)),
        # (1/(12+2), 1/(12-2)),
        # (1/(6+2), 1/(6-2)),
        # (1/(24*7*2 + 24*2), 1/(24*7*2 - 24*2)),
        # (1/(24*30*4 + 24*30*2), 1/(24*30*4 - 24*30*2)),
        # (1/(24*365*3 + 24*365*2), 1/(24*365*3 - 24*365*2))
    ]

    # Loop through and add the hatched areas and boundary lines
    for x_start, x_end in regions_to_highlight:
        # Call the new, corrected helper function
        add_hatched_region(fig, x_start, x_end)
        
        fig.add_vline(x=x_start, line_width=2, line_dash="dash", line_color="black")
        fig.add_vline(x=x_end, line_width=2, line_dash="dash", line_color="black")

    # fig.add_annotation(
    #     x=0.123, y=7000,      # Arrowhead points here
    #     ax=0.102, ay=7000,     # Arrow tail starts here
    #     xref="x", yref="y", axref="x", ayref="y",
    #     text="",              # No text for this object
    #     showarrow=True,
    #     arrowhead=2,
    #     arrowsize=1.5,
    #     arrowwidth=2,
    #     arrowcolor="black"
    # )

    # # STEP 2: Place the text ONLY
    # # This annotation has no arrow and can be placed anywhere.
    # fig.add_annotation(
    #     x=0.112, y=7500,     # Choose the EXACT spot for the text (mid-point of arrow, but higher)
    #     text="widening",      # The text label
    #     showarrow=False,      # No arrow for this object
    #     font=dict(size=15, color="black"),
    #     bgcolor="white"
    # )


    # # # --- SHORTING ANNOTATION (Example 2) ---

    # # # STEP 1: Draw the arrow ONLY
    # fig.add_annotation(
    #     x=0.07, y=7000,       # Arrowhead
    #     ax=0.048, ay=7000,      # Arrow tail
    #     xref="x", yref="y", axref="x", ayref="y",
    #     text="",              # No text
    #     showarrow=True,
    #     arrowhead=2,
    #     arrowsize=1.5,
    #     arrowwidth=2,
    #     arrowcolor="black"
    # )

    # # # STEP 2: Place the text ONLY
    # fig.add_annotation(
    #     x=0.058, y=7500,      # Text position: centered horizontally above the arrow
    #     text="widening",      # The text label
    #     showarrow=False,      # No arrow
    #     font=dict(size=15, color="black"),
    #     bgcolor="white"
    # )


    # # # --- "SHORTING" ARROWS (pointing left) ---
    # fig.add_annotation(
    #     x=0.102, y=2000,      # Arrowhead points here
    #     ax=0.123, ay=2000,     # Arrow tail starts here
    #     xref="x", yref="y", axref="x", ayref="y",
    #     text="",              # No text for this object
    #     showarrow=True,
    #     arrowhead=2,
    #     arrowsize=1.5,
    #     arrowwidth=2,
    #     arrowcolor="black"
    # )

    # # # STEP 2: Place the text ONLY
    # # # This annotation has no arrow and can be placed anywhere.
    # fig.add_annotation(
    #     x=0.112, y=2500,     # Choose the EXACT spot for the text (mid-point of arrow, but higher)
    #     text="shrinking",      # The text label
    #     showarrow=False,      # No arrow for this object
    #     font=dict(size=15, color="black"),
    #     bgcolor="white"
    # )

    # fig.add_annotation(
    #     x=0.048, y=2000,       # Arrowhead
    #     ax=0.07, ay=2000,      # Arrow tail
    #     xref="x", yref="y", axref="x", ayref="y",
    #     text="",              # No text
    #     showarrow=True,
    #     arrowhead=2,
    #     arrowsize=1.5,
    #     arrowwidth=2,
    #     arrowcolor="black"
    # )

    # # # STEP 2: Place the text ONLY
    # fig.add_annotation(
    #     x=0.058, y=2500,      # Text position: centered horizontally above the arrow
    #     text="shrinking",      # The text label
    #     showarrow=False,      # No arrow
    #     font=dict(size=15, color="black"),
    #     bgcolor="white"
    # )

    # # Period tick formatting
    tickvals = keep_freqs
    ticktext = [f"{int(round(1/f))}h" if f != 0 else "" for f in tickvals]

    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=ticktext,
        title="Cyclic Period (hours)"
    )
    fig.update_yaxes(title="Magnitude")
    fig.update_layout(
        title=f"FFT Spectrum with {n_peaks} Peaks for {country.title()} Train Data",
        title_font=dict(size=32),
        template='plotly_white',
        xaxis=dict(range=[0.00001, 0.002], title_font=dict(size=24), tickfont=dict(size=20), tickangle=-45),
        # xaxis=dict(range=[0.0001, 0.12], title_font=dict(size=24), tickfont=dict(size=20), tickangle=-45),
        yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
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

    try:
        # fig.write_image(f"Results/FFT_spectrum_{n_peaks}_peaks_results_{country}_DEOS.pdf", 
        #                 width=1400, height=800, scale=2)
        
        fig.write_image(f"Results/{country}/FFT_spectrum_{n_peaks}_peaks_results_{country}_DEOS_zoomed.pdf", 
                        width=1400, height=800, scale=2)
        print(f"✅ PDF saved to Results/{country}/FFT_spectrum_{n_peaks}_peaks_results_{country}_DEOS_zoomed.pdf")
    except Exception as e:
        print(f"⚠️ Could not save PDF: {e}")
        print("Install kaleido with: pip install kaleido")