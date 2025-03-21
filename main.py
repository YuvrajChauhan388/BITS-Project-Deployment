import streamlit as st
import pandas as pd
import numpy as np
import pywt
from scipy.signal import spectrogram
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis, entropy
from sklearn.metrics import mean_squared_error

# Function to calculate statistical parameters
def calculate_statistical_data(reconstructed_signal, noise):
    params = {
        "Mean": np.mean(reconstructed_signal),
        "Median": np.median(reconstructed_signal),
        "Mode": pd.Series(reconstructed_signal).mode()[0],
        "Std Dev": np.std(reconstructed_signal),
        "Variance": np.var(reconstructed_signal),
        "Mean Square": np.mean(reconstructed_signal**2),
        "RMS": np.sqrt(np.mean(reconstructed_signal**2)),
        "Max": np.max(reconstructed_signal),
        "Peak-to-Peak": np.ptp(reconstructed_signal),
        "Peak-to-RMS": np.max(reconstructed_signal) / np.sqrt(np.mean(reconstructed_signal**2)),
        "Skewness": skew(reconstructed_signal),
        "Kurtosis": kurtosis(reconstructed_signal),
        "Energy": np.trapz(reconstructed_signal**2, np.arange(len(reconstructed_signal))),
        "Power": np.trapz(reconstructed_signal**2, np.arange(len(reconstructed_signal))) / (2 * (1 / 20000)),
        "Crest Factor": np.max(reconstructed_signal) / np.sqrt(np.mean(reconstructed_signal**2)),
        "Impulse Factor": np.max(reconstructed_signal) / np.mean(reconstructed_signal),
        "Shape Factor": np.sqrt(np.mean(reconstructed_signal**2)) / np.mean(reconstructed_signal),
        "Shannon Entropy": entropy(np.abs(reconstructed_signal)),
        "Signal-to-Noise Ratio": 10 * np.log10(np.sum(reconstructed_signal**2) / np.sum(noise**2)),
        "Root Mean Square Error": np.sqrt(mean_squared_error(np.zeros_like(reconstructed_signal), reconstructed_signal)),
        "Maximum Error": np.max(np.abs(np.zeros_like(reconstructed_signal) - reconstructed_signal)),
        "Mean Absolute Error": np.mean(np.abs(np.zeros_like(reconstructed_signal) - reconstructed_signal)),
        "Peak Signal-to-Noise Ratio": 20 * np.log10(np.max(np.zeros_like(reconstructed_signal)) / np.sqrt(mean_squared_error(np.zeros_like(reconstructed_signal), reconstructed_signal))),
        "Coefficient of Variation": np.std(reconstructed_signal) / np.mean(reconstructed_signal)
    }
    return params

# Streamlit app
st.markdown(
    """
    <style>
    .stApp {
        background-color: rgba(135, 206, 235, 0.5); /* Sky Blue with 50% transparency */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

container = st.container()
with container:
    st.write(f"<h1 style='text-align: center;'>Wavelet Based Feature Extraction</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, delimiter='\t', header=None)
        column_options = [f'Column {i+1}' for i in range(df.shape[1])]

        st.write("Select Variables:")
        col1, col2, col3 = st.columns(3)
        with col1:
            time_column = st.selectbox("Time:", column_options)
        with col2:
            variable1_column = st.selectbox("Variable 1:", column_options)
        with col3:
            variable2_column = st.selectbox("Variable 2:", column_options)

        time = df.iloc[:, column_options.index(time_column)].values
        Signal1 = df.iloc[:, column_options.index(variable1_column)].values
        Signal2 = df.iloc[:, column_options.index(variable2_column)].values

        # Source signal plot
        st.subheader("Source Signal")
        source_signal = st.selectbox("Select Source Signal", ['Raw Signal', 'Denoised Signal'])
        fig_source = go.Figure()
        if source_signal == 'Raw Signal':
            fig_source.add_trace(go.Scatter(x=time, y=Signal1, mode='lines', name='Raw Signal 1'))
            fig_source.add_trace(go.Scatter(x=time, y=Signal2, mode='lines', name='Raw Signal 2'))
        elif source_signal == 'Denoised Signal':
            coeffs1 = pywt.wavedec(Signal1, 'bior2.4', level=7)
            threshold = lambda x: np.sqrt(2 * np.log(len(x))) * np.median(np.abs(x) / 0.6745)
            denoised_coeffs1 = [pywt.threshold(c, threshold(c), mode='soft') if i > 0 else c for i, c in enumerate(coeffs1)]
            denoised_signal1 = pywt.waverec(denoised_coeffs1, 'bior2.4')[:len(Signal1)]

            coeffs2 = pywt.wavedec(Signal2, 'bior2.4', level=7)
            denoised_coeffs2 = [pywt.threshold(c, threshold(c), mode='soft') if i > 0 else c for i, c in enumerate(coeffs2)]
            denoised_signal2 = pywt.waverec(denoised_coeffs2, 'bior2.4')[:len(Signal2)]

            fig_source.add_trace(go.Scatter(x=time, y=denoised_signal1, mode='lines', name='Denoised Signal 1'))
            fig_source.add_trace(go.Scatter(x=time, y=denoised_signal2, mode='lines', name='Denoised Signal 2'))
        fig_source.update_layout(
            font=dict(size=14),
            xaxis_title="Time",
            yaxis_title="Amplitude",
            legend=dict(font=dict(size=14)),
            xaxis=dict(tickcolor='black', tickfont=dict(color='black')),
            yaxis=dict(tickcolor='black', tickfont=dict(color='black')),
            xaxis_title_font_color='black',
            yaxis_title_font_color='black'
        )
        st.plotly_chart(fig_source, use_container_width=True, key='source_plot')

        # Wavelet denoising
        st.subheader("Wavelet Denoising")
        wavelet_option = st.selectbox("Select Wavelet Denoising Option", ['Approximate Coefficients', 'Detailed Coefficients', 'Pearson CC (Approximate)', 'Pearson CC (Detailed)'])
        n_levels = st.slider("Define number of levels (1-20):", min_value=1, max_value=20, value=7)
        coeffs1 = pywt.wavedec(Signal1, 'bior2.4', level=n_levels)
        threshold = lambda x: np.sqrt(2 * np.log(len(x))) * np.median(np.abs(x) / 0.6745)
        denoised_coeffs1 = [pywt.threshold(c, threshold(c), mode='soft') if i > 0 else c for i, c in enumerate(coeffs1)]
        denoised_signal1 = pywt.waverec(denoised_coeffs1, 'bior2.4')[:len(Signal1)]

        coeffs2 = pywt.wavedec(Signal2, 'bior2.4', level=n_levels)
        denoised_coeffs2 = [pywt.threshold(c, threshold(c), mode='soft') if i > 0 else c for i, c in enumerate(coeffs2)]
        denoised_signal2 = pywt.waverec(denoised_coeffs2, 'bior2.4')[:len(Signal2)]

        fig_wavelet = go.Figure()
        if wavelet_option == 'Approximate Coefficients':
            fig_wavelet.add_trace(go.Scatter(x=np.arange(len(coeffs1[0])), y=coeffs1[0], mode='lines', name='Approximation Coefficients 1'))
            fig_wavelet.add_trace(go.Scatter(x=np.arange(len(coeffs2[0])), y=coeffs2[0], mode='lines', name='Approximation Coefficients 2'))
        elif wavelet_option == 'Detailed Coefficients':
            for i, (coeff1, coeff2) in enumerate(zip(coeffs1[1:], coeffs2[1:])):
                fig_wavelet.add_trace(go.Scatter(x=np.arange(len(coeff1)), y=coeff1, mode='lines', name=f'Detail Coefficients {i+1} 1'))
                fig_wavelet.add_trace(go.Scatter(x=np.arange(len(coeff2)), y=coeff2, mode='lines', name=f'Detail Coefficients {i+1} 2'))
        elif wavelet_option == 'Pearson CC (Approximate)':
            correlation_approx1 = np.corrcoef(Signal1[:len(coeffs1[0])], coeffs1[0])[0, 1]
            correlation_approx2 = np.corrcoef(Signal2[:len(coeffs2[0])], coeffs2[0])[0, 1]
            fig_wavelet.add_trace(go.Bar(x=['Approx Coefficients 1', 'Approx Coefficients 2'], y=[correlation_approx1, correlation_approx2], name='Pearson CC'))
        elif wavelet_option == 'Pearson CC (Detailed)':
            detail_coeffs1 = coeffs1[1:]
            detail_coeffs2 = coeffs2[1:]
            correlation_detail1 = [np.corrcoef(Signal1[:len(coeff)], coeff)[0, 1] for coeff in detail_coeffs1]
            correlation_detail2 = [np.corrcoef(Signal2[:len(coeff)], coeff)[0, 1] for coeff in detail_coeffs2]
            fig_wavelet.add_trace(go.Bar(x=[f'Detail {i+1} 1' for i in range(len(detail_coeffs1))] + [f'Detail {i+1} 2' for i in range(len(detail_coeffs2))], y=correlation_detail1 + correlation_detail2, name='Pearson CC'))
        fig_wavelet.update_layout(
            font=dict(size=14),
            xaxis_title="Index",
            yaxis_title="Coefficient Value",
            legend=dict(font=dict(size=14)),
            xaxis=dict(tickcolor='black', tickfont=dict(color='black')),
            yaxis=dict(tickcolor='black', tickfont=dict(color='black')),
            xaxis_title_font_color='black',
            yaxis_title_font_color='black'
        )
        st.plotly_chart(fig_wavelet, use_container_width=True, key='wavelet_plot')

        # FFT calculations
        st.subheader("FFT of Signals")
        fft_option = st.selectbox("Select FFT Option", ['FFT of Raw Signal', 'FFT of Denoised Signal', 'FFT of Approx Coefficients', 'FFT of Detail Coefficients'])
        fft_raw1 = np.abs(np.fft.fft(Signal1))[:len(Signal1) // 2]
        fft_raw2 = np.abs(np.fft.fft(Signal2))[:len(Signal2) // 2]
        fft_freqs = np.linspace(0, 20000 / 2, len(fft_raw1))
        fft_denoised1 = np.abs(np.fft.fft(denoised_signal1))[:len(Signal1) // 2]
        fft_denoised2 = np.abs(np.fft.fft(denoised_signal2))[:len(Signal2) // 2]

        fig_fft = go.Figure()
        if fft_option == 'FFT of Raw Signal':
            fig_fft.add_trace(go.Scatter(x=fft_freqs, y=fft_raw1, mode='lines', name='FFT of Raw Signal 1'))
            fig_fft.add_trace(go.Scatter(x=fft_freqs, y=fft_raw2, mode='lines', name='FFT of Raw Signal 2'))
        elif fft_option == 'FFT of Denoised Signal':
            fig_fft.add_trace(go.Scatter(x=fft_freqs, y=fft_denoised1, mode='lines', name='FFT of Denoised Signal 1'))
            fig_fft.add_trace(go.Scatter(x=fft_freqs, y=fft_denoised2, mode='lines', name='FFT of Denoised Signal 2'))
        elif fft_option == 'FFT of Approx Coefficients':
            fft_approx_coeffs1 = np.abs(np.fft.fft(coeffs1[0]))[:len(coeffs1[0]) // 2]
            fft_approx_coeffs2 = np.abs(np.fft.fft(coeffs2[0]))[:len(coeffs2[0]) // 2]
            fft_freqs_approx = np.linspace(0, 20000 / 2, len(fft_approx_coeffs1))
            fig_fft.add_trace(go.Scatter(x=fft_freqs_approx, y=fft_approx_coeffs1, mode='lines', name='FFT of Approx Coefficients 1'))
            fig_fft.add_trace(go.Scatter(x=fft_freqs_approx, y=fft_approx_coeffs2, mode='lines', name='FFT of Approx Coefficients 2'))
        elif fft_option == 'FFT of Detail Coefficients':
            detail_coeffs1 = coeffs1[1:]
            detail_coeffs2 = coeffs2[1:]
            for i, (coeff1, coeff2) in enumerate(zip(detail_coeffs1, detail_coeffs2)):
                fft_detail_coeffs1 = np.abs(np.fft.fft(coeff1))[:len(coeff1) // 2]
                fft_detail_coeffs2 = np.abs(np.fft.fft(coeff2))[:len(coeff2) // 2]
                fft_freqs_detail = np.linspace(0, 20000 / 2, len(fft_detail_coeffs1))
                fig_fft.add_trace(go.Scatter(x=fft_freqs_detail, y=fft_detail_coeffs1, mode='lines', name=f'FFT of Detail Coefficients {i+1} 1'))
                fig_fft.add_trace(go.Scatter(x=fft_freqs_detail, y=fft_detail_coeffs2, mode='lines', name=f'FFT of Detail Coefficients {i+1} 2'))
        fig_fft.update_layout(
            font=dict(size=14),
            xaxis_title="Frequency",
            yaxis_title="Amplitude",
            legend=dict(font=dict(size=14)),
            xaxis=dict(tickcolor='black', tickfont=dict(color='black')),
            yaxis=dict(tickcolor='black', tickfont=dict(color='black')),
            xaxis_title_font_color='black',
            yaxis_title_font_color='black'
        )
        st.plotly_chart(fig_fft, use_container_width=True, key='fft_plot')

        # Time-frequency spectrum plot
        st.subheader("Time-Frequency Spectrum")
        spectrum_option = st.selectbox("Select Time-Frequency Spectrum Option", ['Raw Signal', 'Denoised Signal'])
        if spectrum_option == 'Raw Signal':
            f, t, Sxx1 = spectrogram(Signal1, 20000)
            f, t, Sxx2 = spectrogram(Signal2, 20000)
            fig_spectrum = go.Figure(data=[go.Heatmap(z=10 * np.log10(Sxx1), x=t, y=f, colorscale='Viridis', name='Signal 1'), 
                                           go.Heatmap(z=10 * np.log10(Sxx2), x=t, y=f, colorscale='Plasma', name='Signal 2', visible='legendonly')])
        elif spectrum_option == 'Denoised Signal':
            f, t, Sxx1 = spectrogram(denoised_signal1, 20000)
            f, t, Sxx2 = spectrogram(denoised_signal2, 20000)
            fig_spectrum = go.Figure(data=[go.Heatmap(z=10 * np.log10(Sxx1), x=t, y=f, colorscale='Viridis', name='Signal 1'), 
                                           go.Heatmap(z=10 * np.log10(Sxx2), x=t, y=f, colorscale='Plasma', name='Signal 2', visible='legendonly')])
        fig_spectrum.update_layout(
            font=dict(size=14),
            xaxis_title="Time",
            yaxis_title="Frequency",
            xaxis=dict(tickcolor='black', tickfont=dict(color='black')),
            yaxis=dict(tickcolor='black', tickfont=dict(color='black')),
            xaxis_title_font_color='black',
            yaxis_title_font_color='black'
        )
        st.plotly_chart(fig_spectrum, use_container_width=True, key='spectrum_plot')

        # Download statistical parameters
        st.markdown(f"<h3 style='text-align: center;'>Download Statistical Parameters</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            noise = np.zeros_like(Signal1)
            stats1 = calculate_statistical_data(Signal1, noise)
            df_stats1 = pd.DataFrame(stats1.items(), columns=["Parameter", "Value"])
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            csv1 = convert_df(df_stats1)
            st.download_button("Download Raw Signal 1 Stats", data=csv1, file_name="raw_signal1_stats.csv", mime='text/csv')
        with col3:
            noise = Signal1 - denoised_signal1
            stats1 = calculate_statistical_data(denoised_signal1, noise)
            df_stats_denoised1 = pd.DataFrame(stats1.items(), columns=["Parameter", "Value"])
            csv_denoised1 = convert_df(df_stats_denoised1)
            st.download_button("Download Denoised Signal 1 Stats", data=csv_denoised1, file_name="denoised_signal1_stats.csv", mime='text/csv')

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            noise = np.zeros_like(Signal2)
            stats2 = calculate_statistical_data(Signal2, noise)
            df_stats2 = pd.DataFrame(stats2.items(), columns=["Parameter", "Value"])
            csv2 = convert_df(df_stats2)
            st.download_button("Download Raw Signal 2 Stats", data=csv2, file_name="raw_signal2_stats.csv", mime='text/csv')
        with col3:
            noise = Signal2 - denoised_signal2
            stats2 = calculate_statistical_data(denoised_signal2, noise)
            df_stats_denoised2 = pd.DataFrame(stats2.items(), columns=["Parameter", "Value"])
            csv_denoised2 = convert_df(df_stats_denoised2)
            st.download_button("Download Denoised Signal 2 Stats", data=csv_denoised2, file_name="denoised_signal2_stats.csv", mime='text/csv')
