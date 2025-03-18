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
        "Energy": np.trapezoid(reconstructed_signal**2, np.arange(len(reconstructed_signal))),
        "Power": np.trapezoid(reconstructed_signal**2, np.arange(len(reconstructed_signal))) / (2 * (1 / 20000)),
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
st.title("Stability Prediction")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter='\t', header=None)
    time = df.iloc[:, 0].values
    Signal = df.iloc[:, 1].values

    # Source signal plot
    st.subheader("Source Signal")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=Signal, mode='lines', name='Raw Signal'))
    st.plotly_chart(fig, use_container_width=True)

    # Source signal dropdown
    source_signal = st.selectbox("Select Source Signal", ['Raw Signal', 'Denoised Signal'])
    if source_signal == 'Denoised Signal':
        # Wavelet denoising
        st.subheader("Wavelet Denoising")
        n_levels = st.slider("Define number of levels (1-20):", min_value=1, max_value=20, value=7)
        coeffs = pywt.wavedec(Signal, 'bior2.4', level=n_levels)
        threshold = lambda x: np.sqrt(2 * np.log(len(x))) * np.median(np.abs(x) / 0.6745)
        denoised_coeffs = [pywt.threshold(c, threshold(c), mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
        denoised_signal = pywt.waverec(denoised_coeffs, 'bior2.4')[:len(Signal)]

        # Denoised signal plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=denoised_signal, mode='lines', name='Denoised Signal'))
        st.plotly_chart(fig, use_container_width=True)

    # Wavelet denoising dropdown
    wavelet_option = st.selectbox("Select Wavelet Denoising Option", ['Approximate Coefficients', 'Detailed Coefficients', 'Pearson CC (Approximate)', 'Pearson CC (Detailed)'])
    if wavelet_option == 'Approximate Coefficients':
        coeffs = pywt.wavedec(Signal, 'bior2.4', level=n_levels)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(coeffs[0])), y=coeffs[0], mode='lines', name='Approximation Coefficients'))
        st.plotly_chart(fig, use_container_width=True)
    elif wavelet_option == 'Detailed Coefficients':
        coeffs = pywt.wavedec(Signal, 'bior2.4', level=n_levels)
        detail_coeffs = coeffs[1:]
        for i, coeff in enumerate(detail_coeffs):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(coeff)), y=coeff, mode='lines', name=f'Detail Coefficients {i+1}'))
            st.plotly_chart(fig, use_container_width=True)
    elif wavelet_option == 'Pearson CC (Approximate)':
        coeffs = pywt.wavedec(Signal, 'bior2.4', level=n_levels)
        correlation_approx = np.corrcoef(Signal[:len(coeffs[0])], coeffs[0])[0, 1]
        fig = go.Figure(data=go.Bar(x=['Approx Coefficients'], y=[correlation_approx], name='Pearson CC'))
        st.plotly_chart(fig, use_container_width=True)
    elif wavelet_option == 'Pearson CC (Detailed)':
        coeffs = pywt.wavedec(Signal, 'bior2.4', level=n_levels)
        detail_coeffs = coeffs[1:]
        correlation_detail = [np.corrcoef(Signal[:len(coeff)], coeff)[0, 1] for coeff in detail_coeffs]
        fig = go.Figure(data=go.Bar(x=[f'Detail {i+1}' for i in range(len(detail_coeffs))], y=correlation_detail, name='Pearson CC'))
        st.plotly_chart(fig, use_container_width=True)

    # FFT calculations
    st.subheader("FFT of Signals")
    fft_option = st.selectbox("Select FFT Option", ['FFT of Raw Signal', 'FFT of Denoised Signal', 'FFT of Approx Coefficients', 'FFT of Detail Coefficients'])
    if fft_option == 'FFT of Raw Signal':
        fft_raw = np.abs(np.fft.fft(Signal))[:len(Signal) // 2]
        fft_freqs = np.linspace(0, 20000 / 2, len(fft_raw))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fft_freqs, y=fft_raw, mode='lines', name='FFT of Raw Signal'))
        st.plotly_chart(fig, use_container_width=True)
    elif fft_option == 'FFT of Denoised Signal':
        fft_denoised = np.abs(np.fft.fft(denoised_signal))[:len(Signal) // 2]
        fft_freqs = np.linspace(0, 20000 / 2, len(fft_denoised))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fft_freqs, y=fft_denoised, mode='lines', name='FFT of Denoised Signal'))
        st.plotly_chart(fig, use_container_width=True)
    elif fft_option == 'FFT of Approx Coefficients':
        coeffs = pywt.wavedec(Signal, 'bior2.4', level=n_levels)
        fft_approx_coeffs = np.abs(np.fft.fft(coeffs[0]))[:len(coeffs[0]) // 2]
        fft_freqs = np.linspace(0, 20000 / 2, len(fft_approx_coeffs))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fft_freqs, y=fft_approx_coeffs, mode='lines', name='FFT of Approx Coefficients'))
        st.plotly_chart(fig, use_container_width=True)
    elif fft_option == 'FFT of Detail Coefficients':
        coeffs = pywt.wavedec(Signal, 'bior2.4', level=n_levels)
        detail_coeffs = coeffs[1:]
        for i, coeff in enumerate(detail_coeffs):
            fft_detail_coeffs = np.abs(np.fft.fft(coeff))[:len(coeff) // 2]
            fft_freqs = np.linspace(0, 20000 / 2, len(fft_detail_coeffs))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fft_freqs, y=fft_detail_coeffs, mode='lines', name=f'FFT of Detail Coefficients {i+1}'))
            st.plotly_chart(fig, use_container_width=True)

    # Time-frequency spectrum plot
    st.subheader("Time-Frequency Spectrum")
    spectrum_option = st.selectbox("Select Time-Frequency Spectrum Option", ['Raw Signal', 'Denoised Signal'])
    if spectrum_option == 'Raw Signal':
        f, t, Sxx = spectrogram(Signal, 20000)
        fig = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx), x=t, y=f, colorscale='Viridis'))
        st.plotly_chart(fig, use_container_width=True)
    elif spectrum_option == 'Denoised Signal':
        f, t, Sxx = spectrogram(denoised_signal, 20000)
        fig = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx), x=t, y=f, colorscale='Plasma'))
        st.plotly_chart(fig, use_container_width=True)

    # Download statistical parameters
    st.subheader("Download Statistical Parameters")
    noise = np.zeros_like(Signal)
    stats = calculate_statistical_data(Signal, noise)
    df_stats = pd.DataFrame(stats.items(), columns=["Parameter", "Value"])
    @st.cache
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_stats)
    st.download_button("Download Raw Signal Stats", data=csv, file_name="raw_signal_stats.csv", mime='text/csv')

    noise = Signal - denoised_signal
    stats = calculate_statistical_data(denoised_signal, noise)
    df_stats_denoised = pd.DataFrame(stats.items(), columns=["Parameter", "Value"])
    csv_denoised = convert_df(df_stats_denoised)
    st.download_button("Download Denoised Signal Stats", data=csv_denoised, file_name="denoised_signal_stats.csv", mime='text/csv')
