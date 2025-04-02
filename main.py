import streamlit as st
import pandas as pd
import numpy as np
import pywt
from scipy.signal import spectrogram
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis, entropy
from sklearn.metrics import mean_squared_error

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

st.markdown(
    """
    <style>
    .stApp {
        background-color: rgba(135, 206, 235, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<style>
button {
    height: 40px;
    width: 250px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

container = st.container()
with container:
    st.write(f"<h1 style='text-align: center;'>Wavelet Based Feature Extraction</h1>", unsafe_allow_html=True)

    with st.expander("Introduction", expanded=False):
        st.markdown(
            """
            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px;">
            <p>Sybilytics.AI is a Streamlit-based web application designed for wavelet-based feature extraction...</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, delimiter='\t', header=None)
        column_options = [f'Column {i+1}' for i in range(df.shape[1])]

        st.write("Select Variables:")
        col1, col2 = st.columns(2)
        with col1:
            time_column = st.selectbox("Time:", column_options)
        with col2:
            signal_column = st.selectbox("Signal:", column_options)

        time = df.iloc[:, column_options.index(time_column)].values
        Signal = df.iloc[:, column_options.index(signal_column)].values

        # Source Signal Plot
        st.subheader("Source Signal")
        source_signal = st.selectbox("Select Source Signal", ['Raw Signal', 'Denoised Signal'])
        fig_source = go.Figure()
        
        if source_signal == 'Raw Signal':
            fig_source.add_trace(go.Scatter(x=time, y=Signal, mode='lines', name='Raw Signal'))
        elif source_signal == 'Denoised Signal':
            coeffs = pywt.wavedec(Signal, 'bior2.4', level=7)
            threshold = lambda x: np.sqrt(2 * np.log(len(x))) * np.median(np.abs(x) / 0.6745)
            denoised_coeffs = [pywt.threshold(c, threshold(c), mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
            denoised_signal = pywt.waverec(denoised_coeffs, 'bior2.4')[:len(Signal)]
            fig_source.add_trace(go.Scatter(x=time, y=denoised_signal, mode='lines', name='Denoised Signal'))
        
        fig_source.update_layout(
            font=dict(size=18),
            xaxis_title="Time",
            yaxis_title="Amplitude",
            legend=dict(font=dict(size=18)),
            xaxis=dict(tickcolor='black', tickfont=dict(color='black', size=18)),
            yaxis=dict(tickcolor='black', tickfont=dict(color='black', size=18)),
            xaxis_title_font=dict(size=18),
            yaxis_title_font=dict(size=18)
        )
        st.plotly_chart(fig_source, use_container_width=True, key='source_plot')

        # Wavelet Denoising Plot
        st.subheader("Wavelet Denoising")
        wavelet_option = st.selectbox("Select Wavelet Denoising Option", 
                                    ['Approximate Coefficients', 'Detailed Coefficients', 
                                     'Pearson CC (Approximate)', 'Pearson CC (Detailed)'])
        n_levels = st.slider("Define number of levels (1-20):", 1, 20, 7)
        coeffs = pywt.wavedec(Signal, 'bior2.4', level=n_levels)
        threshold = lambda x: np.sqrt(2 * np.log(len(x))) * np.median(np.abs(x) / 0.6745)
        denoised_coeffs = [pywt.threshold(c, threshold(c), mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
        denoised_signal = pywt.waverec(denoised_coeffs, 'bior2.4')[:len(Signal)]
        
        fig_wavelet = go.Figure()
        if wavelet_option == 'Approximate Coefficients':
            fig_wavelet.add_trace(go.Scatter(x=np.arange(len(coeffs[0])), y=coeffs[0], 
                                           mode='lines', name='Approximation Coefficients'))
        elif wavelet_option == 'Detailed Coefficients':
            for i, coeff in enumerate(coeffs[1:]):
                fig_wavelet.add_trace(go.Scatter(x=np.arange(len(coeff)), y=coeff, 
                                               mode='lines', name=f'Detail Coefficients {i+1}'))
        elif wavelet_option == 'Pearson CC (Approximate)':
            correlation_approx = np.corrcoef(Signal[:len(coeffs[0])], coeffs[0])[0, 1]
            fig_wavelet.add_trace(go.Bar(x=['Approx Coefficients'], y=[correlation_approx], name='Pearson CC'))
        elif wavelet_option == 'Pearson CC (Detailed)':
            detail_coeffs = coeffs[1:]
            correlation_detail = [np.corrcoef(Signal[:len(coeff)], coeff)[0, 1] for coeff in detail_coeffs]
            fig_wavelet.add_trace(go.Bar(x=[f'Detail {i+1}' for i in range(len(detail_coeffs))], 
                                       y=correlation_detail, name='Pearson CC'))
        
        fig_wavelet.update_layout(
            font=dict(size=18),
            xaxis_title="Index",
            yaxis_title="Coefficient Value",
            legend=dict(font=dict(size=18)),
            xaxis=dict(tickcolor='black', tickfont=dict(color='black', size=18)),
            yaxis=dict(tickcolor='black', tickfont=dict(color='black', size=18)),
            xaxis_title_font=dict(size=18),
            yaxis_title_font=dict(size=18)
        )
        st.plotly_chart(fig_wavelet, use_container_width=True, key='wavelet_plot')

        # FFT Plot
        st.subheader("FFT of Signals")
        fft_option = st.selectbox("Select FFT Option", 
                                ['FFT of Raw Signal', 'FFT of Denoised Signal', 
                                 'FFT of Approx Coefficients', 'FFT of Detail Coefficients'])
        fft_raw = np.abs(np.fft.fft(Signal))[:len(Signal) // 2]
        fft_freqs = np.linspace(0, 20000 / 2, len(fft_raw))
        fft_denoised = np.abs(np.fft.fft(denoised_signal))[:len(Signal) // 2]
        
        fig_fft = go.Figure()
        if fft_option == 'FFT of Raw Signal':
            fig_fft.add_trace(go.Scatter(x=fft_freqs, y=fft_raw, mode='lines', name='FFT of Raw Signal'))
        elif fft_option == 'FFT of Denoised Signal':
            fig_fft.add_trace(go.Scatter(x=fft_freqs, y=fft_denoised, mode='lines', name='FFT of Denoised Signal'))
        elif fft_option == 'FFT of Approx Coefficients':
            fft_approx_coeffs = np.abs(np.fft.fft(coeffs[0]))[:len(coeffs[0]) // 2]
            fft_freqs_approx = np.linspace(0, 20000 / 2, len(fft_approx_coeffs))
            fig_fft.add_trace(go.Scatter(x=fft_freqs_approx, y=fft_approx_coeffs, 
                                      mode='lines', name='FFT of Approx Coefficients'))
        elif fft_option == 'FFT of Detail Coefficients':
            detail_coeffs = coeffs[1:]
            for i, coeff in enumerate(detail_coeffs):
                fft_detail_coeffs = np.abs(np.fft.fft(coeff))[:len(coeff) // 2]
                fft_freqs_detail = np.linspace(0, 20000 / 2, len(fft_detail_coeffs))
                fig_fft.add_trace(go.Scatter(x=fft_freqs_detail, y=fft_detail_coeffs, 
                                          mode='lines', name=f'FFT of Detail Coefficients {i+1}'))
        
        fig_fft.update_layout(
            font=dict(size=18),
            xaxis_title="Frequency",
            yaxis_title="Amplitude",
            legend=dict(font=dict(size=18)),
            xaxis=dict(tickcolor='black', tickfont=dict(color='black', size=18)),
            yaxis=dict(tickcolor='black', tickfont=dict(color='black', size=18)),
            xaxis_title_font=dict(size=18),
            yaxis_title_font=dict(size=18)
        )
        st.plotly_chart(fig_fft, use_container_width=True, key='fft_plot')

        # Time-Frequency Spectrum Plot
        st.subheader("Time-Frequency Spectrum")
        spectrum_option = st.selectbox("Select Time-Frequency Spectrum Option", ['Raw Signal', 'Denoised Signal'])
        
        if spectrum_option == 'Raw Signal':
            f, t, Sxx = spectrogram(Signal, 20000)
            fig_spectrum = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx), x=t, y=f, colorscale='Viridis'))
        else:
            f, t, Sxx = spectrogram(denoised_signal, 20000)
            fig_spectrum = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx), x=t, y=f, colorscale='Plasma'))
        
        fig_spectrum.update_layout(
            font=dict(size=18),
            xaxis_title="Time",
            yaxis_title="Frequency",
            xaxis=dict(tickcolor='black', tickfont=dict(color='black', size=18)),
            yaxis=dict(tickcolor='black', tickfont=dict(color='black', size=18)),
            xaxis_title_font=dict(size=18),
            yaxis_title_font=dict(size=18)
        )
        st.plotly_chart(fig_spectrum, use_container_width=True, key='spectrum_plot')

        # Statistical Parameters Download
        st.markdown(f"<h3 style='text-align: center;'>Download Statistical Parameters</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            noise = np.zeros_like(Signal)
            stats = calculate_statistical_data(Signal, noise)
            df_stats = pd.DataFrame(stats.items(), columns=["Parameter", "Value"])
            
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df(df_stats)
            st.download_button(
                "Download Raw Signal Stats",
                data=csv,
                file_name="raw_signal_stats.csv",
                mime='text/csv',
                key='raw_signal_stats',
                use_container_width=True,
                type='primary'
            )

        with col3:
            noise = Signal - denoised_signal
            stats = calculate_statistical_data(denoised_signal, noise)
            df_stats_denoised = pd.DataFrame(stats.items(), columns=["Parameter", "Value"])
            csv_denoised = convert_df(df_stats_denoised)
            st.download_button(
                "Download Denoised Signal Stats",
                data=csv_denoised,
                file_name="denoised_signal_stats.csv",
                mime='text/csv',
                key='denoised_signal_stats',
                use_container_width=True,
                type='primary'
            )
