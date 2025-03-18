import base64
import io
import numpy as np
import pywt
import pandas as pd
from scipy.signal import spectrogram
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis, entropy
from sklearn.metrics import mean_squared_error

app = dash.Dash(__name__)
app.title = "Stability Prediction"

app.layout = html.Div([
    html.H1("Stability Prediction", style={'textAlign': 'center', 'color': '#003f5c', 'marginBottom': '20px'}),

    html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.A('Click here to select the file and upload'),
            html.Button('Upload', style={'marginLeft': '10px', 'height': '30px', 'lineHeight': '30px', 'width': 'auto', 'padding': '0px 10px', 'borderRadius': '10px'})
        ]),
        style={
            'width': 'auto',
            'height': 'auto',
            'lineHeight': '30px',
            'borderWidth': '0px',
            'textAlign': 'left',
            'display': 'inline-block',
            'padding': '0px'
        },
        multiple=False
    )
], style={'marginLeft': '20px', 'marginTop': '20px'}),

    html.Div(id='output-data-upload'),

    html.Div([
        # Dropdown for Source Signals
        html.Div([
            html.Label("Source Signals", style={'fontWeight': 'bold', 'color': '#444'}),
            dcc.Dropdown(
                id='source-signals',
                options=[
                    {'label': 'Raw Signal', 'value': 'raw'},
                    {'label': 'Denoised Signal', 'value': 'denoised'}
                ],
                value='raw',
                style={'marginBottom': '10px'}
            ),
            dcc.Graph(id='source-plot', style={'height': '350px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),

        html.Div([
            html.Label("Wavelet Denoising", style={'fontWeight': 'bold', 'color': '#444'}),
            html.Div([
                dcc.Dropdown(
                    id='wavelet-denoising',
                    options=[
                        {'label': 'Approximate Coefficients', 'value': 'approx'},
                        {'label': 'Detailed Coefficients', 'value': 'detail'},
                        {'label': 'Pearson CC (Approximate)', 'value': 'pearson_approx'},
                        {'label': 'Pearson CC (Detailed)', 'value': 'pearson_detail'}
                    ],
                    value='approx',
                    style={'width': '70%', 'display': 'inline-block', 'marginRight': '10px'}
                ),
                html.Label("Define number of levels(1-20): ", style={'fontWeight': 'bold', 'color': '#444'}),
                dcc.Input(
                    id='levels-input',
                    type='number',
                    min=1,
                    max=20,
                    step=1,
                    value=7,
                    style={'width': '25%', 'display': 'inline-block'}
                )
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
            dcc.Graph(id='wavelet-plot', style={'height': '350px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})
    ]),

    html.Div([
        # Dropdown for FFT Signals
        html.Div([
            html.Label("FFT of Signals", style={'fontWeight': 'bold', 'color': '#444'}),
            dcc.Dropdown(
                id='fft-signals',
                options=[
                    {'label': 'FFT of Raw Signal', 'value': 'fft_raw'},
                    {'label': 'FFT of Denoised Signal', 'value': 'fft_denoised'},
                    {'label': 'FFT of Approx Coefficients', 'value': 'fft_approx'},
                    {'label': 'FFT of Detail Coefficients', 'value': 'fft_detail'}
                ],
                value='fft_raw',
                style={'marginBottom': '10px'}
            ),
            dcc.Graph(id='fft-plot', style={'height': '350px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),

        # Dropdown for Time-Frequency Spectrum
        html.Div([
            html.Label("Time-Frequency Spectrum", style={'fontWeight': 'bold', 'color': '#444'}),
            dcc.Dropdown(
                id='time-freq-spectrum',
                options=[
                    {'label': 'Raw Signal', 'value': 'spectrum_raw'},
                    {'label': 'Denoised Signal', 'value': 'spectrum_denoised'}
                ],
                value='spectrum_raw',
                style={'marginBottom': '10px'}
            ),
            dcc.Graph(id='spectrum-plot', style={'height': '350px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})
    ]),

    html.Div([
    html.Label("Click to download the raw statistical parameters of Raw Signal", style={'fontWeight': 'bold', 'color': '#444', 'fontSize': '16px', 'marginBottom': '10px', 'display': 'inline-block'}),
    html.Button(
        "Download",
        id="download-button-raw",
        n_clicks=0,
        style={
            'display': 'inline-block',
            'textAlign': 'center',
            'marginTop': '10px',
            'color': '#fff',
            'backgroundColor': '#0066cc',
            'fontSize': '14px',
            'border': 'none',
            'padding': '5px 10px',
            'borderRadius': '10px',
            'cursor': 'pointer',
            'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.2)',
            'transition': '0.3s ease',
            'width': '100px',
            'marginLeft': '10px'
        }
    ),
    dcc.Download(id="download-csv-raw")
], style={'textAlign': 'left', 'marginTop': '10px', 'marginLeft': '20px'}),

html.Div([
    html.Label("Click to download the processed statistical parameters of the Denoised Signal", style={'fontWeight': 'bold', 'color': '#444', 'fontSize': '16px', 'marginBottom': '10px', 'display': 'inline-block'}),
    html.Button(
        "Download",
        id="download-button",
        n_clicks=0,
        style={
            'display': 'inline-block',
            'textAlign': 'center',
            'marginTop': '10px',
            'color': '#fff',
            'backgroundColor': '#0066cc',
            'fontSize': '14px',
            'border': 'none',
            'padding': '5px 10px',
            'borderRadius': '10px',
            'cursor': 'pointer',
            'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.2)',
            'transition': '0.3s ease',
            'width': '120px',
            'marginLeft': '10px'
        }
    ),
    dcc.Download(id="download-csv")
], style={'textAlign': 'left', 'marginTop': '10px', 'marginLeft': '20px'})
])

# Global variables
time = None
Signal = None
fs = 20000  #Hz

# Callback to handle file upload
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if '.lvm' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter='\t', header=None)
                global time, Signal
                time = df.iloc[:, 0].values
                Signal = df.iloc[:, 1].values

                return html.Div([
                    html.H5(f'Uploaded file: {filename}'),
                    html.H6('File successfully processed. You can now use the other features.')
                ])
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])


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
        "Energy": np.trapezoid(reconstructed_signal**2, time),
        "Power": np.trapezoid(reconstructed_signal**2, time) / (2 * (1 / fs)),
        "Crest Factor": np.max(reconstructed_signal) / np.sqrt(np.mean(reconstructed_signal**2)),
        "Impulse Factor": np.max(reconstructed_signal) / np.mean(reconstructed_signal),
        "Shape Factor": np.sqrt(np.mean(reconstructed_signal**2)) / np.mean(reconstructed_signal),
        "Shannon Entropy": entropy(np.abs(reconstructed_signal)),
        "Signal-to-Noise Ratio": 10 * np.log10(np.sum(reconstructed_signal**2) / np.sum(noise**2)),
        "Root Mean Square Error": np.sqrt(mean_squared_error(Signal, reconstructed_signal)),
        "Maximum Error": np.max(np.abs(Signal - reconstructed_signal)),
        "Mean Absolute Error": np.mean(np.abs(Signal - reconstructed_signal)),
        "Peak Signal-to-Noise Ratio": 20 * np.log10(np.max(Signal) / np.sqrt(mean_squared_error(Signal, reconstructed_signal))),
        "Coefficient of Variation": np.std(reconstructed_signal) / np.mean(reconstructed_signal)
    }
    return params

# Generate DataFrame from statistical parameters
def generate_dataframe():
    noise = Signal - denoised_signal
    stats = calculate_statistical_data(denoised_signal, noise)
    return pd.DataFrame(stats.items(), columns=["Parameter", "Value"])

@app.callback(
    [Output('source-plot', 'figure'),
     Output('wavelet-plot', 'figure'),
     Output('fft-plot', 'figure'),
     Output('spectrum-plot', 'figure')],
    [Input('source-signals', 'value'),
     Input('wavelet-denoising', 'value'),
     Input('fft-signals', 'value'),
     Input('time-freq-spectrum', 'value'),
     Input('levels-input', 'value'),
     Input('upload-data', 'contents'),
     Input('upload-data', 'filename')]
)
def update_plots(source_signal, wavelet_option, fft_option, spectrum_option, levels_input, contents, filename):
    if contents is None:
        return [go.Figure() for _ in range(4)]

    # Process the uploaded file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if '.lvm' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter='\t', header=None)
            global time, Signal
            time = df.iloc[:, 0].values
            Signal = df.iloc[:, 1].values
    except Exception as e:
        print(e)
        return [go.Figure() for _ in range(4)]
    n_levels = levels_input  # Update number of levels

    # Perform wavelet decomposition and denoising
    coeffs = pywt.wavedec(Signal, 'bior2.4', level=n_levels)
    threshold = lambda x: np.sqrt(2 * np.log(len(x))) * np.median(np.abs(x) / 0.6745)
    denoised_coeffs = [pywt.threshold(c, threshold(c), mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    denoised_signal = pywt.waverec(denoised_coeffs, 'bior2.4')[:len(Signal)]

    # FFT calculations
    fft_raw = np.abs(np.fft.fft(Signal))[:len(Signal) // 2]
    fft_freqs = np.linspace(0, fs / 2, len(fft_raw))
    fft_denoised = np.abs(np.fft.fft(denoised_signal))[:len(Signal) // 2]
    approx_coeffs = coeffs[0]
    detail_coeffs = coeffs[1:]
    fft_approx_coeffs = np.abs(np.fft.fft(approx_coeffs))[:len(approx_coeffs) // 2]
    fft_detail_coeffs = [np.abs(np.fft.fft(coeff))[:len(coeff) // 2] for coeff in detail_coeffs]

    correlation_approx = [np.corrcoef(Signal[:len(approx_coeffs)], approx_coeffs)[0, 1]]
    correlation_detail = [np.corrcoef(Signal[:len(coeff)], coeff)[0, 1] for coeff in detail_coeffs]

    # Source signal plot
    source_fig = go.Figure()
    if source_signal == 'raw':
        source_fig.add_trace(go.Scatter(x=time, y=Signal, mode='lines', name='Raw Signal'))
    elif source_signal == 'denoised':
        source_fig.add_trace(go.Scatter(x=time, y=denoised_signal, mode='lines', name='Denoised Signal'))
    source_fig.update_layout(title="Source Signal", xaxis_title="Time", yaxis_title="Amplitude")

    # Wavelet denoising plot
    wavelet_fig = go.Figure()
    if wavelet_option == 'approx':
        wavelet_fig.add_trace(go.Scatter(x=np.arange(len(approx_coeffs)), y=approx_coeffs, mode='lines', name='Approximation Coefficients'))
    elif wavelet_option == 'detail':
        for i, coeff in enumerate(detail_coeffs):
            wavelet_fig.add_trace(go.Scatter(x=np.arange(len(coeff)), y=coeff, mode='lines', name=f'Detail Coefficients {i+1}'))
    elif wavelet_option == 'pearson_approx':
        wavelet_fig.add_trace(go.Bar(x=['Approx Coefficients'], y=correlation_approx, name='Pearson CC'))
    elif wavelet_option == 'pearson_detail':
        wavelet_fig.add_trace(go.Bar(x=[f'Detail {i+1}' for i in range(len(detail_coeffs))], y=correlation_detail, name='Pearson CC'))
    wavelet_fig.update_layout(title="Wavelet Decomposition and Denoising", xaxis_title="Coefficients", yaxis_title="Value")

    # FFT plot
    fft_fig = go.Figure()
    if fft_option == 'fft_raw':
        fft_fig.add_trace(go.Scatter(x=fft_freqs, y=fft_raw, mode='lines', name='FFT of Raw Signal'))
    elif fft_option == 'fft_denoised':
        fft_fig.add_trace(go.Scatter(x=fft_freqs, y=fft_denoised, mode='lines', name='FFT of Denoised Signal'))
    elif fft_option == 'fft_approx':
        fft_fig.add_trace(go.Scatter(x=np.linspace(0, fs / 2, len(fft_approx_coeffs)), y=fft_approx_coeffs, mode='lines', name='FFT of Approx Coefficients'))
    elif fft_option == 'fft_detail':
        for i, fft_coeff in enumerate(fft_detail_coeffs):
            fft_fig.add_trace(go.Scatter(x=np.linspace(0, fs / 2, len(fft_coeff)), y=fft_coeff, mode='lines', name=f'FFT of Detail Coefficients {i+1}'))
    fft_fig.update_layout(title="FFT of Signals", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")

    # Time-frequency spectrum plot
    spectrum_fig = go.Figure()
    if spectrum_option == 'spectrum_raw':
        f, t, Sxx = spectrogram(Signal, fs)
        spectrum_fig.add_trace(go.Heatmap(z=10 * np.log10(Sxx), x=t, y=f, colorscale='Viridis'))
    elif spectrum_option == 'spectrum_denoised':
        f, t, Sxx = spectrogram(denoised_signal, fs)
        spectrum_fig.add_trace(go.Heatmap(z=10 * np.log10(Sxx), x=t, y=f, colorscale='Plasma'))
    spectrum_fig.update_layout(title="Time-Frequency Spectrum", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")

    return source_fig, wavelet_fig, fft_fig, spectrum_fig

@app.callback(
    Output("download-csv", "data", allow_duplicate=True),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_raw_data(n_clicks):
    if n_clicks > 0:
        noise = np.zeros_like(Signal)  # No noise for raw signal
        stats = calculate_statistical_data(Signal, noise)
        df = pd.DataFrame(stats.items(), columns=["Parameter", "Value"])
        return dcc.send_data_frame(df.to_csv, "denoised_signal_statistical_parameters.csv")

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
