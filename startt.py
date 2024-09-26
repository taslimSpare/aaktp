import dash
from dash import dcc, html, no_update
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import plotly.figure_factory as ff
import numpy as np

df = pd.read_csv('combined_sorted_data.csv')
df['Time'] = pd.to_datetime(df['Time'])
df['Hour'] = df['Time'].dt.hour
df['Weekday'] = df['Time'].dt.day_name()
now = df['Time'].max()
past_24_hours = now - timedelta(hours=24)
past_6_hours = now - timedelta(hours=6)

current_index = 0
def get_next_data_chunk(chunk_size=2):
    global current_index
    if current_index + chunk_size >= len(df):
        chunk = df[current_index:] 
        current_index = 0  
    else:
        chunk = df[current_index:current_index + chunk_size]
        current_index += chunk_size
    return chunk

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
dark_mode_styles = {
    "background-color": "#1e1e1e",
    "color": "#ffffff",
    "font-family": "Arial, sans-serif"  
}
tabs_styles = {
    'height': '30px',
    "font-size": "10px", 
    "text-align": "center"
    }

def build_quick_stats_panel():
    return html.Div(
        id="quick-stats",
        style={
            "flex": "0 0 15%", 
            "height": "100vh",
            "border-right": "2px solid #333",  
            "padding": "10px",
            "overflow": "auto", 
            **dark_mode_styles
        },
        children=[
            html.Div(
                id="temperature-indicator",
                style={"margin-bottom": "20px", "border": "1px solid #333", "padding": "10px"},
                children=[
                    html.P("Current Temperature", style={"font-size": "15px"}),
                    html.Div(
                        style={"display": "flex", "align-items": "center"},
                        children=[
                            daq.LEDDisplay(
                                id="temperature-led",
                                value='0.00',
                                backgroundColor="#1e2130",
                                size=15,
                            ),
                            html.Div( 
                                style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-left": "10px"},
                                children=[
                                    html.P(id="temperature-status", style={"font-size": "12px", "margin": "0"}),  # Status text
                                    daq.Indicator(
                                        id="temperature-indicator-light",
                                        color="green",  
                                        size=15, 
                                    )
                                ]
                            )
                        ]
                    ),
                    html.P(["Current NH", html.Sub("3")], style={"font-size": "15px"}),
                    html.Div(
                        style={"display": "flex", "align-items": "center"},
                        children=[
                            daq.LEDDisplay(
                                id="ammonia-led",
                                value='0.00',
                                backgroundColor="#1e2130",
                                size=15,
                            ),
                            html.Div( 
                                style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-left": "10px"},
                                children=[
                                    html.P(id="ammonia-status", style={"font-size": "12px", "margin": "0"}),
                                    daq.Indicator(
                                        id="ammonia-indicator-light",
                                        color="green", 
                                        size=15,
                                    )
                                ]
                            )
                        ]
                    ),
                    html.P(["Current H", html.Sub("2"), "S"], style={"font-size": "15px"}),
                    html.Div(
                        style={"display": "flex", "align-items": "center"},
                        children=[
                            daq.LEDDisplay(
                                id="h2s-led",
                                value='0.00',
                                backgroundColor="#1e2130",
                                size=15,
                            ),
                            html.Div(  
                                style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-left": "10px"},
                                children=[
                                    html.P(id="h2s-status", style={"font-size": "12px", "margin": "0"}),  
                                    daq.Indicator(
                                        id="h2s-indicator-light",
                                        color="green",  
                                        size=15,
                                    )
                                ]
                            )
                        ]
                    ),
                    html.P(["Current NO", html.Sub("2")], style={"font-size": "15px"}),
                    html.Div(
                        style={"display": "flex", "align-items": "center"},
                        children=[
                            daq.LEDDisplay(
                                id="No2-led",
                                value='0.00',
                                backgroundColor="#1e2130",
                                size=15,
                            ),
                            html.Div(  
                                style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-left": "10px"},
                                children=[
                                    html.P(id="No2-status", style={"font-size": "12px", "margin": "0"}),  
                                    daq.Indicator(
                                        id="No2-indicator-light",
                                        color="green",  
                                        size=15,
                                    )
                                ]
                            )
                        ]
                    ),
                    html.P("Current Voc", style={"font-size": "15px"}),
                    html.Div(
                        style={"display": "flex", "align-items": "center"},
                        children=[
                            daq.LEDDisplay(
                                id="Voc-led",
                                value='0.00',
                                backgroundColor="#1e2130",
                                size=15,
                            ),
                            html.Div(  
                                style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-left": "10px"},
                                children=[
                                    html.P(id="Voc-status", style={"font-size": "12px", "margin": "0"}),  
                                    daq.Indicator(
                                        id="Voc-indicator-light",
                                        color="green",  
                                        size=15,
                                    )
                                ]
                            )
                        ]
                    ),
                    html.P("Current Pm10", style={"font-size": "15px"}),
                    html.Div(
                        style={"display": "flex", "align-items": "center"},
                        children=[
                            daq.LEDDisplay(
                                id="Pm10-led",
                                value='0.00',
                                backgroundColor="#1e2130",
                                size=15,
                            ),
                            html.Div(  
                                style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-left": "10px"},
                                children=[
                                    html.P(id="Pm10-status", style={"font-size": "12px", "margin": "0"}),  
                                    daq.Indicator(
                                        id="Pm10-indicator-light",
                                        color="green",  
                                        size=15,
                                    )
                                ]
                            )
                        ]
                    ),
                    html.P("Current Pm2", style={"font-size": "15px"}),
                    html.Div(
                        style={"display": "flex", "align-items": "center"},
                        children=[
                            daq.LEDDisplay(
                                id="Pm2-led",
                                value='0.00',
                                backgroundColor="#1e2130",
                                size=15,
                            ),
                            html.Div(  
                                style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-left": "10px"},
                                children=[
                                    html.P(id="Pm2-status", style={"font-size": "12px", "margin": "0"}),  
                                    daq.Indicator(
                                        id="Pm2-indicator-light",
                                        color="green",  
                                        size=15,
                                    )
                                ]
                            )
                        ]
                    ),
                ],
            ),
        ],
    )

app.layout = html.Div(
    [
        html.Div(
            [
                build_quick_stats_panel(), 
                html.Div(
                    [
                        dcc.Tabs(
                            id="tabs-example",
                            value='tab-1',
                            children=[
                                dcc.Tab(label='A', value='tab-1', style={"font-size":"16px", "text-align": "center", **dark_mode_styles}, selected_style={"background-color": "#333", "color": "#ffffff"}),
                                dcc.Tab(label='B', value='tab-2', style=dark_mode_styles, selected_style={"background-color": "#333", "color": "#ffffff"}),
                                dcc.Tab(label='C', value='tab-3', style=dark_mode_styles, selected_style={"background-color": "#333", "color": "#ffffff"}),
                            ],
                            colors={
                                "border": "#333",
                                "primary": "#ff5733",
                                "background": "#1e1e1e"
                            },
                        ),
                        html.Div(id='tabs-content-example', style={"flex": "1", **dark_mode_styles}),
                    ],
                    style={"flex": "1", "overflow": "auto"}
                )
            ],
            style={"display": "flex", "height": "100vh", "width": "100vw", "overflow": "hidden"}
            
        ),
        dcc.Interval(id='interval-component', interval=6000, n_intervals=0)
    ])
@app.callback(
    dash.dependencies.Output('tabs-content-example', 'children'),
    [dash.dependencies.Input('tabs-example', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(
            [
                dcc.Graph(id='time-series-chart', style={"width": "100%", "height": "33vh"}),
                dcc.Graph(id='temp-humidity-chart', style={"width": "100%", "height": "33vh"}),
                dcc.Graph(id='scatter-chart', style={"width": "100%", "height": "33vh"}),
            ]
        )
    elif tab == 'tab-2':
        return html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='ammonia-variations', style={"width": "48%", "height": "45vh"}),
                        dcc.Graph(id='No2-variations', style={"width": "48%", "height": "45vh"}),
                    ],
                    style={"display": "flex", "justify-content": "space-between", "padding": "10px"}
                ),
                html.Div(
                    [
                        dcc.Graph(id='Pm-variations', style={"width": "48%", "height": "45vh"}),
                        dcc.Graph(id='VOC-variations', style={"width": "48%", "height": "45vh"}),
                    ],
                    style={"display": "flex", "justify-content": "space-between", "padding": "10px"}
                ),
            ],
            style={"text-align": "center", "padding": "20px", **dark_mode_styles}
        )
    elif tab == 'tab-3':
        return html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='li-variations', style={"width": "48%", "height": "45vh"}),
                        dcc.Graph(id='Np-variations', style={"width": "48%", "height": "45vh"}),
                    ],
                    style={"display": "flex", "justify-content": "space-between", "padding": "10px"}
                ),
                html.Div(
                    [
                        dcc.Graph(id='Na-variations', style={"width": "48%", "height": "45vh"}),
                        dcc.Graph(id='H2s-variations', style={"width": "48%", "height": "45vh"}),
                    ],
                    style={"display": "flex", "justify-content": "space-between", "padding": "10px"}
                ),
            ],
            style={"text-align": "center", "padding": "20px", **dark_mode_styles}
        )

@app.callback(
    dash.dependencies.Output('time-series-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_time_series_chart(n_intervals, active_tab):
    if active_tab != 'tab-1':
        return no_update
    chunk = get_next_data_chunk()
    fig = px.line(chunk, x='Time', y='Tm', title='Temperature Over Time')
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
        ),
        yaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True
        )
    )
    return fig


@app.callback(
    dash.dependencies.Output('temp-humidity-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_temp_humidity_chart(n_intervals, active_tab):
    if active_tab!='tab-1':
        return no_update
    chunk = get_next_data_chunk()
    df_melted = chunk.melt(id_vars='Time', value_vars=['Tm', 'Rh'], 
                           var_name='Variable', value_name='Value')
    fig = px.line(df_melted, x='Time', y='Value', color='Variable',
                  
                  labels={'Value': 'Values', 'Time': 'Hour'},
                  title='Temperature and Humidity by Hour')
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True
        ),
        yaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True
        )
    )
    return fig

@app.callback(
    dash.dependencies.Output('scatter-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_scatter_plot(n_intervals, active_tab):
    if active_tab!='tab-1':
        return no_update
    chunk = get_next_data_chunk()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chunk['Li'], 
        y=chunk['Tm'], 
        mode='markers', 
        name='Li vs. Tm',
        marker=dict(size=10, color='blue')
    ))
    slope, intercept = np.polyfit(chunk['Li'], chunk['Tm'], 1)
    trendline = np.array(chunk['Li']) * slope + intercept
    fig.add_trace(go.Scatter(
        x=chunk['Li'], 
        y=trendline, 
        mode='lines', 
        name='Trendline',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title="Light Intensity vs. Temperature",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Light Intensity",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Temperature",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e",  
        paper_bgcolor="#1e1e1e",  
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True
    )
    return fig

@app.callback(
    dash.dependencies.Output('Pm-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_pm_chart(n_interval, active_tab):
    if active_tab!='tab-2':
        return no_update
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['Pm2'],
        mode='lines',
        name='Pm2',
    ))
    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['Pm10'],
        mode='lines',
        name='Pm10',
    ))

    fig.update_layout(
        title="Particulate Matter by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 990]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True,  
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('VOC-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_VOC_chart(n_intervals, active_tab):
    if active_tab!='tab-2':
        return no_update
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['Voc'],
        mode='lines',
        name='Pm2',
    ))
    fig.update_layout(
        title="Volatile Organic Compounds by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 500]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True, 
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('No2-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_No2_chart(n_intervals, active_tab):
    if active_tab!='tab-2':
        return no_update
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['No2'],
        mode='lines',
        name='Pm2',
    ))
    fig.update_layout(
        title="No2 by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 3]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('ammonia-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_Nh3_chart(n_intervals):
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['Nh3'],
        mode='lines',
        name='Pm2',
    ))
    fig.update_layout(
        title="Ammonia by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 150]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True, 
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('li-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_Li_chart(n_intervals):
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['Li'],
        mode='lines',
        name='Li',
    ))
    fig.update_layout(
        title="Light Intensity by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 750]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True, 
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('Np-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_Np_chart(n_intervals):
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['Np'],
        mode='lines',
        name='Np',
    ))
    fig.update_layout(
        title="Nitrogen Oxides by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 500]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True, 
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig


@app.callback(
    dash.dependencies.Output('Na-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_Na_chart(n_intervals):
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['Na'],
        mode='lines',
        name='Pm2',
    ))
    fig.update_layout(
        title="Na by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 90]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True, 
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('H2s-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_H2s_chart(n_intervals):
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['H2s'],
        mode='lines',
        name='H2s',
    ))
    fig.update_layout(
        title="Hydrogen Sulphide by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 20]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True, 
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

# THRESHOLD_VALUES = {
#     'temperature': 30.0,
#     'ammonia': 0.5,
#     'h2s': 1.0,
#     'no2': 0.2,
#     'voc': 0.3,
#     'pm10': 50.0,
#     'pm2': 25.0
# }

@app.callback(
    [dash.dependencies.Output('temperature-led', 'value'),
     dash.dependencies.Output('temperature-indicator-light', 'color'),
     dash.dependencies.Output('temperature-status', 'children')],  
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_temperature_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_temp = chunk['Tm'].iloc[-1]  
    # return f"{latest_temp:.2f}"  
    if latest_temp >= 41:
        indicator_color = "red"  
        status_text = "Lethal"
    elif 35 <= latest_temp < 41:
        indicator_color = "orange"  
        status_text = "Distress"
    elif 30 <= latest_temp < 35:
        indicator_color = "yellow"  
        status_text = "Stress"
    elif latest_temp < 30:
        indicator_color = "green"  
        status_text = "Comfort"

    return f"{latest_temp:.2f}", indicator_color, status_text

@app.callback(
    [dash.dependencies.Output('ammonia-led', 'value'),
     dash.dependencies.Output('ammonia-indicator-light', 'color'),
     dash.dependencies.Output('ammonia-status', 'children')],  
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_ammonia_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_nh3 = chunk['Nh3'].iloc[-1]  
    
    if latest_nh3 >= 50:
        indicator_color = "red"  
        status_text = "Lethal"
    elif 20 <= latest_nh3 < 25:
        indicator_color = "orange"  
        status_text = "Distress"
    elif 10 <= latest_nh3 < 20:
        indicator_color = "yellow"  
        status_text = "Stress"
    else:
        indicator_color = "green"  
        status_text = "Comfort"
    
    return f"{latest_nh3:.2f}", indicator_color, status_text


@app.callback(
    [dash.dependencies.Output('h2s-led', 'value'),
     dash.dependencies.Output('h2s-indicator-light', 'color'),
     dash.dependencies.Output('h2s-status', 'children')],  
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_h2s_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_h2s = chunk['H2s'].iloc[-1]  
    # return f"{latest_h2s:.2f}" 
    if latest_h2s >= 20:
        indicator_color = "red"  
        status_text = "Lethal"
    elif 10 <= latest_h2s < 20:
        indicator_color = "orange"  
        status_text = "Distress"
    elif 5 <= latest_h2s < 10:
        indicator_color = "yellow"  
        status_text = "Stress"
    else:
        indicator_color = "green"  
        status_text = "Comfort"

    return f"{latest_h2s:.2f}", indicator_color, status_text

@app.callback(
    [dash.dependencies.Output('No2-led', 'value'),
     dash.dependencies.Output('No2-indicator-light', 'color'),
     dash.dependencies.Output('No2-status', 'children')],  
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_no2_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_no2 = chunk['No2'].iloc[-1]  
    # return f"{latest_no2:.2f}" 
    if latest_no2 >= 20:
        indicator_color = "red"  
        status_text = "Lethal"
    elif 10 <= latest_no2 < 20:
        indicator_color = "orange"  
        status_text = "Distress"
    elif 5 <= latest_no2 < 10:
        indicator_color = "yellow"  
        status_text = "Stress"
    else:
        indicator_color = "green"  
        status_text = "Comfort"

    return f"{latest_no2:.2f}", indicator_color, status_text


@app.callback(
    [dash.dependencies.Output('Voc-led', 'value'),
     dash.dependencies.Output('Voc-indicator-light', 'color'),
     dash.dependencies.Output('Voc-status', 'children')],  
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_voc_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_voc = chunk['Voc'].iloc[-1]  
    # return f"{latest_voc:.2f}"
    if latest_voc >= 500:
        indicator_color = "red"  
        status_text = "Lethal"
    elif 250 <= latest_voc < 500:
        indicator_color = "orange"  
        status_text = "Distress"
    elif 150 <= latest_voc < 250:
        indicator_color = "yellow"  
        status_text = "Stress"
    else:
        indicator_color = "green"  
        status_text = "Comfort"

    return f"{latest_voc:.2f}", indicator_color, status_text


@app.callback(
    [dash.dependencies.Output('Pm10-led', 'value'),
     dash.dependencies.Output('Pm10-indicator-light', 'color'),
     dash.dependencies.Output('Pm10-status', 'children')],  
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_pm10_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_pm10 = chunk['Pm10'].iloc[-1]  
    # return f"{latest_pm10:.2f}"
    if latest_pm10 >= 250:
        indicator_color = "red"  
        status_text = "Lethal"
    elif 150 <= latest_pm10 < 250:
        indicator_color = "orange"  
        status_text = "Distress"
    elif 60 <= latest_pm10 < 150:
        indicator_color = "yellow"  
        status_text = "Stress"
    else:
        indicator_color = "green"  
        status_text = "Comfort"

    return f"{latest_pm10:.2f}", indicator_color, status_text


@app.callback(
    [dash.dependencies.Output('Pm2-led', 'value'),
     dash.dependencies.Output('Pm2-indicator-light', 'color'),
     dash.dependencies.Output('Pm2-status', 'children')],  
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_pm2_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_pm2 = chunk['Pm2'].iloc[-1]  
    # return f"{latest_pm2:.2f}"
    if latest_pm2 >= 50:
        indicator_color = "red"  
        status_text = "Lethal"
    elif 25 <= latest_pm2 < 50:
        indicator_color = "orange"  
        status_text = "Distress"
    elif 15 <= latest_pm2 < 25:
        indicator_color = "yellow"  
        status_text = "Stress"
    else:
        indicator_color = "green"  
        status_text = "Comfort"

    return f"{latest_pm2:.2f}", indicator_color, status_text


if __name__ == '__main__':
    app.run_server(debug=True, port=5000)