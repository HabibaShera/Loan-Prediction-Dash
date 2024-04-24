from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from handle_data import HandleReturns
from sklearn.preprocessing import StandardScaler
import joblib
import dash_bootstrap_components as dbc



data = HandleReturns('loan_data_updated.csv')
app = Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ])
server = app.server

numeric_cols = ['Applicant Income', 'Coapplicant Income', 'Loan Amount', 'Loan Amount Term']
cat_cols = ['Gender', 'Married', 'Dependents', 'Education','Self Employed', 'Property Area', 'Loan Status',  'Credit History']
cat_cols_WithoutTarget = cat_cols.copy()
cat_cols_WithoutTarget.pop(-2)

fig = data.return_pie(col_name='Gender')
fig1 = data.return_hist(x='Applicant Income', title='Applicant Income Distribution')
fig2 = data.return_grouped_bar(x='Loan Status', y='Gender')

# Update the figures with transparent background
#fig.update_layout(plot_bgcolor='rgb(0, 0, 0)') # , paper_bgcolor='rgba(255, 255, 255, 0.9)'
#fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,1)')
#fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')


tab1_layout = dbc.Container([
    dbc.Row([
    dbc.Col([
        dbc.RadioItems(
            id='cat-features',
            options=[{'label': col, 'value': col} for col in cat_cols],
            value='Gender',
            inline=False, 
            style={'font-weight': 'bold', 'font-size': '20px'},
            
        ),
    ], width=3),
    dbc.Col([
        dcc.Graph(id='cat-single-variable', figure=fig,  style={'width': '100%', 'height': '500px'})
    ] , width=9),
])
,
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.RadioItems(id='num-features', options=[{'label': col, 'value': col} for col in numeric_cols], value='ApplicantIncome', 
                           inline=True, style={'font-weight': 'bold', 'font-size': '20px'}),
        ], width=3),
        dbc.Col([
            dcc.Graph(id='num-single-variable', figure=fig1)
    ], width=9),

    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            dbc.RadioItems(id='cat-features_loan', options=[{'label': col, 'value': col} for col in cat_cols_WithoutTarget], value='Gender', 
                           inline=False, style={'font-weight': 'bold', 'font-size': '20px'}),
        ], width=3),
        dbc.Col([
        dcc.Graph(id='cat-vs-loan_status', figure=fig2)
    ], width=9),

    ]),

])
#  , style={'opacity': '0.3'}

@callback(
    Output('cat-vs-loan_status', 'figure'), 
    Input('cat-features_loan', 'value'))
def filter_feature_loan(feature):
    return data.return_grouped_bar('Loan Status', y=feature)

@callback(
    Output('cat-single-variable', 'figure'),
    Input('cat-features', 'value'))
def filter_feature(feature):
    return data.return_pie(col_name=feature)

@callback(
    Output('num-single-variable', 'figure'),
    Input('num-features', 'value'))
def filter_feature(feature):
    return data.return_hist(x=feature, title=feature + ' Distribution')

model = joblib.load('loanModel.pkl')
scaler = joblib.load('standardScaler.pkl')

tab2_layout = dbc.Container([
    html.H1("Loan Approval Prediction", 
        style={
            'font-size': '2em',
            'text-align': 'center',
            'text-shadow': '2px 2px 4px #000000',
            'margin-top': '20px'
        }
),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Gender", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
    dcc.RadioItems(
        id='gender-radio',
        options=[
            {'label': 'Male', 'value': 1},
            {'label': 'Female', 'value': 0}
        ],
        value=1,
        labelStyle={'display': 'inline-block', 'margin-right': '100px', 'font-size': '1.5em', 'font-weight': 'bold'}
    ),
], width=6),
    ]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Married", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
            dcc.RadioItems(
                id='married-radio',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                value=1,
                labelStyle={'display': 'inline-block', 'margin-right': '100px', 'font-size': '1.5em', 'font-weight': 'bold'}

            ),
        ], width=6),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Education", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
            dcc.RadioItems(
                id='education-radio',
                options=[
                    {'label': 'Graduate', 'value': 1},
                    {'label': 'Not Graduate', 'value': 0}
                ],
                value=1,
                labelStyle={'display': 'inline-block', 'margin-right': '100px', 'font-size': '1.5em', 'font-weight': 'bold'}

            ),
        ], width=6),
    ]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Self Employed", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
            dcc.RadioItems(
                id='self-employed-radio',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                value=1,
                labelStyle={'display': 'inline-block', 'margin-right': '100px', 'font-size': '1.5em', 'font-weight': 'bold'}

            ),
        ], width=6),
    ]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Applicant Income", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
            dcc.Input(id='applicant-income', type='number', placeholder='Enter Applicant Income', min=150,
    max=9703, style={'width': '60%'}),
        ], width=6),
    ]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Coapplicant Income", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
            dcc.Input(id='coapplicant-income', type='number', placeholder='Enter Coapplicant Income',min=0,
    max=33837,  style={'width': '60%'}),
        ], width=6),
    ]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Loan Amount", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
            dcc.Input(id='loan-amount', type='number', placeholder='Enter Loan Amount', min=9,
    max=150,  style={'width': '60%'}),
        ], width=6),
    ]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Loan Amount Term", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
            dcc.Input(id='loan-term', type='number', placeholder='Enter Loan Amount Term', min=12,
    max=480,  style={'width': '60%'}),
        ], width=6),
    ]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Credit History", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
            dcc.RadioItems(
                id='credit-history-radio',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                value=1,
                labelStyle={'display': 'inline-block', 'margin-right': '100px', 'font-size': '1.5em', 'font-weight': 'bold'}

            ),
        ], width=6),
    ]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Number of Dependents", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
            dcc.RadioItems(
                id='dependents-radio',
                options=[
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                    {'label': '3+', 'value': 3}
                ],
                value=0,
                labelStyle={'display': 'inline-block', 'margin-right': '100px', 'font-size': '1.5em', 'font-weight': 'bold'}

            ),
        ], width=6),
    ]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Property Area", style={'font-size': '1.5em', 'font-weight': 'bold'}),
        ], width=6),
        dbc.Col([
            dcc.RadioItems(
                id='property-area-radio',
                options=[
                    {'label': 'Semiurban', 'value': 'Semiurban'},
                    {'label': 'Urban', 'value': 'Urban'},
                    {'label': 'Rural', 'value': 'Rural'}
                ],
                value='Semiurban',
                labelStyle={'display': 'inline-block', 'margin-right': '80px', 'font-size': '1.5em', 'font-weight': 'bold'}

            ),
        ], width=6),
    ]),
    html.Hr(style={'height': '3px', 'background-color': 'weight', 'font-weight': 'bold'}),
    
    dbc.Row([
    dbc.Col([
        html.Button('Predict', id='predict-button', n_clicks=0, style={'font-size': '1.5em', 'margin-top': '20px','font-weight': 'bold', 'color': 'crimson'}),
    ], width={'size': 6, 'offset': 5}),
]),

dbc.Row([
    dbc.Col([
        html.Img(id='image', style={'max-width': '100%'})
    ], width={'size': 6, 'offset': 3}),
]),
])
@app.callback(
    Output('image', 'src'),
    [Input('predict-button', 'n_clicks')],
    [Input('gender-radio', 'value'),
     Input('married-radio', 'value'),
     Input('education-radio', 'value'),
     Input('self-employed-radio', 'value'),
     Input('applicant-income', 'value'),
     Input('coapplicant-income', 'value'),
     Input('loan-amount', 'value'),
     Input('loan-term', 'value'),
     Input('credit-history-radio', 'value'),
     Input('dependents-radio', 'value'),
     Input('property-area-radio', 'value')]
)
def predict(n_clicks, gender, married, education, self_employed, applicant_income, coapplicant_income,
            loan_amount, loan_term, credit_history, dependents, property_area):
    if n_clicks > 0:
        input_features = [
            gender, married, education, self_employed, applicant_income,
            coapplicant_income, loan_amount, loan_term, credit_history
        ]
     
        if dependents == 1:
            input_features.extend([1, 0, 0])
        elif dependents == 2:
            input_features.extend([0, 1, 0])
        elif dependents == 3:
            input_features.extend([0, 0, 1])
        else:
            input_features.extend([0, 0, 0])
        
        if property_area == 'Semiurban':
            input_features.extend([1, 0])
        elif property_area == 'Urban':
            input_features.extend([0, 1])
        else:
            input_features.extend([0, 0])
        
        scaled_inputs = scaler.transform([input_features])
        prediction = model.predict(scaled_inputs)
        
        if prediction[0] == 1:
            return app.get_asset_url('approved.png')
        else:
            return app.get_asset_url('denied.png')

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Statistical Facts About Data', children=tab1_layout, 
                style={'color':'grey', 'fontSize':'20px', 'backgroundColor':'rgba(255,255,255,0.7)'},
                # This property is used to define the style of the tab when it is selected (active)
                selected_style={'color': 'rgb(69,148,205)', 'fontWeight': 'bold', 
                'fontSize': '24px', 'backgroundColor': 'rgba(255, 255, 255, 0.5)'}), 
        

        dcc.Tab(label='Loan Prediction', children=tab2_layout, style={'color':'grey', 'fontSize':'20px', 'backgroundColor':'rgba(255,255,255,0.7)'}, 
                selected_style={'color': 'rgb(69,148,205)', 'fontWeight': 'bold', 
                'fontSize': '24px', 'backgroundColor': 'rgba(255, 255, 255, 0.5)'}),
    
    ])
])

if __name__ == '__main__':
    app.run_server(use_reloader=True)
