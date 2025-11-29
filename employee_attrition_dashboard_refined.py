
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import dash.dash_table as dt

# Load data
df = pd.read_csv(r"C:\Data Science Project\Data_Dashboard.csv")

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def filter_df(age_min, age_max, income_min, income_max, selected_roles, selected_genders, selected_attrition):
    d = df.copy()
    if 'Age' in d.columns:
        d = d[(d['Age'] >= age_min) & (d['Age'] <= age_max)]
    if 'Monthly Income' in d.columns:
        d = d[(d['Monthly Income'] >= income_min) & (d['Monthly Income'] <= income_max)]
    if selected_roles:
        d = d[d['Job Role'].isin(selected_roles)]
    if selected_genders:
        d = d[d['Gender'].isin(selected_genders)]
    if selected_attrition:
        d = d[d['Attrition'].isin(selected_attrition)]
    return d

# Layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Employee Attrition Dashboard", className='text-center my-3'))),
    dbc.Row([
        dbc.Col([
            html.H5("Filters"),
            html.Label("Age (Min)"),
            dcc.Input(id='age_min', type='number', value=int(df['Age'].min()), debounce=True, style={'width': '100%'}),
            html.Label("Age (Max)"),
            dcc.Input(id='age_max', type='number', value=int(df['Age'].max()), debounce=True, style={'width': '100%'}),
            html.Br(),
            html.Label("Monthly Income (Min)"),
            dcc.Input(id='income_min', type='number', value=int(df['Monthly Income'].min()), debounce=True, style={'width': '100%'}),
            html.Label("Monthly Income (Max)"),
            dcc.Input(id='income_max', type='number', value=int(df['Monthly Income'].max()), debounce=True, style={'width': '100%'}),
            html.Br(),
            html.Label("Job Role"),
            dcc.Dropdown(
                id='jobrole_dropdown',
                options=[{'label': v, 'value': v} for v in sorted(df['Job Role'].astype(str).unique())],
                multi=True
            ),
            html.Br(),
            html.Label("Gender"),
            dcc.Checklist(
                id='gender_checklist',
                options=[{'label': v, 'value': v} for v in sorted(df['Gender'].astype(str).unique())],
                value=list(sorted(df['Gender'].astype(str).unique()))
            ),
            html.Br(),
            html.Label("Attrition"),
            dcc.Checklist(
                id='attrition_checklist',
                options=[{'label': v, 'value': v} for v in sorted(df['Attrition'].astype(str).unique())],
                value=list(sorted(df['Attrition'].astype(str).unique()))
            ),
        ], width=3, style={'borderRight': '1px solid #e6e6e6', 'paddingRight': '20px'}),

        dbc.Col([
            dcc.Tabs(id='tabs', value='overview_tab', children=[
                dcc.Tab(label='Overview', value='overview_tab'),
                dcc.Tab(label='Univariate', value='uni_tab'),
                dcc.Tab(label='Bivariate', value='bi_tab'),
            ]),
            html.Div(id='tab_content', style={'marginTop': '20px'})
        ], width=9)
    ])
], fluid=True)

@app.callback(
    Output('tab_content', 'children'),
    Input('tabs', 'value'),
    Input('age_min', 'value'),
    Input('age_max', 'value'),
    Input('income_min', 'value'),
    Input('income_max', 'value'),
    Input('jobrole_dropdown', 'value'),
    Input('gender_checklist', 'value'),
    Input('attrition_checklist', 'value')
)
def render_tab(tab, age_min, age_max, income_min, income_max, jobroles, genders, attritions):
    dff = filter_df(age_min, age_max, income_min, income_max, jobroles, genders, attritions)

    if tab == 'overview_tab':
        total = len(dff)
        left_count = (dff['Attrition'] == 'Left').sum()
        left_pct = 100 * left_count / total if total > 0 else 0
        avg_income = dff['Monthly Income'].mean()
        avg_age = dff['Age'].mean()

        cards = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Total Employees"), html.H3(f"{total}")])]), width=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Left (count)"), html.H3(f"{left_count} ({left_pct:.1f}%)")])]), width=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Avg Monthly Income"), html.H3(f"{avg_income:.0f}")])]), width=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Avg Age"), html.H3(f"{avg_age:.1f}")])]), width=3),
        ], className='mb-3')

        role_fig = px.histogram(dff, x='Job Role', color='Attrition', barmode='group', title='Attrition by Job Role')
        inc_fig = px.histogram(dff, x='Monthly Income', nbins=30, color='Attrition', title='Monthly Income Distribution')

        return html.Div([cards, dbc.Row([dbc.Col(dcc.Graph(figure=role_fig), width=6), dbc.Col(dcc.Graph(figure=inc_fig), width=6)])])

    elif tab == 'uni_tab':
        selectable = categorical_cols + numeric_cols
        return html.Div([
            html.H4("Univariate Analysis"),
            html.Label("Choose Feature:"),
            dcc.Dropdown(id='uni_feature', options=[{'label': c, 'value': c} for c in selectable], value=selectable[0]),
            dcc.Graph(id='uni_plot'),
            html.Div(id='uni_table')
        ])

    elif tab == 'bi_tab':
        return html.Div([
            html.H4("Bivariate Analysis"),
            html.Label("Choose X (categorical):"),
            dcc.Dropdown(id='bi_x', options=[{'label': c, 'value': c} for c in categorical_cols], value='Job Role'),
            html.Label("Choose Y (numeric):"),
            dcc.Dropdown(id='bi_y', options=[{'label': c, 'value': c} for c in numeric_cols], value='Monthly Income'),
            html.Label("Choose Plot Type:"),
            dcc.Dropdown(id='bi_type', options=[{'label': 'Box Plot', 'value': 'box'}, {'label': 'Violin Plot', 'value': 'violin'}], value='box'),
            dcc.Graph(id='bi_plot')
        ])

# Univariate plot
@app.callback(
    Output('uni_plot', 'figure'),
    Output('uni_table', 'children'),
    Input('uni_feature', 'value'),
    Input('age_min', 'value'),
    Input('age_max', 'value'),
    Input('income_min', 'value'),
    Input('income_max', 'value'),
    Input('jobrole_dropdown', 'value'),
    Input('gender_checklist', 'value'),
    Input('attrition_checklist', 'value')
)
def update_uni(feature, age_min, age_max, income_min, income_max, jobroles, genders, attritions):
    dff = filter_df(age_min, age_max, income_min, income_max, jobroles, genders, attritions)

    if feature in numeric_cols:
        fig = px.box(dff, y=feature, color='Attrition', points='all', title=f'{feature} by Attrition')
        table = dt.DataTable(data=dff[feature].describe().reset_index().to_dict('records'))
        return fig, table
    else:
        fig = px.histogram(dff, x=feature, color='Attrition', barmode='group', title=f'{feature} Distribution by Attrition')
        counts = dff[feature].value_counts().reset_index()
        counts.columns = ['Value', 'Count']
        table = dt.DataTable(data=counts.to_dict('records'), columns=[{'name': i, 'id': i} for i in counts.columns])
        return fig, table

# Bivariate plot
@app.callback(
    Output('bi_plot', 'figure'),
    Input('bi_x', 'value'),
    Input('bi_y', 'value'),
    Input('bi_type', 'value'),
    Input('age_min', 'value'),
    Input('age_max', 'value'),
    Input('income_min', 'value'),
    Input('income_max', 'value'),
    Input('jobrole_dropdown', 'value'),
    Input('gender_checklist', 'value'),
    Input('attrition_checklist', 'value')
)
def update_bi(x, y, plot_type, age_min, age_max, income_min, income_max, jobroles, genders, attritions):
    dff = filter_df(age_min, age_max, income_min, income_max, jobroles, genders, attritions)

    if plot_type == 'violin':
        fig = px.violin(dff, x=x, y=y, color='Attrition', box=True, points='all', title=f'{y} vs {x} (Violin Plot)')
    else:
        fig = px.box(dff, x=x, y=y, color='Attrition', points='all', title=f'{y} vs {x} (Box Plot)')
    return fig

if __name__ == '__main__':
    app.run()
