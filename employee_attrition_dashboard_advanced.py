import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv(r"C:\Data Science Project\Data_Dashboard.csv")

# Basic cleaning: trim column names
df.columns = [c.strip() for c in df.columns]

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
# Try to coerce numeric-like object columns
for col in df.columns:
    if col not in numeric_cols:
        # attempt convert
        try:
            coerced = pd.to_numeric(df[col], errors='coerce')
            if coerced.notna().sum() / len(df) > 0.8:
                df[col] = coerced
                numeric_cols.append(col)
        except Exception:
            pass

categorical_cols = [c for c in df.columns if c not in numeric_cols]

# Standard encoders for modeling & PCA
le_dict = {}
for c in categorical_cols:
    try:
        le = LabelEncoder()
        # fillna with string for encoding
        df[c] = df[c].fillna('Missing').astype(str)
        le.fit(df[c])
        le_dict[c] = le
    except Exception:
        pass

# App setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Helper to produce filtered dataframe
def filter_df(age_range, income_range, selected_roles, selected_genders, selected_attrition):
    d = df.copy()
    if 'Age' in d.columns and age_range is not None:
        d = d[(d['Age'] >= age_range[0]) & (d['Age'] <= age_range[1])]
    if 'Monthly Income' in d.columns and income_range is not None:
        d = d[(d['Monthly Income'] >= income_range[0]) & (d['Monthly Income'] <= income_range[1])]
    if selected_roles:
        d = d[d['Job Role'].isin(selected_roles)]
    if selected_genders:
        d = d[d['Gender'].isin(selected_genders)]
    if selected_attrition:
        d = d[d['Attrition'].isin(selected_attrition)]
    return d

# Layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Advanced Employee Attrition Dashboard", className='text-center my-3 text-white'), style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'padding': '20px', 'borderRadius': '12px', 'boxShadow': '0 10px 30px rgba(0,0,0,0.1)'})),
    dbc.Row([
        dbc.Col([
            html.H5("Filters"),
            html.Label("Age range"),
            dcc.RangeSlider(
                id='age_slider',
                min=int(df['Age'].min()) if 'Age' in df.columns else 0,
                max=int(df['Age'].max()) if 'Age' in df.columns else 100,
                step=1,
                marks= [1,2,3,4,5,6],
                value=[int(df['Age'].min()), int(df['Age'].max())] if 'Age' in df.columns else [0,100],
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Br(),
            html.Label("Monthly Income range"),
            dcc.RangeSlider(
                id='income_slider',
                min=int(df['Monthly Income'].min()) if 'Monthly Income' in df.columns else 0,
                max=int(df['Monthly Income'].max()) if 'Monthly Income' in df.columns else 1,
                step=10000,
                value=[int(df['Monthly Income'].min()), int(df['Monthly Income'].max())] if 'Monthly Income' in df.columns else [0,1],
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Br(),
            html.Label("Job Role"),
            dcc.Dropdown(
                id='jobrole_dropdown',
                options=[{'label': v, 'value': v} for v in sorted(df['Job Role'].astype(str).unique())],
                multi=True,
                placeholder="Filter by job role"
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
            html.Hr(),
            html.Button("Reset filters", id='reset_button', n_clicks=0, className='btn btn-secondary')
        ], width=3, style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '12px', 'boxShadow': '0 10px 30px rgba(0,0,0,0.1)'}),
        dbc.Col([
            dcc.Tabs(id='tabs', value='overview_tab', children=[
                dcc.Tab(label='Overview', value='overview_tab'),
                dcc.Tab(label='Univariate', value='uni_tab'),
                dcc.Tab(label='Bivariate', value='bi_tab'),
                dcc.Tab(label='Multivariate', value='multi_tab'),
                dcc.Tab(label='Correlations', value='corr_tab'),
                dcc.Tab(label='Model (Feature Importance)', value='model_tab'),
            ]),
            html.Div(id='tab_content', style={'marginTop': '20px'})
        ], width=9)
    ])
], fluid=True)

# Callbacks for resetting filters
@app.callback(
    Output('age_slider', 'value'),
    Output('income_slider', 'value'),
    Output('jobrole_dropdown', 'value'),
    Output('gender_checklist', 'value'),
    Output('attrition_checklist', 'value'),
    Input('reset_button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n):
    return (
        [int(df['Age'].min()), int(df['Age'].max())] if 'Age' in df.columns else [0,100],
        [int(df['Monthly Income'].min()), int(df['Monthly Income'].max())] if 'Monthly Income' in df.columns else [0,1],
        None,
        list(sorted(df['Gender'].astype(str).unique())),
        list(sorted(df['Attrition'].astype(str).unique()))
    )

# Main content callback
@app.callback(
    Output('tab_content', 'children'),
    Input('tabs', 'value'),
    Input('age_slider', 'value'),
    Input('income_slider', 'value'),
    Input('jobrole_dropdown', 'value'),
    Input('gender_checklist', 'value'),
    Input('attrition_checklist', 'value')
)
def render_tab(tab, age_range, income_range, jobroles, genders, attritions):
    dff = filter_df(age_range, income_range, jobroles, genders, attritions)

    if tab == 'overview_tab':
        # KPI cards and basic charts
        total = len(dff)
        left_count = (dff['Attrition'] == 'Left').sum() if 'Attrition' in dff.columns else 0
        left_pct = 100 * left_count / total if total>0 else 0
        avg_income = dff['Monthly Income'].mean() if 'Monthly Income' in dff.columns else None
        avg_age = dff['Age'].mean() if 'Age' in dff.columns else None

        cards = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Total Employees"), html.H3(f"{total}")])]), width=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Left (count)"), html.H3(f"{left_count} ({left_pct:.1f}%)")])]), width=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Avg Monthly Income"), html.H3(f"{avg_income:.0f}" if avg_income==avg_income else 'N/A')])]), width=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Avg Age"), html.H3(f"{avg_age:.1f}" if avg_age==avg_age else 'N/A')])]), width=3),
        ], className='mb-3')

        # Attrition by Job Role
        role_fig = px.histogram(dff, x='Job Role', color='Attrition', barmode='group', title='Attrition by Job Role') if 'Job Role' in dff.columns else go.Figure()
        # Monthly income distribution
        inc_fig = px.histogram(dff, x='Monthly Income', nbins=30, title='Monthly Income distribution', color='Attrition') if 'Monthly Income' in dff.columns else go.Figure()

        return html.Div([cards, dbc.Row([dbc.Col(dcc.Graph(figure=role_fig), width=6), dbc.Col(dcc.Graph(figure=inc_fig), width=6)])])

    elif tab == 'uni_tab':
        # Univariate: boxplot and bar for selected feature
        selectable = categorical_cols + numeric_cols
        feature_dropdown = dcc.Dropdown(id='uni_feature', options=[{'label':c,'value':c} for c in selectable], value=selectable[0])
        return html.Div([
            html.H4("Univariate analysis"),
            html.Div([html.Label("Choose feature"), feature_dropdown]),
            dcc.Graph(id='uni_plot'),
            html.Div(id='uni_stats')
        ])

    elif tab == 'bi_tab':
        # Bivariate: scatter with color & size + grouped bar
        return html.Div([
            html.H4("Bivariate analysis"),
            dbc.Row([
                dbc.Col([html.Label("X (numeric)"), dcc.Dropdown(id='bi_x', options=[{'label':c,'value':c} for c in numeric_cols], value=numeric_cols[0])], width=4),
                dbc.Col([html.Label("Y (numeric)"), dcc.Dropdown(id='bi_y', options=[{'label':c,'value':c} for c in numeric_cols], value=numeric_cols[1] if len(numeric_cols)>1 else numeric_cols[0])], width=4),
                dbc.Col([html.Label("Color (categorical)"), dcc.Dropdown(id='bi_color', options=[{'label':c,'value':c} for c in categorical_cols], value='Attrition' if 'Attrition' in categorical_cols else (categorical_cols[0] if categorical_cols else None))], width=4)
            ]),
            dcc.Graph(id='bi_scatter'),
            html.Hr(),
            html.H5("Grouped counts"),
            dcc.Dropdown(id='group_cat', options=[{'label':c,'value':c} for c in categorical_cols], value='Job Role' if 'Job Role' in categorical_cols else (categorical_cols[0] if categorical_cols else None)),
            dcc.Graph(id='grouped_bar')
        ])

    elif tab == 'multi_tab':
        # Multivariate: 3D scatter, parallel coords, PCA
        return html.Div([
            html.H4("Multivariate analysis"),
            dbc.Row([
                dbc.Col([html.Label("3D: X"), dcc.Dropdown(id='m3_x', options=[{'label':c,'value':c} for c in numeric_cols], value=numeric_cols[0])], width=4),
                dbc.Col([html.Label("3D: Y"), dcc.Dropdown(id='m3_y', options=[{'label':c,'value':c} for c in numeric_cols], value=numeric_cols[1] if len(numeric_cols)>1 else numeric_cols[0])], width=4),
                dbc.Col([html.Label("3D: Z"), dcc.Dropdown(id='m3_z', options=[{'label':c,'value':c} for c in numeric_cols], value=numeric_cols[2] if len(numeric_cols)>2 else numeric_cols[0])], width=4),
            ]),
            dcc.Graph(id='scatter3d'),
            html.Hr(),
            html.H5("Parallel Coordinates (select up to 6 numeric features)"),
            dcc.Dropdown(id='par_coords', options=[{'label':c,'value':c} for c in numeric_cols], value=numeric_cols[:4], multi=True),
            dcc.Graph(id='parcoords'),
            html.Hr(),
            html.H5("PCA projection (2 components)"),
            dcc.Graph(id='pca_scatter')
        ])

    elif tab == 'corr_tab':
        # Correlation heatmap and scatter matrix
        corr = dff[numeric_cols].corr()
        heatmap = px.imshow(corr, text_auto=True, title='Correlation matrix (numeric features)') if not corr.empty else go.Figure()
        sm = px.scatter_matrix(dff[numeric_cols].dropna().sample(n=min(300, len(dff))), dimensions=numeric_cols[:6] if len(numeric_cols)>6 else numeric_cols, color=dff['Attrition'] if 'Attrition' in dff.columns else None, title='Scatter matrix (sample)') if numeric_cols else go.Figure()
        return html.Div([dcc.Graph(figure=heatmap), html.Hr(), dcc.Graph(figure=sm)])

    elif tab == 'model_tab':
        # Feature importance with RandomForest (simple)
        # Prepare training data (encode categoricals)
        model_df = dff.copy().dropna(subset=['Attrition']) if 'Attrition' in dff.columns else dff.copy()
        X = pd.DataFrame()
        for c in model_df.columns:
            if c == 'Attrition':
                continue
            if c in numeric_cols:
                X[c] = model_df[c].fillna(model_df[c].median())
            else:
                X[c] = le_dict.get(c).transform(model_df[c].astype(str)) if (c in le_dict and model_df[c].notnull().any()) else 0
        y = le_dict['Attrition'].transform(model_df['Attrition'].astype(str)) if 'Attrition' in le_dict else None

        if y is None or X.shape[0] < 20:
            return html.Div([html.H5('Not enough data to train a model or Attrition not available')])

        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        feats = X.columns.tolist()
        imp_df = pd.DataFrame({'feature':feats, 'importance':importances}).sort_values('importance', ascending=False).head(30)
        fig_imp = px.bar(imp_df, x='importance', y='feature', orientation='h', title='Feature importance (RandomForest)')
        return html.Div([dcc.Graph(figure=fig_imp), html.Div([html.P('Model trained on filtered data. Use filters to explore how importance changes.')])])

    return html.Div()

# Additional callbacks for dynamic visuals inside tabs
# Univariate plot
@app.callback(
    Output('uni_plot', 'figure'),
    Output('uni_stats', 'children'),
    Input('uni_feature', 'value'),
    Input('age_slider', 'value'),
    Input('income_slider', 'value'),
    Input('jobrole_dropdown', 'value'),
    Input('gender_checklist', 'value'),
    Input('attrition_checklist', 'value')
)
def update_uni(feature, age_range, income_range, jobroles, genders, attritions):
    dff = filter_df(age_range, income_range, jobroles, genders, attritions)
    if feature in numeric_cols:
        fig = px.box(dff, x='Attrition', y=feature, points='all', title=f'Box plot of {feature} by Attrition' if 'Attrition' in dff.columns else f'Box plot of {feature}')
        stats = dff[feature].describe().to_frame().to_html()
        return fig, html.Div([html.H5('Summary stats'), html.Div(dash.dash_table.DataTable(data=dff[feature].describe().reset_index().to_dict('records')))])
    else:
        fig = px.histogram(dff, x=feature, color='Attrition', barmode='group', title=f'Distribution of {feature} by Attrition')
        counts = dff[feature].value_counts().reset_index().rename(columns={'index':feature, feature:'count'})
        return fig, html.Div([html.H5('Counts'), dash.dash_table.DataTable(data=counts.to_dict('records'))])

# Bivariate callbacks
@app.callback(
    Output('bi_scatter', 'figure'),
    Output('grouped_bar', 'figure'),
    Input('bi_x', 'value'),
    Input('bi_y', 'value'),
    Input('bi_color', 'value'),
    Input('age_slider', 'value'),
    Input('income_slider', 'value'),
    Input('jobrole_dropdown', 'value'),
    Input('gender_checklist', 'value'),
    Input('attrition_checklist', 'value')
)
def update_bi(x, y, color, age_range, income_range, jobroles, genders, attritions):
    dff = filter_df(age_range, income_range, jobroles, genders, attritions)

    if x not in dff.columns or y not in dff.columns:
        return go.Figure(), go.Figure()

    scatter = px.scatter(dff, x=x, y=y, color=color if color in dff.columns else None, hover_data=dff.columns, title=f'{y} vs {x} colored by {color}')
    # grouped bar
    group_cat = 'Attrition' if 'Attrition' in dff.columns else None
    if color in dff.columns:
        grouped = px.histogram(dff, x=color, color=group_cat, barmode='group', title=f'Counts of {color} by {group_cat}')
    else:
        grouped = go.Figure()
    return scatter, grouped

# 3D scatter + parallel coords + PCA
@app.callback(
    Output('scatter3d', 'figure'),
    Output('parcoords', 'figure'),
    Output('pca_scatter', 'figure'),
    Input('m3_x', 'value'),
    Input('m3_y', 'value'),
    Input('m3_z', 'value'),
    Input('par_coords', 'value'),
    Input('age_slider', 'value'),
    Input('income_slider', 'value'),
    Input('jobrole_dropdown', 'value'),
    Input('gender_checklist', 'value'),
    Input('attrition_checklist', 'value')
)
def update_multi(x, y, z, par_features, age_range, income_range, jobroles, genders, attritions):
    dff = filter_df(age_range, income_range, jobroles, genders, attritions)
    # 3D scatter
    fig3d = go.Figure()
    if all([c in dff.columns for c in [x,y,z]]):
        fig3d = px.scatter_3d(dff, x=x, y=y, z=z, color='Attrition' if 'Attrition' in dff.columns else None, hover_data=dff.columns, title=f'3D: {x} / {y} / {z}')

    # parallel coordinates
    parfig = go.Figure()
    if par_features and len(par_features) > 0:
        # limit to numeric values
        pf = [c for c in par_features if c in numeric_cols]
        if pf:
            parfig = px.parallel_coordinates(dff.dropna(subset=pf), dimensions=pf, color=le_dict['Attrition'].transform(dff['Attrition'].astype(str)) if 'Attrition' in le_dict else None)

    # PCA projection
    pca_fig = go.Figure()
    try:
        X = dff[numeric_cols].fillna(dff[numeric_cols].median())
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        pca = PCA(n_components=2)
        comps = pca.fit_transform(Xs)
        pca_df = pd.DataFrame(comps, columns=['PC1','PC2'])
        if 'Attrition' in dff.columns:
            pca_df['Attrition'] = dff['Attrition'].astype(str).values
        pca_fig = px.scatter(pca_df, x='PC1', y='PC2', color='Attrition' if 'Attrition' in pca_df.columns else None, title='PCA (2 components)')
    except Exception as e:
        pca_fig = go.Figure()

    return fig3d, parfig, pca_fig

if __name__ == '__main__':
    app.run()
