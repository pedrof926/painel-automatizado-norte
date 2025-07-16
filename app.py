import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import requests
import geopandas as gpd
import plotly.express as px
import time
import os

# Diretório base (para Render usar o atual)
DIR_BASE = "."

# Arquivos locais
ARQ_CODIGOS = f"{DIR_BASE}/codigos_mun_norte.xlsx"
ARQ_HISTORICO = f"{DIR_BASE}/temperatura_30_dias_municipios_corrigido_completo.xlsx"
ARQ_PERCENTIS = f"{DIR_BASE}/ehf_percentis_norte_painel.xlsx"
ARQ_GEOJSON = f"{DIR_BASE}/municipios_norte_simplificado.geojson"

# Ler dados
df_codigos = pd.read_excel(ARQ_CODIGOS)
df_hist = pd.read_excel(ARQ_HISTORICO)
df_percentis = pd.read_excel(ARQ_PERCENTIS)
gdf_shape = gpd.read_file(ARQ_GEOJSON)

# Padronizar nome municípios (maiúsculo)
df_codigos['NM_MUN'] = df_codigos['NM_MUN'].str.upper()
df_hist['NM_MUN'] = df_hist['NM_MUN'].str.upper()
df_percentis['NM_MUN'] = df_percentis['NM_MUN'].str.upper()
gdf_shape['NM_MUN'] = gdf_shape['NM_MUN'].str.upper()

# Média móvel 30 dias histórica
tmean_30d = df_hist.groupby('NM_MUN')['Tmedia'].mean().reset_index()
tmean_30d.rename(columns={'Tmedia': 'Tmean_30d'}, inplace=True)

# Consulta API INMET (timeout maior)
def consulta_previsao_inmet(codigo_municipio):
    url = f"https://apiprevmet3.inmet.gov.br/previsao/{codigo_municipio}"
    try:
        resposta = requests.get(url, timeout=30)
        resposta.raise_for_status()
        dados = resposta.json()

        if str(codigo_municipio) not in dados:
            print(f"INMET: Código {codigo_municipio} não encontrado.")
            return pd.DataFrame(columns=['NM_MUN', 'Data', 'Tmedia'])

        municipio_data = dados[str(codigo_municipio)]
        resultados = []

        for data_str, periodos in municipio_data.items():
            tmaxs, tmins = [], []
            for periodo in ['manha', 'tarde', 'noite']:
                if periodo in periodos:
                    tmax = periodos[periodo].get('temp_max')
                    tmin = periodos[periodo].get('temp_min')
                    if tmax is not None:
                        tmaxs.append(tmax)
                    if tmin is not None:
                        tmins.append(tmin)

            if tmaxs and tmins:
                tmax_media = np.mean(tmaxs)
                tmin_media = np.mean(tmins)
                tmedia_diaria = (tmax_media + tmin_media) / 2
            else:
                tmedia_diaria = np.nan

            resultados.append({
                'NM_MUN': '',  # preenche depois
                'Data': pd.to_datetime(data_str, dayfirst=True),
                'Tmedia': tmedia_diaria
            })

        df = pd.DataFrame(resultados)
        df['NM_MUN'] = df_codigos.loc[df_codigos['CD_MUN'] == codigo_municipio, 'NM_MUN'].values[0]
        return df

    except Exception as e:
        print(f"Erro INMET {codigo_municipio}: {e}")
        return pd.DataFrame(columns=['NM_MUN', 'Data', 'Tmedia'])

# Buscar previsões sequencialmente, delay alto pra evitar erro 429
def obter_previsoes_todos(df_codigos):
    resultados = []
    for codigo in df_codigos['CD_MUN']:
        df_tmp = consulta_previsao_inmet(codigo)
        if not df_tmp.empty:
            resultados.append(df_tmp)
        print(f"Requisição feita para município {codigo}, aguardando 10 segundos...")
        time.sleep(10)  # delay de 10 segundos entre requisições
    return pd.concat(resultados, ignore_index=True)

print("Buscando previsões INMET para todos os municípios, aguarde...")
df_previsoes = obter_previsoes_todos(df_codigos)
print("Previsões carregadas.")

# Média móvel 3 dias da previsão
df_previsoes = df_previsoes.sort_values(['NM_MUN', 'Data'])
df_previsoes['Tmedia_3d'] = df_previsoes.groupby('NM_MUN')['Tmedia'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

# Juntar com histórico e percentis para cálculo do EHF
df_previsoes = df_previsoes.merge(tmean_30d, on='NM_MUN', how='left')
df_previsoes = df_previsoes.merge(df_percentis[['NM_MUN', 'Tmédia_p95', 'p95_EHF', 'p98_EHF', 'p99_EHF']], on='NM_MUN', how='left')

# Calcular os índices do EHF
def calcula_ehf(row):
    if pd.isna(row['Tmedia_3d']) or pd.isna(row['Tmédia_p95']):
        return np.nan
    if row['Tmedia_3d'] < row['Tmédia_p95']:
        return 0.0
    ehi_accl = row['Tmedia_3d'] - row['Tmean_30d']
    ehi_sig = row['Tmedia_3d'] - row['Tmédia_p95']
    ehf = ehi_accl * max(0, ehi_sig)
    return ehf

df_previsoes['EHF'] = df_previsoes.apply(calcula_ehf, axis=1)

# Classificar EHF
def classificar_ehf(row):
    if pd.isna(row['EHF']):
        return 'Sem Dados'
    if row['EHF'] == 0:
        return 'Normal'
    if row['EHF'] < row['p95_EHF']:
        return 'Normal'
    elif row['EHF'] < row['p98_EHF']:
        return 'Severo'
    elif row['EHF'] >= row['p99_EHF']:
        return 'Extremo'
    else:
        return 'Normal'

df_previsoes['Classificacao'] = df_previsoes.apply(classificar_ehf, axis=1)

# Merge com GeoJSON simplificado
gdf_final = gdf_shape.merge(df_previsoes, on='NM_MUN', how='left')

# Criar app Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Painel EHF Região Norte - Previsão INMET"),
    dcc.Dropdown(
        id='select-date',
        options=[{'label': d.strftime('%d/%m/%Y'), 'value': str(d)} for d in sorted(df_previsoes['Data'].unique())],
        value=str(sorted(df_previsoes['Data'].unique())[0])
    ),
    dcc.Graph(id='map-ehf'),
    dcc.Graph(id='graph-municipio'),
    html.Div(id='info-municipio')
])

@app.callback(
    Output('map-ehf', 'figure'),
    Input('select-date', 'value')
)
def update_map(selected_date):
    selected_date = pd.to_datetime(selected_date)
    gdf_date = gdf_final[gdf_final['Data'] == selected_date]

    fig = px.choropleth_mapbox(
        gdf_date,
        geojson=gdf_date.geometry.__geo_interface__,
        locations=gdf_date.index,
        color='Classificacao',
        hover_name='NM_MUN',
        hover_data=['EHF', 'Tmedia_3d', 'Classificacao'],
        mapbox_style="carto-positron",
        zoom=4,
        center={"lat": -3.5, "lon": -53.0},
        opacity=0.7,
        category_orders={"Classificacao": ["Normal", "Severo", "Extremo", "Sem Dados"]},
        color_discrete_map={
            "Normal": "green",
            "Severo": "orange",
            "Extremo": "red",
            "Sem Dados": "gray"
        }
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@app.callback(
    Output('graph-municipio', 'figure'),
    Output('info-municipio', 'children'),
    Input('map-ehf', 'clickData')
)
def update_graph(clickData):
    if not clickData:
        return {}, "Clique em um município no mapa para ver a evolução."

    idx = clickData['points'][0]['location']
    mun = gdf_final.loc[idx, 'NM_MUN']

    df_mun = df_previsoes[df_previsoes['NM_MUN'] == mun].sort_values('Data')

    fig = px.line(df_mun, x='Data', y=['EHF', 'Tmedia_3d'], labels={'value': 'Valor', 'Data': 'Data'}, title=f'Evolução EHF e Tmedia - {mun}')
    info_text = f'Município: {mun}'

    return fig, info_text

app = dash.Dash(__name__)
server = app.server  # ESSENCIAL para Render/Gunicorn

# Seu layout e callbacks aqui

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(host='0.0.0.0', port=port, debug=True)

























