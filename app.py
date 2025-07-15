import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import requests
import geopandas as gpd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
import time
import json
import os

DIR_BASE = "."

ARQ_CODIGOS = "./codigos_mun_norte.xlsx"
ARQ_HISTORICO = "./temperatura_30_dias_municipios_corrigido_completo.xlsx"
ARQ_PERCENTIS = "./ehf_percentis_norte_painel.xlsx"
ARQ_GEOJSON = "./municipios_norte_simplificado.geojson"

# Carregar dados locais
df_codigos = pd.read_excel(ARQ_CODIGOS)
df_hist = pd.read_excel(ARQ_HISTORICO)
df_percentis = pd.read_excel(ARQ_PERCENTIS)

# Carregar GeoJSON (no lugar do shapefile)
with open(ARQ_GEOJSON, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

# Padronizar nomes dos municípios (maiúsculo)
df_codigos['NM_MUN'] = df_codigos['NM_MUN'].str.upper()
df_hist['NM_MUN'] = df_hist['NM_MUN'].str.upper()
df_percentis['NM_MUN'] = df_percentis['NM_MUN'].str.upper()
# GeoJSON tem o campo NM_MUN dentro das propriedades (não no GeoDataFrame)
# Para merge, vamos montar um DataFrame com NM_MUN único do geojson:
geojson_municipios = [feature['properties']['NM_MUN'].upper() for feature in geojson_data['features']]
df_geojson = pd.DataFrame({'NM_MUN': geojson_municipios})

# Calcular média móvel 30 dias histórica
tmean_30d = df_hist.groupby('NM_MUN')['Tmedia'].mean().reset_index()
tmean_30d.rename(columns={'Tmedia': 'Tmean_30d'}, inplace=True)

# Função para consultar API INMET para um município (formato simples)
def consulta_previsao_inmet(codigo_municipio):
    url = f"https://apiprevmet3.inmet.gov.br/previsao/{codigo_municipio}"
    try:
        resposta = requests.get(url, timeout=10)
        resposta.raise_for_status()
        dados = resposta.json()

        if str(codigo_municipio) not in dados:
            print(f"INMET: Código {codigo_municipio} não encontrado.")
            return pd.DataFrame(columns=['NM_MUN', 'Data', 'Tmedia'])

        municipio_data = dados[str(codigo_municipio)]
        resultados = []

        for data_str, periodos in municipio_data.items():
            tmaxs = []
            tmins = []
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
                'NM_MUN': '',  # vamos preencher depois
                'Data': pd.to_datetime(data_str, dayfirst=True),
                'Tmedia': tmedia_diaria
            })

        df = pd.DataFrame(resultados)
        df['NM_MUN'] = df_codigos.loc[df_codigos['CD_MUN'] == codigo_municipio, 'NM_MUN'].values[0]
        return df

    except Exception as e:
        print(f"Erro INMET {codigo_municipio}: {e}")
        return pd.DataFrame(columns=['NM_MUN', 'Data', 'Tmedia'])

# Buscar previsões para todos municípios em paralelo com delay para evitar erro 429
def obter_previsoes_todos(df_codigos):
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for codigo in df_codigos['CD_MUN']:
            futures.append(executor.submit(consulta_previsao_inmet, codigo))
            time.sleep(0.3)  # Delay de 300ms entre requisições para evitar bloqueio
        for future in futures:
            df_tmp = future.result()
            if not df_tmp.empty:
                resultados.append(df_tmp)
    return pd.concat(resultados, ignore_index=True)

# Obter as previsões ao iniciar o app (pode demorar)
print("Buscando previsões INMET para todos os municípios, aguarde...")
df_previsoes = obter_previsoes_todos(df_codigos)
print("Previsões carregadas.")

# Média móvel 3 dias da previsão
df_previsoes = df_previsoes.sort_values(['NM_MUN', 'Data'])
df_previsoes['Tmedia_3d'] = df_previsoes.groupby('NM_MUN')['Tmedia'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

# Juntar com histórico e percentis para cálculo do EHF
df_previsoes = df_previsoes.merge(tmean_30d, on='NM_MUN', how='left')
df_previsoes = df_previsoes.merge(df_percentis[['NM_MUN', 'Tmédia_p95', 'p95_EHF', 'p98_EHF', 'p99_EHF']], on='NM_MUN', how='left')

# Filtrar df_previsoes para municípios que existem no GeoJSON (evita problemas de chave)
df_previsoes = df_previsoes[df_previsoes['NM_MUN'].isin(df_geojson['NM_MUN'])]

# Calcular os índices do EHF com condição de Tmedia_3d >= Tmédia_p95
def calcula_ehf(row):
    if pd.isna(row['Tmedia_3d']) or pd.isna(row['Tmédia_p95']):
        return np.nan
    if row['Tmedia_3d'] < row['Tmédia_p95']:
        return 0.0  # sem risco, EHF = 0
    ehi_accl = row['Tmedia_3d'] - row['Tmean_30d']
    ehi_sig = row['Tmedia_3d'] - row['Tmédia_p95']
    ehf = ehi_accl * max(0, ehi_sig)
    return ehf

df_previsoes['EHF'] = df_previsoes.apply(calcula_ehf, axis=1)

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

# Agora não faz merge com GeoDataFrame, o geojson é usado diretamente no plotly

# Montar app Dash
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
    df_date = df_previsoes[df_previsoes['Data'] == selected_date]

    fig = px.choropleth_mapbox(
        df_date,
        geojson=geojson_data,
        locations='NM_MUN',
        featureidkey='properties.NM_MUN',
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

    mun = clickData['points'][0]['location']

    df_mun = df_previsoes[df_previsoes['NM_MUN'] == mun].sort_values('Data')

    fig = px.line(df_mun, x='Data', y=['EHF', 'Tmedia_3d'],
                  labels={'value': 'Valor', 'Data': 'Data'},
                  title=f'Evolução EHF e Tmedia - {mun}')
    info_text = f'Município: {mun}'

    return fig, info_text

if __name__ == '__main__':
    app.run(debug=True)





















