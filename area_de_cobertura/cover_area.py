import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import json

# Diretório de dados e título
data_dir = "/home/gui/SnippetFinder/time_cuts"
st.title("Visualizador de Gráfico com Janela Deslizante e Busca de Snippet")

# Dados iniciais para visualização de hr e motion
with open("/home/gui/SnippetFinder/area_de_cobertura/dtw_results.json", 'r') as f:
    snippet_data = json.load(f)

# Seleção de arquivo e categoria (hr ou motion) para o gráfico de barras
file_options = list(snippet_data.keys())
file_choice = st.selectbox("Selecione o arquivo para visualizar", file_options)
category_choice = st.selectbox("Selecione a categoria", ["hr", "motion"])

# Gráfico de barras para hr ou motion baseado na seleção
st.subheader("Visualização de Dados de hr e motion")
fig_bar = go.Figure()

if file_choice and category_choice:
    snippets = snippet_data[file_choice][category_choice]
    for snippet, value in snippets.items():
        fig_bar.add_trace(go.Bar(
            x=[file_choice],
            y=[value],
            name=f"{category_choice} - {snippet}",
            hoverinfo="y+name",
            marker=dict(line=dict(width=0.8))
        ))

fig_bar.update_layout(
    title=f"Dados de '{category_choice}' - {file_choice}",
    xaxis=dict(title="Arquivo"),
    yaxis=dict(title="Valor"),
    barmode="group",
    legend_title="Snippet"
)

st.plotly_chart(fig_bar, use_container_width=True)

# Seleção de arquivos CSV para dados de frequência cardíaca
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
csv_file_choice = st.selectbox("Selecione o arquivo CSV para visualizar", csv_files)

# Leitura do arquivo CSV selecionado e plotagem do gráfico completo
if csv_file_choice:
    df = pd.read_csv(os.path.join(data_dir, csv_file_choice))
    df['date'] = pd.to_datetime(df['date']).dt.time

    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(x=df["date"], y=df["HEART_RATE(bpm)"], mode="lines", name="Dados completos"))

    # Upload e busca de snippet
    uploaded_file = st.file_uploader("Faça upload do arquivo .npy com o snippet de padrão para busca", type="npy")
    matches = []
    if uploaded_file is not None:
        snippet = np.load(uploaded_file)
        snippet_size = len(snippet)
        if snippet_size > len(df):
            st.error("O snippet é maior do que o conjunto de dados. Reduza o tamanho do snippet.")
        else:
            i = 0
            while i <= len(df) - snippet_size:
                window = df["HEART_RATE(bpm)"].iloc[i:i + snippet_size].values
                if np.array_equal(window, snippet):
                    matches.append(i)
                    i += snippet_size
                else:
                    i += 1

            for match in matches:
                match_dates = df["date"].iloc[match:match + snippet_size]
                match_values = df["HEART_RATE(bpm)"].iloc[match:match + snippet_size]
                fig_full.add_trace(go.Scatter(
                    x=match_dates, y=match_values,
                    mode="markers+lines",
                    name=f"Correspondência em {match}",
                    marker=dict(symbol="star", size=10)
                ))

            st.write("Posições onde o snippet foi encontrado:", matches if matches else "Nenhuma correspondência encontrada.")
    else:
        st.write("Nenhum snippet carregado. Exibindo apenas a janela deslizante.")

    # Configurações do layout do gráfico completo
    fig_full.update_layout(
        title=f"Gráfico Completo com Correspondências - {csv_file_choice}",
        xaxis_title="date",
        yaxis_title="HEART_RATE(bpm)",
        showlegend=True
    )
    st.plotly_chart(fig_full, use_container_width=True)

    # Configurações da Janela Deslizante
    st.write("**Configurações da Janela Deslizante**")
    window_size = st.slider("Tamanho da Janela", 5, 50, 20)
    start_position = st.slider("Posição Inicial", 0, len(df) - window_size, 0)

    window_df = df[start_position:start_position + window_size]

    fig_window = go.Figure()
    fig_window.add_trace(go.Scatter(x=window_df["date"], y=window_df["HEART_RATE(bpm)"], mode="lines+markers", name="Janela Deslizante"))

    fig_window.update_layout(
        title=f"Gráfico com Janela Deslizante - {csv_file_choice}",
        xaxis_title="date",
        yaxis_title="HEART_RATE(bpm)",
        showlegend=True
    )
    st.plotly_chart(fig_window, use_container_width=True)
