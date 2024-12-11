import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from scipy.stats import mode
from sklearn.cluster import MiniBatchKMeans
from numpy.linalg import norm
from time import time
import matplotlib.pyplot as plt

# Funções para processamento e clustering
def get_freq(df):
    time_idx = df.time
    mean_sample_interval = time_idx.rolling(2).aggregate(lambda ts: ts.values[1] - ts.values[0]).mean()
    return 1 / mean_sample_interval

def get_clustered_profiles(T, m, num_clusters, selection_method="medoid", num_nearest=3):
    segments = np.array([T[i: i + m] for i in range(len(T) - m + 1)])
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0).fit(segments)
    centroids = kmeans.cluster_centers_

    closest_snippets = []
    for center_idx, center in enumerate(centroids):
        distances = np.linalg.norm(segments - center, axis=1)
        
        if selection_method == "medoid":
            closest_snippet_idx = np.argmin(distances)
            closest_snippets.append(closest_snippet_idx)
        
        elif selection_method == "mean":
            nearest_indices = np.argsort(distances)[:num_nearest]
            mean_index = int(np.mean(nearest_indices))
            closest_snippets.append(mean_index)
        
        elif selection_method == "mode":
            cluster_indices = np.where(kmeans.labels_ == center_idx)[0]
            closest_snippet_idx = mode(cluster_indices).mode[0]
            closest_snippets.append(closest_snippet_idx)
    
    D = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(num_clusters):
            D[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    return D, kmeans.labels_, closest_snippets

def find_snippet_positions(ts, snippet, subseq_size, tolerance=0):
    positions = []
    snippet_size = len(snippet)
    i = 0
    while i <= len(ts) - snippet_size:
        window = ts[i:i + snippet_size]
        # Verificar se todos os valores da janela estão dentro da tolerância em relação ao snippet
        if np.all(np.abs(window - snippet) <= tolerance):
            positions.append(i)
            i += snippet_size
        else:
            i += 1
    return positions

# Interface do Streamlit
st.title("Snippet Analysis with Full Time Series Visualization")

# Listar arquivos na pasta 'time_cuts'
data_dir = "/home/gui/SnippetFinder/time_cuts"
available_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
selected_file = st.selectbox("Selecione o arquivo CSV", available_files)

# Carregar o arquivo selecionado
if selected_file:
    df = pd.read_csv(os.path.join(data_dir, selected_file))
    df['time'] = pd.to_datetime(df['timestamp']).dt.time
    df['bpm'] = df['HEART_RATE(bpm)'].replace(0.0, np.nan)
    df.dropna(subset=['bpm'], inplace=True)

    # Seleção de parâmetros
    series_col = st.selectbox("Select Time Series", ["bpm", "magnitude"])
    calculation_method = st.selectbox("Calculation method for Motion", ["norm", "enmo"]) if series_col == "magnitude" else None
    selection_method = st.selectbox("Snippets Selection Method", ["medoid", "mean", "mode"])
    num_clusters = st.slider("Number of Clusters", 2, 20, 5)
    num_nearest = st.slider("Number of closest Snippets", 1, 10, 3) if selection_method == "mean" else None
    k = st.slider("Number of Snippets(k)", 1, 10, 3)
    subseq_size = st.slider("Subsequence Size", 5, 50, 15)
    
    # Slider para o valor de tolerância
    tolerance = st.slider("Set the tolerance value", 0, 20, 5)

    if st.button("Analyze"):
        # Seleção do método de cálculo para o motion
        if series_col == "magnitude":
            df["magnitude"] = df[["ACC_X(m/s^2)", "ACC_Y", "ACC_Z"]].apply(
                lambda x: max(norm(x) - 1, 0) if calculation_method == "enmo" else norm(x),
                axis=1
            )
            ts = df["magnitude"].values.astype(float)
        else:
            ts = df["bpm"].values.astype(float)

        # Cálculo dos snippets e áreas de cobertura
        start = time()
        D, labels, closest_snippets = get_clustered_profiles(ts, subseq_size, num_clusters, selection_method, num_nearest)
        snippets, snippet_profiles, areas = [], [], []
        Q = np.full((1, D.shape[1]), np.inf)
        # Armazenar as áreas de cobertura de cada snippet
        coverage_areas = []
        Q_evolution = [Q.copy()] 

        for _ in range(k):
            minimum_area = np.inf
            index_min = -1

            for i in range(D.shape[0]):
                # Calcula a área de cobertura atual como a interseção mínima
                profile_area = np.sum(np.minimum(D[i, :], Q))
                
                # Seleciona o snippet com a menor área de cobertura
                if profile_area < minimum_area:
                    minimum_area = profile_area
                    index_min = i

            if index_min == -1:
                break

            Q = np.minimum(D[index_min, :], Q)
            Q_evolution.append(Q.copy())  
            
            coverage_areas.append(minimum_area)
            representative_idx = closest_snippets[index_min]
            snippets.append((representative_idx, minimum_area))
            snippet_profiles.append(D[index_min, :])
            areas.append(minimum_area)

        # Cálculo do tempo de computação e área total
        total_area = sum(areas)
        percent_areas = [area / total_area * 100 for area in areas]
        comp_time = time() - start

        # Exibição das áreas de cobertura e tempo de computação
        st.write(f"Computation Time: {comp_time:.2f}s")

        # Visualizar a área de cobertura para cada snippet
        fig_coverage_area = go.Figure()
        fig_coverage_area.add_trace(go.Bar(
            x=[f"Snippet {i}" for i in range(len(coverage_areas))],
            y=coverage_areas,
            name="Coverage Area",
            marker_color="green"
        ))
        fig_coverage_area.update_layout(
            title="Coverage Area for Each Snippet (Minimum Intersection)",
            xaxis_title="Snippet",
            yaxis_title="Coverage Area (Minimum Intersection)",
        )
        st.plotly_chart(fig_coverage_area, use_container_width=True)

        fig_q_evolution = go.Figure()
        for idx, q_vector in enumerate(Q_evolution):
            fig_q_evolution.add_trace(go.Scatter(
                x=list(range(len(q_vector[0]))),
                y=q_vector[0],
                mode="lines+markers",
                name=f"Iteração {idx}"
            ))

        fig_q_evolution.update_layout(
            title="Evolution of Q (Minimum Coverage) Over Iterations",
            xaxis_title="Clusters",
            yaxis_title="Distance",
        )
        st.plotly_chart(fig_q_evolution, use_container_width=True)

        time_coverages_direct = []
        for i, (snippet_idx, _) in enumerate(snippets):
            snippet_data = ts[snippet_idx:snippet_idx + subseq_size]
            # Encontrar correspondências diretas
            direct_positions = find_snippet_positions(ts, snippet_data, subseq_size, tolerance=0)
            # Calcula a cobertura de tempo
            covered_points_direct = len(direct_positions) * subseq_size
            coverage_percentage_direct = (covered_points_direct / len(ts)) * 100 if len(ts) > 0 else 0
            time_coverages_direct.append(coverage_percentage_direct)

        fig_time_coverage_direct = go.Figure()
        fig_time_coverage_direct.add_trace(go.Bar(
            x=[f"Snippet {i}" for i in range(len(time_coverages_direct))],
            y=time_coverages_direct,
            name="Time Coverage (%) - Direct",
            marker_color="blue"
        ))
        fig_time_coverage_direct.update_layout(
            title="Time Coverage for Each Snippet (Direct Matches)",
            xaxis_title="Snippet",
            yaxis_title="Coverage Percentage (%)",
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_time_coverage_direct, use_container_width=True)

        # Gráfico da série temporal completa com destaque apenas para o snippet com maior área de cobertura
        max_area_index = np.argmax(areas)
        selected_snippet = ts[snippets[max_area_index][0]:snippets[max_area_index][0] + subseq_size]
        positions = find_snippet_positions(ts, selected_snippet, subseq_size, tolerance=tolerance)

        fig_full = go.Figure()
        fig_full.add_trace(go.Scatter(x=df["time"], y=ts, mode="lines", name="Complete Time Serie", line=dict(color="white")))

        for pos in positions:
            match_dates = df["time"].iloc[pos:pos + subseq_size]
            match_values = ts[pos:pos + subseq_size]
            fig_full.add_trace(go.Scatter(
                x=match_dates, y=match_values,
                mode="lines",
                name="Correspondência do Snippet de Maior Cobertura",
                line=dict(color="blue", width=3)
            ))

        fig_full.update_layout(
            title=f"Full Graph with Largest Area Snippet Match - {selected_file}",
            xaxis_title="time",
            yaxis_title="Time Serie Value",
            showlegend=True
        )
        st.plotly_chart(fig_full, use_container_width=True)
