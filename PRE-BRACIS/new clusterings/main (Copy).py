import streamlit as st
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image   

st.title("Visualizador de Resultados de Clusterização")

base_dir = "results_hierarchical_40%"
series = sorted(os.listdir(base_dir))

selected_series = st.selectbox("Selecione a Série:", series)

methods = sorted(os.listdir(os.path.join(base_dir, selected_series)))
selected_method = st.selectbox("Selecione o Método:", methods)

output_dir = os.path.join(base_dir, selected_series, selected_method)

st.subheader(f"Série: {selected_series} | Método: {selected_method}")


serie_txt_path = os.path.join('./MixedBag', f'{selected_series}.txt')

if os.path.exists(serie_txt_path):
    with open(serie_txt_path, 'r') as f:
        line = f.readline()
        serie = np.array([float(val) for val in line.strip().split(',')])
    st.success(f"Série original carregada de: {serie_txt_path}")
    st.write(f"Primeiros valores da série: {serie[:10]}")
else:
    st.warning("Série original não encontrada no formato esperado (.txt em MixedBag).")
    serie = None


# Mostrar imagens
for img_name in ['regime_bar.png', 'dendrograma.png', 'silhouette.png']:
    img_path = os.path.join(output_dir, img_name)
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption=img_name, use_container_width=True)

# Mostrar métricas
metrics_path = os.path.join(output_dir, 'metrics.json')
if os.path.exists(metrics_path):
    st.subheader("Métricas")
    with open(metrics_path) as f:
        metrics = json.load(f)
    st.json(metrics)

# Carregar snippets
snippets_json = os.path.join(output_dir, 'snippets.json')
snippets = None
if os.path.exists(snippets_json):
    with open(snippets_json) as f:
        snippets = json.load(f)
elif os.path.exists(os.path.join(output_dir, 'snippets.npy')):
    snippets = np.load(os.path.join(output_dir, 'snippets.npy'), allow_pickle=True)
    # Convertendo para lista padrão
    snippets = [{'index': int(s[0]), 'subsequence': s[1].tolist()} for s in snippets]

if snippets:
    st.subheader("Visualização dos Snippets")
    
    max_snippets = len(snippets)
    num_snippets = st.slider("Número de snippets a visualizar:", min_value=1, max_value=max_snippets, value=min(5, max_snippets))

    selected_snippets = snippets[:num_snippets]

    # Gráfico 1: shapes dos snippets
    fig1, ax1 = plt.subplots(figsize=(8, 4))

    color_map = cm.get_cmap('tab10', num_snippets)

    for i, snip in enumerate(selected_snippets):
        ax1.plot(snip['subsequence'], color=color_map(i), label=f'Snippet {i+1}', alpha=0.8)

    ax1.set_title('Shapes dos Snippets')
    ax1.legend()
    st.pyplot(fig1)

    # Gráfico 2: snippets na série temporal
    if serie is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(serie, label='Série Original', color='gray')

        subseq_size = len(snippets[0]['subsequence'])

        for i, snip in enumerate(selected_snippets):
            idx = snip['index']
            ax2.plot(range(idx, idx + subseq_size), serie[idx:idx + subseq_size], 
                     linewidth=2, color=color_map(i), label=f'Snippet {i+1}')

        ax2.set_title('Snippets na Série Temporal')
        ax2.legend()
        st.pyplot(fig2)
else:
    st.warning("Snippets não encontrados.")
