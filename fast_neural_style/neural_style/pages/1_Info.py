import streamlit as st
from PIL import Image
import os

# Configuración de página
st.set_page_config(
    page_title="Galería de Estilos Artísticos",
    layout="wide",
    page_icon="🎨"
)

# CSS mejorado
st.markdown("""
<style>
    .main-header {
        background: rgba(38, 39, 48, 0.9);
        color: white;
        padding: 2rem;
        border-radius: 0;
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 4px solid #f63366;
    }
    .style-title {
        color: #f63366;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-align: center;
        width: 100%;
    }
    .style-subtitle {
        color: #a1a5b0;
        font-size: 1.1rem;
        margin-bottom: 0;
        text-align: center;
    }
    .style-img-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    .meta-info {
        color: #a1a5b0;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .description-text {
        text-align: justify;
        margin: 1rem 0;
        padding: 0 1rem;
        color: #f0f2f6;
    }
    .column-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }
    .style-grid {
        display: flex;
        flex-direction: column;
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado simplificado
st.markdown("""
<div class="main-header">
    <h1 class="style-title">🎨 Catálogo de Estilos</h1>
    <p class="style-subtitle">Transforma tus imágenes con transferencia de estilo neuronal</p>
</div>
""", unsafe_allow_html=True)

# Diccionario de estilos
ESTILOS = {
    "Candy": {
        "imagen": "images/style-images/candy.jpg",
        "descripcion": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec non massa sodales, faucibus lectus quis, semper lectus. Nam elementum tellus non tincidunt hendrerit. Donec at laoreet diam, id blandit dui. Phasellus semper dolor in nisl condimentum ultricies vestibulum eu sem. Ut ullamcorper leo mauris, ut rhoncus odio venenatis vitae. Morbi finibus lectus ex, ac scelerisque ante interdum eu. Etiam nec vestibulum mi, a posuere odio. Ut laoreet odio justo, eget dapibus tellus ultrices non.",
        "tecnica": "Redes Neuronales Convolucionales (VGG-19)",
        "año": "2016"
    },
    "Mosaic": {
        "imagen": "images/style-images/mosaic.jpg",
        "descripcion": "Morbi elementum elit eget erat tincidunt, in aliquam sem mollis. Nullam bibendum molestie facilisis. Cras felis diam, cursus ut venenatis at, fermentum at lacus. Sed quis ligula neque. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam porttitor, diam a aliquet dapibus, augue sem tempor ligula, vel vehicula turpis mauris sit amet lectus. Etiam condimentum placerat venenatis.",
        "tecnica": "Algoritmos de teselado adaptativo",
        "año": "Inspiración siglo VI"
    },
    "Rain Princess": {
        "imagen": "images/style-images/rain_princess.jpg",
        "descripcion": "Praesent laoreet, risus eu fermentum fringilla, turpis nibh feugiat risus, in congue erat turpis ac ante. Cras placerat molestie mauris, eget euismod libero tristique sit amet. Nunc sed dapibus metus. Nam ut ligula felis. Cras sit amet quam mi. Nam libero quam, fringilla at auctor vitae, commodo vel neque. Maecenas nec nisi arcu. Etiam imperdiet velit ut nisl dictum mollis.",
        "tecnica": "Transferencia de estilo neural (AdaIN)",
        "año": "2018"
    },
    "Udnie": {
        "imagen": "images/style-images/udnie.jpg",
        "descripcion": "Proin et enim at orci mattis molestie at non velit. In hac habitasse platea dictumst. Pellentesque posuere mollis libero, ac eleifend risus pellentesque quis. Vestibulum nunc felis, aliquam vel erat ac, pellentesque hendrerit turpis. Donec tincidunt nibh risus, vitae congue risus pulvinar at. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque nisl turpis, euismod tempor elementum id.",
        "tecnica": "Transformación geométrica profunda",
        "año": "Inspiración 1913"
    }
}

# Grid de estilos
st.markdown('<div class="style-grid">', unsafe_allow_html=True)

for i, (estilo, info) in enumerate(ESTILOS.items()):
    # Alternar la posición de la imagen
    if i % 2 == 0:
        col1, col2 = st.columns([1, 2])
    else:
        col2, col1 = st.columns([2, 1])
    
    with st.container():
        with col1:
            # Contenedor para centrar la imagen verticalmente
            st.markdown('<div class="style-img-container">', unsafe_allow_html=True)
            try:
                if os.path.exists(info["imagen"]):
                    img = Image.open(info["imagen"])
                    st.image(img, use_container_width=True)
                else:
                    st.warning(f"Imagen no encontrada: {info['imagen']}")
            except Exception as e:
                st.error(f"Error al cargar imagen: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Contenedor para el texto centrado verticalmente
            st.markdown('<div class="column-container">', unsafe_allow_html=True)
            st.markdown(f'<h3 class="style-title" style="text-align: center;">🎨 {estilo}</h3>', unsafe_allow_html=True)
            st.markdown(f'<p class="meta-info"><strong>📅 Año:</strong> {info["año"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="meta-info"><strong>🛠️ Técnica:</strong> {info["tecnica"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="description-text">{info["descripcion"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Pie de página
st.markdown("---")
st.markdown("### ¿Listo para transformar tus imágenes?")
if st.button("Ir a la Aplicación Principal", type="primary"):
    st.switch_page("main.py")