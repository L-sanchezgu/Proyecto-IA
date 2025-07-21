import streamlit as st
from PIL import Image
import os
import style

# Configuración de la página
st.set_page_config(layout="centered")
st.markdown("<h1 style='text-align: center;'>🎨 Transferencia de Estilo de Imagen</h1>", unsafe_allow_html=True)

# Sidebar para controles
with st.sidebar:
    
    # Opción para subir imagen o usar ejemplo
    upload_option = st.radio(
        "Seleccionar imagen de contenido",
        ("Usar imagen de ejemplo", "Subir mi propia imagen")
    )
    
    if upload_option == "Subir mi propia imagen":
        uploaded_file = st.file_uploader(
            "Sube tu imagen (JPEG/PNG)", 
            type=["jpg", "jpeg", "png"]
        )
    else:
        uploaded_file = None
    
    style_name = st.selectbox(
        'Seleccionar estilo artístico',
        ('candy', 'mosaic', 'rain_princess', 'udnie')
    )

# Rutas de archivos
model = f"saved_models/{style_name}.pth"
style_image = f"images/style-images/{style_name}.jpg"

# Layout de dos columnas
col1, col2 = st.columns(2)

with col1:
    st.header("Imagen de Contenido")
    if upload_option == "Subir mi propia imagen" and uploaded_file is not None:
        # Mostrar imagen subida por el usuario
        content_img = Image.open(uploaded_file)
        st.image(content_img, width=300)
        input_image = uploaded_file
    else:
        # Mostrar imagen de ejemplo por defecto
        try:
            default_img = "content.jpg"
            content_img = Image.open(f"images/content-images/{default_img}")
            st.image(content_img, width=300)
            input_image = f"images/content-images/{default_img}"
        except FileNotFoundError:
            st.error("Imagen de ejemplo no encontrada")

with col2:
    st.header("Estilo de Referencia")
    try:
        style_img = Image.open(style_image)
        st.image(style_img, width=300)
    except FileNotFoundError:
        st.error(f"Imagen de estilo no encontrada: {style_image}")

# Botón centrado
st.write("")
clicked = st.button("✨ Aplicar Transformación Estilística", type="primary")

# Procesamiento y resultados
if clicked:
    try:
        with st.spinner(f"Aplicando estilo {style_name}..."):
            model = style.load_model(model)
            
            # Manejar tanto archivos subidos como rutas locales
            if isinstance(input_image, str):
                # Es una ruta de archivo local
                output_image = f"images/output-images/{style_name}-{os.path.basename(input_image)}"
                style.stylize(model, input_image, output_image)
            else:
                # Es un archivo subido por el usuario
                output_image = f"images/output-images/{style_name}-{uploaded_file.name}"
                # Guardar temporalmente la imagen subida
                with open(output_image, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Procesar la imagen
                style.stylize(model, output_image, output_image)
        
        st.success("¡Transformación completada!")
        
        # Mostrar resultado centrado
        st.write("")
        st.markdown("<h3 style='text-align: center;'>Resultado Estilizado</h3>", unsafe_allow_html=True)
        
        try:
            result_img = Image.open(output_image)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(result_img, width=400)
                
                # Botón de descarga
                with open(output_image, "rb") as file:
                    st.download_button(
                        label="⬇️ Descargar Imagen",
                        data=file,
                        file_name=f"estilo_{style_name}_{os.path.basename(output_image)}",
                        mime="image/png"
                    )
                    
        except FileNotFoundError:
            st.error("No se pudo generar la imagen de salida")
            
    except Exception as e:
        st.error(f"Error durante el procesamiento: {str(e)}")
        