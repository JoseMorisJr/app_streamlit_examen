# Importar las bibliotecas necesarias
import streamlit as st
from PIL import Image           
from utils import *


def main_page():
    # Crear un título para la página principal
    st.set_page_config(layout='wide')

    st.title("Página Web de reconocimiento de imágenes DMC" )


    # Divide la ventana en dos columnas
    col1, col2 = st.columns(2)

    # Contenido de la primera columna
    with col1:
        st.header("Obtención de imágenes")
        st.write("Escriba en el recuadro la imagen que desea generar.")
        promt_image = str(st.text_input("Ingresa el nombre de la imagen a generar."))
        
        if st.button('Generar imagen'):
            if len(promt_image) == 0:
                st.write("¡Ingrese prompt de imagen!")
            else:
                generacion_imagen(promt_image)
                image_path_1 = "./image_temp/temporal.png" 
                # Abre la imagen usando PIL
                image_open = Image.open(image_path_1)
                st.session_state["generated_image"] = image_open

                # Muestra la imagen en Streamlit
                if "generated_image" in st.session_state:
                    st.image(st.session_state["generated_image"], caption=promt_image, use_column_width=True)

        # Crear un contenedor vacío para el botón centrado debajo de las dos columnas
        button_container = st.empty()


        # Colocar el botón en el centro debajo de las columnas
        with button_container.container():
            if st.button('Predecir imagen generada', type = 'primary', use_container_width = False):
                
                if len(promt_image) == 0:
                    st.write("¡Ingrese prompt de imagen!") 
                
                else:
                    try:

                        image_path_2 = "image_temp/temporal.png"

                        # Hacer la predicción
                        st.header("Etiqueta predecida:")
                        st.write(f"{predict_image(image_path_2)}")
                    except:
                        st.write("¡Ingrese prompt de imagen!")

    

    # Añade contenido a la segunda columna
    with col2:
        st.header("Predicción de imagen cargada")
        st.write("Ingrese en el recuadro la imagen que desea predecir su contenido.")
        file = st.file_uploader("Insertar imagen aquí.")
        try:
            st.image(file, caption='Imagen subida', use_column_width=True)
        except:
            pass 

        try: 
            with open("image_temp/imagen_descargada.png", "wb") as f:
                f.write(file.getbuffer())
        except:
            pass


        if st.button('Predecir imagen'):
            try:
                prediction = predict_image(file)
                st.header("Etiqueta predecida:")
                st.write(f"{prediction}")
            except:     
                st.write("¡Cargue una imagen!") 


# Punto de entrada principal de la aplicación Streamlit
if __name__ == "__main__":
    main_page()