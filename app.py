import streamlit as st

import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Select which embeddings we want to use
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])


# Título WebApp
st.title('Chat con Documentos')

# Subtítulo
st.subheader('Elige el documento y escribe tu pregunta.')

# Definimos las opciones de las otras listas desplegables
vectorstore = ['Reportaje IA', 'Cap. 5 Ordenanza General de Urbanismo y Construcciones', 'NCh 433 Of.96 Mod.2009']

# Creamos las listas desplegables en la interfaz
documento_seleccionado = st.selectbox('Documento:', vectorstore)

# Cargando el vectorstore. Puede ser 'reportaje', 'OGUC_2016_removed' o 'nch_433_mod_test_2'. También se
# define el valor de 'k', que es el número de resultados que genera la búsqueda sobre los chunks y
# define el link a utilizar para mostrar el documento en la app.
if documento_seleccionado == 'Reportaje IA':
    doc = 'https:/github.com/HaroldConley/doc_chat_faiss/blob/main/reportaje_faiss/'
    k = 4
    link = "[Enlace al documento](https://drive.google.com/file/d/146f91rndeXFOpfY2IT9ybF5tHw6YFWyP/view?usp=share_link)"
elif documento_seleccionado == 'Cap. 5 Ordenanza General de Urbanismo y Construcciones':
    doc = 'https://github.com/HaroldConley/doc_chat_faiss/tree/main/OGUC_2016_faiss/'
    k = 2
    link = "[Enlace al documento](https://drive.google.com/file/d/1IORJZnoKxdF44FAGY5UE8Na4iVZxX0YK/view?usp=share_link)"
elif documento_seleccionado == 'NCh 433 Of.96 Mod.2009':
    doc = 'https://github.com/HaroldConley/doc_chat_faiss/tree/main/nch_433_faiss/'
    k = 4
    link = "[Enlace al documento](https://drive.google.com/file/d/1_htone_jV9mk-RYddheTis1a2_4KiKbS/view?usp=share_link)"

# Cargando el vectorstore. Puede ser 'reportaje', 'OGUC_2016_removed' o 'nch_433_mod_test_2'
db = FAISS.load_local(doc, embeddings)

# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

# create a chain to answer questions
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

# Muestra link al documento correspondiente
st.markdown(link)

# Ingreso de la pregunta.
query = st.text_area("Escribe tu pregunta aquí:")

# Generación de la respuesta.
# Agregue un botón "Responder" a la interfaz de usuario
if st.button('Responder'):
    with st.spinner('Leyendo el texto...'):
        # Procesamiento
        result = qa({"query": query})

        # Muestra de la respuesta
        st.markdown(result['result'], unsafe_allow_html=True)
