import streamlit as st
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
import sqlite3
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configuración de la página
st.set_page_config(
    page_title="Cumplimiento POA Tec.Azuay",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
  .main { background: #f9f9fb; padding: 20px 40px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
  .sidebar .sidebar-content { background-color: #003366; color: white; padding: 20px; }
  h1, h2, h3 { color: #003366; font-weight: 700; }
  .styled-header { background-color: #003366; color: white; padding: 10px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 1em 0; }
  hr { border: none; border-top: 2px solid #003366; margin: 2em 0; }
  .streamlit-expanderHeader { font-weight: 600; font-size: 1.1em; }
  div[style*="text-align: center;"] img { border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
</style>
""", unsafe_allow_html=True)

# Función para cargar y codificar imagen
def load_and_encode(path, width=400):
    try:
        img = Image.open(path)
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        return f"<div style='text-align:center;'><img src='data:image/png;base64,{data}' width='{width}'/></div>"
    except FileNotFoundError:
        return None


# ================================
# Función para inicializar la base de datos
# ================================

def init_db():
    conn = sqlite3.connect("Seguimiento_poa.db")
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS Tipo_Unidad (
        ID_Tipo_Unidad INTEGER PRIMARY KEY AUTOINCREMENT,
        Nombre_Tipo TEXT NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS Persona (
        ID_Persona INTEGER PRIMARY KEY AUTOINCREMENT,
        Cedula TEXT NOT NULL UNIQUE,
        Nombre TEXT NOT NULL,
        Apellidos TEXT NOT NULL,
        Celular TEXT,
        Correo TEXT
    );

    CREATE TABLE IF NOT EXISTS Unidades (
        ID_Unidad INTEGER PRIMARY KEY AUTOINCREMENT,
        Nombre_Unidad TEXT,
        Descripcion TEXT,
        Informe_Semestral TEXT,
        Matriz_Semestral TEXT,
        Evidencias_Semestral TEXT,
        Informe_Final TEXT,
        Matriz_Final TEXT,
        Evidencias_Final TEXT,
        Porcentaje INTEGER,
        ID_Tipo_Unidad INTEGER NOT NULL,
        FOREIGN KEY (ID_Tipo_Unidad) REFERENCES Tipo_Unidad(ID_Tipo_Unidad)
    );

    CREATE TABLE IF NOT EXISTS Unidad_Responsable (
        ID_Unidad INTEGER NOT NULL,
        ID_Persona INTEGER NOT NULL,
        PRIMARY KEY (ID_Unidad, ID_Persona),
        FOREIGN KEY (ID_Unidad) REFERENCES Unidades(ID_Unidad),
        FOREIGN KEY (ID_Persona) REFERENCES Persona(ID_Persona)
    );

    CREATE TABLE IF NOT EXISTS POA (
        ID_POA INTEGER PRIMARY KEY AUTOINCREMENT,
        Entregable TEXT,
        Observaciones TEXT,
        Porcentaje DECIMAL(5,2),
        Anio INTEGER,
        ID_Unidad INTEGER NOT NULL,
        FOREIGN KEY (ID_Unidad) REFERENCES Unidades(ID_Unidad)
    );

    CREATE TABLE IF NOT EXISTS Actividad (
        ID_Actividad INTEGER PRIMARY KEY AUTOINCREMENT,
        Objetivo_Operativo TEXT,
        Fecha_Entregable DATE,
        Tarea TEXT,
        Area_Responsable TEXT,
        Entregable TEXT,
        Ejecutado DECIMAL(5,2),
        Costos_Fijos DECIMAL(10,2),
        Observacion TEXT,
        ID_Unidad INTEGER NOT NULL,
        FOREIGN KEY (ID_Unidad) REFERENCES Unidades(ID_Unidad)
    );
    """)
    
    conn.commit()
    conn.close()
    print("✅ Estructura de la base de datos verificada y tablas creadas si no existían.")

    
init_db()

# Mostrar cabecera
st.markdown('<h1 style="text-align:center;">📈 Cumplimiento del POA en el Tec.Azuay</h1>', unsafe_allow_html=True)
st.markdown("---")

# Mostrar logo
html_logo = load_and_encode("Imagen/logo.png", 450)
if html_logo:
    st.markdown(html_logo, unsafe_allow_html=True)
else:
    st.warning("Imagen del Tec.Azuay no encontrada.")
st.markdown("---")



# ================================
# 📑 MENÚ LATERAL
# ================================
opcion = st.sidebar.selectbox(
    "📌 Navegación",
    ["🏠 Inicio", "📂 Carga de Datos", "⚙️ Pre procesamiento", "🤖 Modelado", "🧪 Pruebas", "📊 Consultas", "📈 Gráfica", "👥 Información del Grupo"]
)


# Variables inicializadas para evitar errores
unidad_sel = None
tipo_vinculacion = None


# ================================
# 🧭 SECCIÓN: INICIO
# ================================
if opcion == "🏠 Inicio":
    st.markdown("## 👋 Bienvenido al Cumplimiento del POA en el Tec.Azuay")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # Columna izquierda
    with col1:
        with st.expander("❓ ¿Qué es el POA?"):
            st.markdown("""
            El **Plan Operativo Anual (POA)** es un instrumento de planificación que detalla las acciones específicas que una organización, ya sea pública o privada, 
            se propone ejecutar durante un año fiscal, con el objetivo de contribuir al cumplimiento de metas estratégicas.

            El POA incluye actividades, responsables, cronogramas, indicadores y recursos necesarios para llevar a cabo dichas acciones. 
            Permite alinear el trabajo diario con los objetivos institucionales de mediano y largo plazo, facilitando así el seguimiento, evaluación y rendición de cuentas.
            """)

    # Columna derecha
    with col2:
        with st.expander("📊 ¿De qué trata el proyecto?"):
            st.markdown("""
            El proyecto consiste en el análisis del cumplimiento del **Plan Operativo Anual (POA)** durante el año **2024**, 
            y en realizar una predicción sobre el posible comportamiento del cumplimiento en el año **2025**.
            """)

    st.markdown("---")

    st.markdown('<div class="styled-header">📄 Objetivo del Proyecto</div>', unsafe_allow_html=True)
    st.write("""
        El objetivo es analizar el cumplimiento del POA en el Instituto Tec. Azuay y predecir su evolución futura,
        utilizando técnicas de análisis de datos y modelado predictivo.
        Se busca identificar áreas de mejora y proponer acciones concretas para optimizar la planificación institucional anual.
    """)

    st.markdown("---")
    st.markdown('<div class="styled-header">🔍 Problema y Justificación</div>', unsafe_allow_html=True)
    st.write("""
        El problema radica en la falta de seguimiento sistemático del cumplimiento del POA, lo cual puede ocasionar desviaciones 
        en la ejecución de actividades institucionales. Este proyecto busca justificar la necesidad de contar con herramientas 
        analíticas que permitan evaluar el avance real de los objetivos planteados y tomar decisiones basadas en datos.
    """)

    st.markdown("---")
    st.markdown('<div class="styled-header">🧠 Tecnologías Utilizadas</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Análisis Descriptivo:** Resumen de las características básicas del cumplimiento registrado por unidad y periodo.

    - **Análisis Exploratorio de Datos (EDA):** Exploración y visualización para identificar patrones, retrasos o problemas en el cumplimiento del POA.

    - **Técnicas de Visualización:** Gráficos, diagramas de barras, pastel y líneas de tiempo para mostrar el desempeño por unidad y categoría.

    - **Análisis Estadístico:** Evaluación de métricas de cumplimiento e identificación de unidades críticas.

    - **Modelado Predictivo:** Algoritmos de *machine learning* (como Naive Bayes) para predecir el nivel de cumplimiento en 2025 con base en datos anteriores.
    """)


# ================================
# 📂 SECCIÓN: CARGA DE DATOS
# ================================
elif opcion == "📂 Carga de Datos":
    st.markdown("## 🗂️ Carga de Datos")
    st.markdown("📂 **En esta sección se visualizan los datos originales correspondientes al seguimiento de informes de las unidades administrativas.**")
    st.markdown("""
    Los datos provienen del archivo `Data.csv` y reflejan el estado de cumplimiento de entregables como informes semestrales, matrices, evidencias y porcentajes de avance final.  
    Esta visualización permite identificar rápidamente el progreso individual de cada responsable, así como detectar entregas incompletas, pendientes o con observaciones.
    """)

    try:
        # Cargar el archivo CSV llamado Data.csv
        df = pd.read_csv("Data.csv", encoding="utf-8-sig")

        st.success("✅ Data.csv cargado correctamente.")
        st.write("### Vista previa de los datos:")
        st.dataframe(df.head(30))

    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'Data.csv'. Asegúrate de que esté en la misma carpeta que tu archivo `.py`.")


# ================================
# ⚙️ SECCIÓN: PRE PROCESAMIENTO
# ================================
elif opcion == "⚙️ Pre procesamiento":
    st.markdown("## ⚙️ Pre procesamiento")
    st.markdown("🧹 **Preprocesamiento de Datos**")
    st.markdown("""
    En esta sección se realiza el preprocesamiento del conjunto de datos para facilitar el análisis y el entrenamiento del modelo de predicción.

    La principal transformación aplicada consiste en la creación de una variable categórica binaria que representa el cumplimiento final de cada unidad:
    - Se asigna el valor **1** si el registro está **COMPLETADO**.
    - Se asigna el valor **0** si el registro está **INCOMPLETO**.

    Esta codificación permite que el modelo aprenda a diferenciar entre unidades que cumplieron completamente con los informes y aquellas que no lo hicieron.
    """)
    
    df_pre = pd.read_csv("data_preprocesada.csv")
    st.success("✅ Data.csv cargado correctamente.")

    # Mostrar algunos datos
    st.write("### Vista previa de los datos preprocesados")
    st.dataframe(df_pre.head(30))


# ================================
# 🤖 SECCIÓN: MODELADO
# ================================
elif opcion == "🤖 Modelado":
    st.markdown("🤖 **Modelado de Cumplimiento**")
    st.markdown("""
    En esta sección se entrena un modelo de clasificación supervisada para predecir si una unidad administrativa ha **cumplido completamente con la entrega de la evidencia final** (`1`) o no (`0`).

    Se utilizó el algoritmo **Random Forest Classifier** por las siguientes razones:

    - 🔍 **Precisión en clasificación binaria**: es eficaz para predecir categorías como 'Completado' e 'Incompleto'.
    - 🌲 **Robustez y rendimiento**: Random Forest combina múltiples árboles de decisión, lo que mejora el rendimiento general y reduce el riesgo de sobreajuste.
    - 📊 **Interpretabilidad moderada**: permite analizar la importancia de cada variable en la predicción final.

    El modelo fue entrenado con el 80% de los datos y evaluado con el 20% restante, mostrando la **precisión (accuracy)** y un **informe de clasificación**.
    """)
    st.markdown("---")
    
    # Cargar datos preprocesados
    df = pd.read_csv("data_preprocesada.csv")

    # Separar X e y
    X = df.drop("Evidencia Final", axis=1)
    y = df["Evidencia Final"]

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Crear y entrenar modelo
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Predecir y evaluar
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"Accuracy: {acc:.2f}")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    # Guardar modelo entrenado localmente
    joblib.dump(clf, "modelo_rf.pkl")
    st.success("Modelo entrenado y guardado como modelo_rf.pkl")

# ================================
# 🧪 SECCIÓN: PRUEBAS
# ================================
elif opcion == "🧪 Pruebas":
    st.write("### Prueba de predicción con el modelo entrenado")
    st.markdown("🧪 **Pruebas del Modelo**")
    st.markdown("""
    En esta sección puedes **probar el modelo de predicción de cumplimiento** entrenado previamente en la sección 🤖 *Modelado*.
    El modelo permite simular si una unidad administrativa cumplirá con la entrega de la **Evidencia Final** (`1`) o no (`0`), en función de sus características y datos asociados.

    🔍 **¿Qué puedes hacer aquí?**
    - Seleccionar una unidad por su nombre.
    - Ver la predicción generada por el modelo para esa unidad.
    - Obtener la probabilidad estimada de cumplimiento.
    """)
    try:
        clf = joblib.load("modelo_rf.pkl")
        st.success("✅ Modelo cargado correctamente.")

        # Cargar los datos preprocesados
        df = pd.read_csv("data_preprocesada.csv")

        # Detectar columnas que contienen unidades (dummies)
        columnas_unidades = [col for col in df.columns if col.startswith("Nombre de la Unidad_")]
        unidades = [col.replace("Nombre de la Unidad_", "") for col in columnas_unidades]

        # Separar X y y
        X = df.drop(columns=["Evidencia Final"])

        # Obtener mapeo de unidad a filas
        opciones = {}
        for nombre in unidades:
            col_dummy = f"Nombre de la Unidad_{nombre}"
            filas_con_unidad = X[X[col_dummy] == 1]
            if not filas_con_unidad.empty:
                opciones[nombre] = filas_con_unidad

        unidad_seleccionada = st.selectbox("Selecciona la carrera o unidad:", list(opciones.keys()))

        if st.button("🔍 Predecir"):
            fila = opciones[unidad_seleccionada].iloc[[0]]  # Se usa la primera fila encontrada
            pred = clf.predict(fila)[0]
            prob = clf.predict_proba(fila)[0]
            porcentaje_clase = round(prob[pred] * 100, 2)

            st.info(f"""
            🏢 **Unidad:** {unidad_seleccionada}  
            🔮 **Probabilidad de Cumplimiento:** {porcentaje_clase}%  
            📌 **Predicción (Clase):** {pred}
            """)

    except FileNotFoundError:
        st.error("❌ No se encontró el modelo entrenado. Asegúrate de entrenarlo en la sección '🤖 Modelado'.")

# ================================
# 📈 SECCIÓN: GRÁFICAS
# ================================

elif opcion == "📈 Gráfica":
    tipo_vista = st.sidebar.selectbox("🗂️ Selecciona el tipo de gráfica que deseas ver:",
                                       options=["Carreras", "Unidad Académica", "Unidad Administrativa"])

    datos_seleccionados = {}
    porcentaje_ejecutado = 0
    
    # ==========================
    # 1️⃣ VISTA POR CARRERAS
    # ==========================
    if tipo_vista == "Carreras":
        st.sidebar.header("📌 Filtro de Visualización")
        datos = {
            "Todas las carreras": {"TOTAL": 608, "VERDE": 508, "ROJO": 88, "AMARILLO": 12, "NARANJA": 0, "EJECUTADO": 84.54},
            "Entrenamiento Deportivo": {"TOTAL": 38, "VERDE": 33, "ROJO": 4, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 88.16},
            "Producción Audiovisual": {"TOTAL": 38, "VERDE": 30, "ROJO": 8, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 78.95},
            "Penitenciaria": {"TOTAL": 38, "VERDE": 26, "ROJO": 12, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 68.42},
            "Plataformas tecnológicas": {"TOTAL": 38, "VERDE": 33, "ROJO": 5, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 86.84},
            "Asesoría Financiera": {"TOTAL": 38, "VERDE": 23, "ROJO": 15, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 60.53},
            "Big Data": {"TOTAL": 38, "VERDE": 37, "ROJO": 0, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 98.68},
            "Ciberseguridad": {"TOTAL": 38, "VERDE": 37, "ROJO": 0, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 98.68},
            "Desarrollo de Software": {"TOTAL": 38, "VERDE": 37, "ROJO": 0, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 98.68},
            "Desarrollo infantil": {"TOTAL": 38, "VERDE": 37, "ROJO": 0, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 98.68},
            "Patrimonio": {"TOTAL": 38, "VERDE": 32, "ROJO": 5, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 85.53},
            "Mantenimiento eléctrico": {"TOTAL": 38, "VERDE": 31, "ROJO": 6, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 82.89},
            "Mecatrónica": {"TOTAL": 38, "VERDE": 31, "ROJO": 7, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 81.58},
            "Metalmecánica": {"TOTAL": 38, "VERDE": 27, "ROJO": 10, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 72.37},
            "Madera": {"TOTAL": 38, "VERDE": 27, "ROJO": 10, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 72.37},
            "Seguridad y Prevención de Riesgos Laborales": {"TOTAL": 38, "VERDE": 36, "ROJO": 1, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 96.05},
            "Tributación": {"TOTAL": 38, "VERDE": 31, "ROJO": 5, "AMARILLO": 2, "NARANJA": 0, "EJECUTADO": 84.21},
        }
        opciones = list(datos.keys())
        seleccion = st.sidebar.selectbox("🎓 Selecciona una carrera:", opciones)
        datos_fijos = datos[seleccion]
        porcentaje_ejecutado = datos_fijos["EJECUTADO"]

    # ==========================
    # 2️⃣ VISTA POR UNIDAD ACADÉMICA
    # ==========================
    elif tipo_vista == "Unidad Académica":
        st.sidebar.header("📌 Filtro de Visualización")
        datos = {
            "Todas las Unidades Académicas": {"TOTAL": 83, "VERDE": 66, "ROJO": 11, "AMARILLO": 6, "NARANJA": 0, "EJECUTADO": 84.23},
            "Centro de Formación Integral y Servicios Especializados": {"TOTAL": 5, "VERDE": 5, "ROJO": 0, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 100.00},
            "Centro de Idiomas": {"TOTAL": 10, "VERDE": 9, "ROJO": 1, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 90.00},
            "Coordinación de carreras Tecnológicas": {"TOTAL": 10, "VERDE": 8, "ROJO": 0, "AMARILLO": 2, "NARANJA": 0, "EJECUTADO": 96.00},
            "Coordinación de carreras Universitarias": {"TOTAL": 7, "VERDE": 6, "ROJO": 0, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 97.00},
            "Coordinación de Investigación, Desarrollo Tecnológico e Innovación": {"TOTAL": 17, "VERDE": 13, "ROJO": 2, "AMARILLO": 2, "NARANJA": 0, "EJECUTADO": 86.00},
            "Coordinación de Posgrados": {"TOTAL": 5, "VERDE": 2, "ROJO": 3, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 40.00},
            "Coordinación de Vinculación con la Sociedad": {"TOTAL": 29, "VERDE": 23, "ROJO": 5, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 81.00},
        }
        opciones = list(datos.keys())
        seleccion = st.sidebar.selectbox("🏛️ Selecciona una Unidad Académica:", opciones)
        datos_fijos = datos[seleccion]
        porcentaje_ejecutado = datos_fijos["EJECUTADO"]

    # ==========================
    # 3️⃣ UNIDAD ADMINISTRATIVA
    # ==========================
    elif tipo_vista == "Unidad Administrativa":
        st.sidebar.header("📌 Filtro de Visualización")
        datos = {
            "Todas las Unidades Administrativas": {"TOTAL": 292, "VERDE": 212, "ROJO": 57, "AMARILLO": 23, "NARANJA": 0, "EJECUTADO": 80.00},
            "Aseguramiento de Aseguramiento de la Calidad": {"TOTAL": 24, "VERDE": 19, "ROJO": 3, "AMARILLO": 2, "NARANJA": 0, "EJECUTADO": 86.09},
            "Coordinación de Bienestar Institucional": {"TOTAL": 39, "VERDE": 9, "ROJO": 18, "AMARILLO": 12, "NARANJA": 0, "EJECUTADO": 36.84},
            "Coordinación de Estratégia": {"TOTAL": 7, "VERDE": 7, "ROJO": 0, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 100.00},
            "Dirección Administrativa Financiera": {"TOTAL": 7, "VERDE": 6, "ROJO": 1, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 85.71},
            "Procuraduría": {"TOTAL": 11, "VERDE": 9, "ROJO": 2, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 81.82},
            "Secretaria General": {"TOTAL": 12, "VERDE": 6, "ROJO": 4, "AMARILLO": 2, "NARANJA": 0, "EJECUTADO": 60.83},
            "Unidad de  Servicios de Biblioteca": {"TOTAL": 25, "VERDE": 20, "ROJO": 0, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 80.00},
            "Unidad de Comunicación": {"TOTAL": 23, "VERDE": 22, "ROJO": 0, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 99.13},
            "Unidad de Mantenimiento E Infraestructura": {"TOTAL": 30, "VERDE": 25, "ROJO": 1, "AMARILLO": 4, "NARANJA": 0, "EJECUTADO": 93.10},
            "Unidad de Planificación y Gestión de Calidad": {"TOTAL": 14, "VERDE": 11, "ROJO": 2, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 82.14},
            "Unidad de Relaciones Internacionales e Institucionales": {"TOTAL": 18, "VERDE": 3, "ROJO": 15, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 17.65},
            "Unidad de Seguridad y Salud Ocupacional": {"TOTAL": 16, "VERDE": 11, "ROJO": 5, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 67.50},
            "Unidad de Talento Humano": {"TOTAL": 18, "VERDE": 17, "ROJO": 6, "AMARILLO": 0, "NARANJA": 0, "EJECUTADO": 94.44},
            "Unidad de Tecnologia de la Información y Comunicación": {"TOTAL": 48, "VERDE": 47, "ROJO": 0, "AMARILLO": 1, "NARANJA": 0, "EJECUTADO": 99.58}
        }
        opciones = list(datos.keys())
        seleccion = st.sidebar.selectbox("🏢 Selecciona una Unidad Administrativa:", opciones)
        datos_fijos = datos[seleccion]
        porcentaje_ejecutado = datos_fijos["EJECUTADO"]

    # ==========================
    # VISUALIZACIÓN COMÚN
    # ==========================
    st.markdown("""
    <style>
        .titulo {
            font-size: 28px;
            font-weight: 700;
            color: #2C3E50;
            margin-bottom: 15px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .tarjeta {
            background-color: #ffffff;
            padding: 25px 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .leyenda-div div {
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            font-weight: 600;
            font-size: 16px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .verde {
            background-color: #28a745; /* Verde */
            color: white;
        }
        .amarillo {
            background-color: #ffc107; /* Amarillo */
            color: black;
            border: 1px solid #d39e00;
        }
        .naranja {
            background-color: #fd7e14; /* Naranja */
            color: white;
        }
        .rojo {
            background-color: #dc3545; /* Rojo */
            color: white;
        }
        .total {
            background-color: #17a2b8; /* Azul celeste */
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    # Solo una columna para mostrar el % Ejecutado más pequeño
    st.markdown("<div class='tarjeta'>", unsafe_allow_html=True)
    st.markdown("<div class='titulo'>🎯 % Ejecutado</div>", unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(porcentaje_ejecutado, 2),
        number={'suffix': "%", 'font': {'size': 38, 'family': 'Segoe UI'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'size': 12}},
            'bar': {'color': "#28a745"},
            'steps': [
                {'range': [0, 25], 'color': "#dc3545"},
                {'range': [25, 50], 'color': "#fd7e14"},
                {'range': [50, 90], 'color': "#ffc107"},
                {'range': [90, 100], 'color': "#28a745"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': round(porcentaje_ejecutado, 2)
            }
        }
    ))

    fig_gauge.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        font=dict(family="Segoe UI"),
        height=250,  # Tamaño más pequeño
        width=250,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig_gauge, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='tarjeta'>", unsafe_allow_html=True)

    st.markdown("<div class='titulo'>📊 Cumplimiento </div>", unsafe_allow_html=True)

    total = datos_fijos["TOTAL"]
    if total > 0:
        porc_verde = datos_fijos["VERDE"] / total * 100
        porc_rojo = datos_fijos["ROJO"] / total * 100
        porc_amarillo = datos_fijos["AMARILLO"] / total * 100
        porc_naranja = datos_fijos["NARANJA"] / total * 100
    else:
        porc_verde = porc_rojo = porc_amarillo = porc_naranja = 0

    fig_bar = px.bar(
        x=[porc_verde, porc_rojo, porc_amarillo, porc_naranja],
        y=["VERDE", "ROJO", "AMARILLO", "NARANJA"],
        orientation='h',
        color=["VERDE", "ROJO", "AMARILLO", "NARANJA"],
        color_discrete_map={
            "VERDE": "#28a745",
            "ROJO": "#dc3545",
            "AMARILLO": "#ffc107",
            "NARANJA": "#fd7e14"
        },
        text=[f"{porc_verde:.2f}%", f"{porc_rojo:.2f}%", f"{porc_amarillo:.2f}%", f"{porc_naranja:.2f}%"],
        height=500
    )

    # Aplicar estilo sin borde ni sombra
    fig_bar.update_traces(
        textposition='outside',
        marker=dict(line=dict(width=0))  # Esto elimina los bordes oscuros
    )


    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title='Porcentaje (%)',
        yaxis_title='Nivel de Cumplimiento',
        font=dict(family="Segoe UI", size=16),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100])
    )

    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ ¿Qué significan los porcentajes?")
    st.markdown("""
    Los **porcentajes** representan el **grado de avance o cumplimiento** que ha alcanzado una unidad con respecto a los objetivos establecidos en el POA (Plan Operativo Anual).

    #### 📌 Clasificación por colores:
    - 🟢 **Verde (≥ 90%)**: Indica que las unidades o actividades han alcanzado un **alto nivel de cumplimiento**. Refleja una gestión eficiente y un seguimiento adecuado del POA.
    - 🟡 **Amarillo (75% - 89%)**: Representa un **cumplimiento medio**. Aunque las metas están parcialmente alcanzadas, aún se requiere seguimiento para garantizar su ejecución total.
    - 🟠 **Naranja (50% - 74%)**: Muestra un **nivel de avance bajo-moderado**. Estas actividades necesitan atención, ya que su progreso es limitado y puede afectar el cumplimiento global.
    - 🔴 **Rojo (< 50%)**: Señala un **cumplimiento crítico o deficiente**. Las unidades en esta categoría deben ser priorizadas, ya que presentan un riesgo significativo de no alcanzar sus objetivos del POA.

    Estos colores permiten identificar rápidamente el estado de cada unidad y facilitar la toma de decisiones para el seguimiento y mejora del POA.
        """)



# ================================
# 📊 SECCIÓN: INGRESO DE DATOS
# ================================

elif opcion == "📊 Consultas":
    st.markdown("## 📊 Seguimiento POA")
    st.info("""
    **📌 En esta sección podrás consultar el avance del POA (Plan Operativo Anual) de las distintas unidades institucionales.**

    Para comenzar, selecciona el **Tipo de Unidad** (por ejemplo: Carrera, Departamento, Dirección, etc.) y luego elige una unidad específica dentro de esa categoría.

    Una vez seleccionada la unidad, se mostrará la siguiente información:

    - ✅ El **nombre** de la unidad seleccionada.
    - 📈 El **porcentaje registrado**, que indica el grado de cumplimiento de las actividades planificadas.
    - 📝 La **descripción de la unidad**, que explica brevemente su función.
    - 👤 Las **personas responsables**, junto con su contacto.
    """)


    # Conexión a la base de datos
    conn = sqlite3.connect("Seguimiento_poa.db")
    cur = conn.cursor()

    # Opciones visibles para el usuario
    tipo_opciones = ["", "Carrera", "Unidad Administrativa", "Unidad Académica", "Vinculación"]
    c1, = st.columns(1)
    with c1:
        unidad_sel = st.selectbox("🏢 Tipo de Unidad:", tipo_opciones, index=0)

        # Mapeo personalizado de tipos
        if unidad_sel == "Carrera":
            ids_tipo = (1, 2, 3, 4, 5, 6)
        elif unidad_sel == "Unidad Administrativa":
            ids_tipo = (7,)
        elif unidad_sel == "Unidad Académica":
            ids_tipo = (8,)
        elif unidad_sel == "Vinculación":
            ids_tipo = (9,)
        else:
            ids_tipo = ()

        if ids_tipo:
            query = f"""
                SELECT U.Nombre_Unidad, U.ID_Unidad, U.Descripcion, U.Porcentaje
                FROM Unidades U
                WHERE U.ID_Tipo_Unidad IN ({','.join(['?'] * len(ids_tipo))})
            """
            cur.execute(query, ids_tipo)
            resultados = cur.fetchall()

            if resultados:
                nombres_unidades = [r[0] for r in resultados]
                id_unidades = [r[1] for r in resultados]
                descripciones_unidades = [r[2] if r[2] else "Sin descripción" for r in resultados]
                porcentajes_unidades = [r[3] if r[3] is not None else 0 for r in resultados]

                # Mostrar selector de unidad específica debajo
                seleccion = st.selectbox("📚 Selecciona la unidad específica:", [""] + nombres_unidades)

                if seleccion:
                    st.success(f"✅ Has seleccionado: {seleccion}")
                    id_seleccion = id_unidades[nombres_unidades.index(seleccion)]
                    descripcion_sel = descripciones_unidades[nombres_unidades.index(seleccion)]
                    porcentaje_sel = porcentajes_unidades[nombres_unidades.index(seleccion)]

                    # ← AQUÍ agregas esta línea para mostrar el porcentaje
                    st.markdown(f"🔢 **Porcentaje registrado:** {porcentaje_sel}%")

                    st.markdown(f"📝 **Descripción de la unidad:** {descripcion_sel}")
                    # ... el resto de tu código para responsables y actividades

                    # Consulta para obtener personas responsables
                    cur.execute("""
                        SELECT P.Nombre, P.Apellidos, P.Cedula, P.Celular, P.Correo
                        FROM Persona P
                        JOIN Unidad_Responsable UR ON P.ID_Persona = UR.ID_Persona
                        WHERE UR.ID_Unidad = ?
                    """, (id_seleccion,))
                    personas = cur.fetchall()

                    if personas:
                        st.markdown("### 👥 Personas responsables:")
                        for p in personas:
                            nombre_completo = f"{p[0].strip()} {p[1].strip()}"
                            cedula = p[2].strip()
                            celular = p[3].strip() if p[3] else "No registrado"
                            correo = p[4].strip() if p[4] else "No registrado"
                            st.markdown(f"""
                - **{nombre_completo}**  
                Cédula: {cedula}  
                Celular: {celular}  
                Correo: {correo}
                            """)
                    else:
                        st.info("ℹ️ No hay personas responsables registradas para esta unidad.")

                    # Consulta de actividades asociadas
                    cur.execute("""
                        SELECT Tarea, Fecha_Entregable, Entregable, Ejecutado, Costos_Fijos, Observacion
                        FROM Actividad
                        WHERE ID_Unidad = ?
                    """, (id_seleccion,))
                    actividades = cur.fetchall()

                    # Mostrar tabla de actividades o mensaje personalizado
                    if actividades:
                        st.markdown("### 📋 Actividades registradas:")
                        actividades_df = pd.DataFrame(actividades, columns=["Tarea", "Fecha Entregable", "Entregable", "Porcentaje", "Costo Fijo", "Observación"])
                        st.dataframe(actividades_df, use_container_width=True)

                    else:
                        if unidad_sel in ["Carrera", "Vinculación"]:
                            st.info("ℹ️ Esta Unidad de carrera o vinculación no tiene actividades registradas en el POA.")
                        else:
                            st.warning("📭 No hay actividades registradas para esta unidad aún.")
                            st.markdown("👉 Por favor verifica con los responsables si hay actividades pendientes por ingresar.")


    conn.close()


elif opcion == "👥 Información del Grupo":
    st.markdown("## 👥 Información del Grupo")

    miembros = [
    {
        "nombre": "Allison Bueno",
        "imagen": "Imagen/imagen1.png",
        "correo": "allison.bueno.est@tecazuay.edu.ec",
        "aporte": "Diseño y desarrollo completo de la interfaz web en Streamlit: estructura de navegación, estilos CSS, carga y visualización de datos, componentes interactivos y manejo de imágenes."
    },
    {
        "nombre": "Byron Mendieta",
        "imagen": "Imagen/imagen2.png",
        "correo": "byron.mendieta.est@tecazuay.edu.ec",
        "aporte": "Elaboración de dashboards y gráficas avanzadas (Plotly, Altair), creación de indicadores de cumplimiento, generación de informes semestrales y validación de resultados de modelado predictivo."
    }
]


    for m in miembros:
        # Leer y codificar imagen para mostrar circular con CSS
        img_html = load_and_encode(m['imagen'], 150) if load_and_encode else None

        tarjeta = f"""
        <div style="
            display: flex;
            align-items: center;
            background: #f0f2f6;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <div style="flex-shrink: 0; width: 120px; height: 120px; margin-right: 20px;">
                {img_html if img_html else '<div style="width:120px;height:120px;background:#ccc;border-radius:50%;"></div>'}
            </div>
            <div>
                <h3 style="margin:0; color: #333;">{m['nombre']}</h3>
                <p style="margin: 5px 0;">
                    📧 <a href="mailto:{m['correo']}" style="text-decoration:none; color:#007acc;">{m['correo']}</a><br>
                    🛠️ <strong>Aporte:</strong> {m['aporte']}
                </p>
            </div>
        </div>
        """
        st.markdown(tarjeta, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray; font-size:0.9em;'>© 2025 Grupo 2 - Tec.Azuay</p>", unsafe_allow_html=True)