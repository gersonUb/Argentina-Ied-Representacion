import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inline_sql import sql, sql_val

# importamos las tablas de datos
direccion = "./TablasOriginales/"
df_sedes_completo = pd.read_csv(direccion + 'lista-sedes-datos.csv', on_bad_lines="warn")
# se agrega on_bad_lines, pues hay un error en la linea 16 del csv, con esto se logra ignorarla
df_sedes = pd.read_csv(direccion + 'lista-sedes.csv')
df_secciones = pd.read_csv(direccion + 'lista-secciones.csv')
archivo_inversiones = 'flujos-monetarios-netos-inversion-extranjera-directa.csv'
df_IED = pd.read_csv(direccion + archivo_inversiones, index_col="indice_tiempo").T # IED = Inversion Extranjera Directa
df_paises = pd.read_csv(direccion + 'paises.csv')

# corregimos el nombre de la columna para países
df_IED.index.name = 'pais'

# cambiamos el nombre de las columnas de IED para que solo sea el año
años = np.arange(1980,2023)
df_IED.columns = años

####################
    #ESQUEMAS
####################

#ESQUEMA SEDE
consulta_sede = """
                SELECT
                    sc.sede_id,
                    COUNT(
                        secc.tipo_seccion
                    ) AS n_secciones,
                    sc.pais_iso_3 AS iso3_sede
                FROM
                    df_sedes_completo sc
                    INNER JOIN df_sedes s ON sc.sede_id = s.sede_id
                    LEFT JOIN df_secciones secc ON sc.sede_id = secc.sede_id
                    GROUP BY sc.sede_id, s.sede_tipo, sc.pais_iso_3
                """
esquema_sede = sql^ consulta_sede
esquema_sede.set_index('sede_id', inplace=True)

#ESQUEMA RED SOCIAL

# es conveniente tener una lista de los links:
def separar_links(rrss):
    """
    Dada un string con links redes sociales separadas por //
    devuelve un array conteniendo estos links
    """
    if isinstance(rrss, str):
        # se obtiene la lista, el separador es //
        # se quita el ultimo elemento porque siempre es un string vacío
        # debido a como vienen los datos
        redes = rrss.split("  //  ")[:-1]
        return redes
    else:
        # si es algo distinto a un string, devuelve una lista vacía
        # (se asume que la sede no tiene redes sociales)
        return []

df_sedes_completo['redes_sociales'] = [separar_links(rrss) for rrss in df_sedes_completo['redes_sociales']]

# vamos a necesitar la plataforma
def obtener_plataforma(link):
    """
    Dado un link o un arroba, devuelve la plataforma a la cual pertenece.
    Se utiliza para esto la sentencia LIKE de SQL
    """
    consulta_plataforma = """
                SELECT
                CASE
                    WHEN $link LIKE '%facebook%' THEN 'Facebook'
                    WHEN $link LIKE '%twitter%' OR $link LIKE '@%' THEN 'X'
                    WHEN $link LIKE '%linkedin%' THEN 'Linkedin'
                    WHEN $link LIKE '%instagram%' THEN 'Instagram'
                    WHEN $link LIKE '%youtube%' THEN 'Youtube'
                    WHEN $link LIKE '%flickr%' THEN 'Flickr'
                    ELSE 'Desconocida'
                END
                """
    
    plataforma = sql_val^ consulta_plataforma
    return plataforma

# Primero obtenemos los datos del esquema y los guardamos en diccionarios
data_rrss = {'sede_id':[], 'id_rs':[], 'plataforma':[]}

for i, sede in df_sedes_completo.iterrows():
    current_rrss = sede['redes_sociales']
    for j in range(len(current_rrss)):
        current_link = current_rrss[j]
        data_rrss['sede_id'].append(sede['sede_id'])
        data_rrss['id_rs'].append(current_link)
        data_rrss['plataforma'].append(obtener_plataforma(current_link))

esquema_rrss = pd.DataFrame(data_rrss)
# como qatar está duplicado:
esquema_rrss.drop_duplicates(inplace=True)
esquema_rrss.set_index('id_rs', inplace=True)

# ESQUEMA PAIS
df_paises.rename(columns={' name':'name', ' nom':'nom', ' iso2':'iso2', ' iso3':'iso3', ' phone_code':'phone_code'},
                 inplace=True)

consulta_pais = """
                SELECT DISTINCT p.iso3, p.nombre, s.region_geografica
                FROM df_sedes_completo s
                LEFT JOIN df_paises p
                ON s.pais_iso_3 = p.iso3
                """
esquema_pais = sql^ consulta_pais
esquema_pais.set_index('iso3', inplace=True)

#ESQUEMA INVERSION

#nos quedamos solo con los años que nos interesan
df_IED = df_IED[[2018, 2019, 2020, 2021, 2022]]
df_IED['pais'] = df_IED.index

def quitar_tildes(nombre_pais):
    # toma un string y quita todos los tildes sobre vocales
    pais_sin_tildes = nombre_pais
    pais_sin_tildes = pais_sin_tildes.replace('á','a').replace('é','e').replace('ú','u').replace('ó','o').replace('í','i')

    return pais_sin_tildes

df_paises['nombre'] = [quitar_tildes(pais.lower()) for pais in df_paises['nombre'].values]

def pais_a_iso3(nombre_pais):
    """
    Obtiene el iso3 de un pais usando su nombre,
    se usan funciones de SQL para eliminar diferencias
    en el formato de los dos dataframes.
    """

    consulta_pais_iso3 = """
                    SELECT iso3
                    FROM df_paises
                    WHERE REPLACE(nombre, ' ','') =
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE(
                                REPLACE($nombre_pais, '_',''),
                                  'mejico','mexico'),
                                  'nn','ñ'),
                                  'estadosunidos','estadosunidosdeamerica'),
                                  'chinaprovinciade', ''),
                                  'chinaraede',''),
                                  'repdecorea','coreadelsur'),
                                  'katar','qatar'),
                                  'federacionrusa','rusia'),
                                  'czechia','republicacheca'),
                                  'guineabissau','guinea-bissau'),
                                  'polinesiafrances','polinesiafrancesa'),
                                  'santotomeprincipe','santotomeyprincipe'),
                                  'estadodepalestina','palestina'),
                                  'bosniaherzegovina','bosniayherzegovina'),
                                  'bruneidarussalam','brunei'),
                                  'repdemdel','republicademocraticadel'),
                                  'guayana','guayanafrancesa'),
                                  'republicademoldova','moldavia'),
                                  'trinidadtobago','trinidadytobago')
                    OR LOWER(REPLACE(name, ' ','')) = REPLACE($nombre_pais, '_','')
                    """
    pais = sql_val^ consulta_pais_iso3
    if pais != None:
        return pais
    else:
        return nombre_pais

iso3 = [pais_a_iso3(nombre_pais) for nombre_pais in df_IED['pais']]
df_IED['pais'] = iso3

# eliminamos paises que no tienen info sobre inversiones en ningun año
df_IED.dropna(subset=[2018,2019,2020,2021,2022], how="all", inplace = True)

# Esto hace que las columnas de año se fusionen en una sola
# ahora el pk esta formado por pais y año.
esquema_IED = pd.melt(df_IED, id_vars=['pais']\
                    , value_vars=[2018,2019,2020,2021,2022]\
                    ,var_name='anio', value_name='inversion')

# guardamos todos los esquemas en la carpeta TablasLimpias
esquema_pais.to_csv("TablasLimpias/esquema_pais.csv")
esquema_rrss.to_csv("TablasLimpias/esquema_rrss.csv")
esquema_sede.to_csv("TablasLimpias/esquema_sede.csv")
esquema_IED.to_csv("TablasLimpias/esquema_IED.csv")

#################
# REPORTES CON SQL
#################

# para facilitar y acortar las consultas
# queremos usar los indices de los esquemas en las consultas
# nos ahorramos de poner la palabra esquema en cada consulta
sedes = pd.read_csv("./TablasLimpias/esquema_sede.csv")
rrss = pd.read_csv("./TablasLimpias/esquema_rrss.csv")
pais = pd.read_csv("./TablasLimpias/esquema_pais.csv")
IED = pd.read_csv("./TablasLimpias/esquema_IED.csv")

consulta1 = sql ^( """
            SELECT DISTINCT COUNT(sede_id) AS sedes , iso3_sede , AVG(n_secciones) AS secciones_promedio
            FROM sedes
            GROUP BY iso3_sede
            """)

reporte_1 = sql ^( """
            SELECT DISTINCT nombre AS País, sedes, ROUND(secciones_promedio,2) AS secciones_promedio, ROUND(inversion,2) AS inversion_2022
            FROM consulta1 
            INNER JOIN pais ON consulta1.iso3_sede = pais.iso3
            INNER JOIN IED ON consulta1.iso3_sede = IED.pais
            WHERE IED.anio = 2022
            ORDER BY sedes DESC,  País ASC
            """)

reporte_2 =sql ^( """
            SELECT DISTINCT region_geografica, COUNT(iso3) AS paises_con_sedes_en_Argentina, ROUND(AVG(inversion),2) AS inversiones
            FROM pais
            INNER JOIN IED ON pais.iso3 = IED.pais
            WHERE pais.iso3 IN (SELECT iso3_sede FROM sedes) AND IED.anio = 2022
            GROUP BY region_geografica
            ORDER BY inversiones DESC
            """) 

reporte_3 =sql^( """
            SELECT nombre AS Pais, COUNT( DISTINCT plataforma) AS cant_plataformas
            FROM rrss
            INNER JOIN sedes ON rrss.sede_id = sedes.sede_id
            INNER JOIN pais ON sedes.iso3_sede = pais.iso3
            GROUP BY nombre
            ORDER BY cant_plataformas DESC
            """) 

reporte_4 = sql^("""
            SELECT nombre AS Pais, rrss.sede_id AS Sede, plataforma AS Red_Social, id_rs AS URL
            FROM rrss
            INNER JOIN sedes ON rrss.sede_id = sedes.sede_id
            INNER JOIN pais ON sedes.iso3_sede = pais.iso3
            ORDER BY Pais ASC, Sede ASC, Red_Social ASC, URL ASC
            """)


###################
#   VISUALIZACION
###################

#### 1
# Creamos el gráfico de barras
paleta_verde = ["#008000","#008A00","#009400","#009E00","#00A800","#00B200","#00BC00","#00C600","#00D000"]

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9.0

# Graficar el diagrama de barras
ax.bar(data = reporte_2.sort_values(by='paises_con_sedes_en_Argentina', ascending=False),
       x='region_geografica', height='paises_con_sedes_en_Argentina', 
       color=paleta_verde)

# Añadimos etiquetas y título
ax.set_xlabel('Región Geográfica')  
ax.set_ylabel('Número de Sedes Argentinas')
ax.set_title('Número de Sedes Argentinas por Región Geográfica')

# Rotamos las etiquetas del eje x 
plt.xticks(rotation=45, ha='right')
fig.savefig('AnalisisResultados/grafico1.jpg')

#### 2
plt.clf()
# tabla intermedia a usar en la siguiente consulta,
# obtiene el porcentaje de inversion 2018-22 por cada pais.

avg_inversion_pais = sql ^("""
                    SELECT pais, ROUND(AVG(inversion), 2) AS avg_inversion
                    FROM esquema_IED
                    GROUP BY pais
                    """)

consulta_boxplot = sql ^("""
                SELECT p.region_geografica, avg.avg_inversion AS promedio_inversiones
                FROM pais p INNER JOIN avg_inversion_pais avg
                ON p.iso3 = avg.pais
                WHERE p.iso3 IN (SELECT iso3_sede FROM sedes)
                """)

# calculamos la mediana de inversion de cada region
medianas_region = {}
for region in consulta_boxplot["region_geografica"].unique():
    mediana = consulta_boxplot[consulta_boxplot["region_geografica"] == region]["promedio_inversiones"].median()
    medianas_region[region] = mediana

fig, ax = plt.subplots()

ax = sns.boxplot(x="region_geografica", 
                 y="promedio_inversiones",  
                 data=consulta_boxplot,
                 order = sorted(medianas_region, key=medianas_region.get)
                 )

# Añadimos etiquetas y título
plt.xlabel('Región Geográfica')
plt.ylabel('Promedio de Inversiones')
plt.title('Boxplot del Promedio de Inversiones por Región Geográfica')

plt.xticks(rotation=45, ha='right')
fig.savefig('AnalisisResultados/grafico2.jpg')

##### 3
plt.clf()

consulta7 = sql^("""
        SELECT sedes, AVG(inversion_2022) AS promedio_de_inversiones
        FROM reporte_1
        GROUP BY sedes
            """)
            
consulta7 = consulta7[(consulta7['promedio_de_inversiones'] > 0)]

fig, ax = plt.subplots()

q_95 = consulta7["promedio_de_inversiones"].quantile(0.95)
q_75 = consulta7["promedio_de_inversiones"].quantile(0.75)
q_50 = consulta7["promedio_de_inversiones"].quantile(0.50)
q_25 = consulta7["promedio_de_inversiones"].quantile(0.25)

def asignar_intensidad(avg):
    if avg >= q_95:
        return "#FF0000"
    elif avg >= q_75:
        return "#FF3333"
    elif avg >= q_50:
        return "#FF6666"
    else:
        return "#FFCCCC"

colores = [asignar_intensidad(avg) for avg in consulta7['promedio_de_inversiones']]

plt.rcParams['font.family'] = 'sans-serif'           


ax.bar(data=consulta7, x='sedes', height='promedio_de_inversiones', color = colores)
       
ax.set_title('Relacion de Cantidad de Sedes - Inversiones 2022')
ax.set_xlabel('cantidad de sedes', fontsize='medium')                       
ax.set_ylabel('Promedio de Inversiones (M U$S) por cantidad de sedes', fontsize='medium')    
ax.set_xlim(0, 12)
ax.set_ylim(0, max(consulta7["promedio_de_inversiones"] + 15000))

ax.set_xticks(range(1,12,1))               # Muestra todos los ticks del eje x
ax.set_yticks([])                          # Remueve los ticks del eje y
ax.bar_label(ax.containers[0], fontsize=8)

fig.savefig('AnalisisResultados/grafico3A.jpg')

### OTRO GRAFICO PARA EL PUNTO 3 
plt.clf()
# Filtramos los datos por numero de sedes e inversion
reporte_1_positivos = reporte_1[(reporte_1['inversion_2022'] > 0) & ((reporte_1['sedes'] >= 2) | (reporte_1['inversion_2022'] > 10000)) ]


# Ordenamos los datos por inversión en 2022 de forma descendente
reporte_1_sorted = reporte_1_positivos.sort_values(by='inversion_2022', ascending=False)

# Creamos el gráfico de barras
fig, ax = plt.subplots()

colores = ['red' if pais in ["Uruguay","Chile","Paraguay","Brasil","Bolivia"] else 'skyblue' for pais in reporte_1_sorted['País']]

ax.bar(data=reporte_1_sorted, x='País', height='inversion_2022', width=0.5, color = colores)

# Añadimos números de sedes como anotaciones en las barras
for i, (pais, inversion, sedes) in enumerate(zip(reporte_1_sorted['País'], reporte_1_sorted['inversion_2022'], reporte_1_sorted['sedes'])):
  ax.text(i, inversion + 10000, str(sedes), ha='center', va='bottom', rotation=0,  fontsize='5')

# Establecemos título y etiquetas
ax.set_title('Relación Cantidad de Sedes - IED 2022')
ax.set_xlabel('Países y su cantidad de sedes')
ax.set_ylabel('IED 2022 (M U$S)')
ax.set_xlim(-1, 11)

# Asignamos un tick a cada barra en el eje x
ax.set_xticks(range(len(reporte_1_sorted)))
ax.set_xticklabels(reporte_1_sorted['País'], rotation=90)

# Achicamos y rotamos ticks
plt.xticks(rotation=90)
plt.xticks(size=6)
ax.tick_params(axis='y', labelsize='7')

# Aumentamos un poco la calidad
plt.figure(figsize=(14, 10), dpi=200)


# Saco los bordes de arriba y la derecha
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig('AnalisisResultados/grafico3B.jpg')

### UN ULTIMO GRAFICO MAS ENCLARECEDOR
plt.clf()
sedes_vs_inversion = sql^("""
        SELECT CASE WHEN sedes < 3 THEN '0-2'
                WHEN sedes >= 3 AND sedes < 6 THEN '3-5'
                WHEN sedes >= 6 AND sedes < 9 THEN '6-8'
                WHEN sedes >= 9 THEN '9-max'
                END AS sedes, 
                AVG(inversion_2022) AS promedio_de_inversiones
        FROM reporte_1
        GROUP BY sedes
        ORDER BY sedes
            """)
            
sedes_vs_inversion = sedes_vs_inversion[(sedes_vs_inversion['promedio_de_inversiones'] > 0)]

fig, ax = plt.subplots()


plt.rcParams['font.family'] = 'sans-serif'           

ax.bar(data=sedes_vs_inversion, x='sedes', height='promedio_de_inversiones', color = ["#FF0000","#FF3333","#FF6666","#FFCCCC"])
       
ax.set_title('Relacion de Cantidad de Sedes - Inversiones 2022')
ax.set_xlabel('cantidad de sedes', fontsize='medium')                       
ax.set_ylabel('Promedio de Inversiones (M U$S) por cantidad de sedes', fontsize='medium')    
ax.set_ylim(0, max(sedes_vs_inversion["promedio_de_inversiones"] + 15000))

fig.savefig('AnalisisResultados/grafico3C.jpg')