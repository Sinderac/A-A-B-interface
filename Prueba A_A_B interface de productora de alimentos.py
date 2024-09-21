#!/usr/bin/env python
# coding: utf-8

# # Sprint 11: Proyecto integrado 2

# Analizaremos los datos de un dataframe de una prueba A/A/B, evaluaremos el desempeño de la prueba B usando dos prueba A. 
# Previamente habiando filtrado, depurado y corregido los datos base.

# ## Inicialización

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ## Preparar los datos

# In[2]:


logs_df = pd.read_csv("/datasets/logs_exp_us.csv", 
                      sep= "\t" 
                     )


# In[3]:


logs_df.head(5)


# In[4]:


#Renombramos las columnas
logs_df= logs_df.rename(columns = { "EventName": "event",
                          "DeviceIDHash": "id",
                          "EventTimestamp": "timestamp",
                          "ExpId": "group"
                         }
              )


# In[5]:


#Revisamos los tipos de valores
logs_df.info()


# In[6]:


#Buscaremos valores ausentes

logs_df.isna().sum()


# In[7]:


#Buscaremos valores duplicados

logs_df.duplicated().sum()


# In[8]:


#Hacemos un dataframe con los valores duplicados
duplicated = logs_df[logs_df.duplicated()]

#Buscamos si hya alguna constante en los valores duplicados
duplicated.value_counts()


# In[9]:


#Vemos si hay un rango de fechas donde se encuentran los valores duplicados
duplicated["timestamp"].describe()


# No logro determinar alguna constante en estos valores duplicados. Hay una diferencia considerable en las fechas de una a otra, diferentes eventos, diferentes grupos y dispositivos.
# Eliminaremos los valores duplicados.

# In[10]:


#Eliminamos valores duplicados
logs_df = logs_df.drop_duplicates()


# In[11]:


#Creamos la columna date
logs_df["date"] = (pd.to_datetime(logs_df["timestamp"], 
                                  unit= "s")
                  )


# In[12]:


logs_df["only_date"] = logs_df["date"].dt.date
logs_df["only_hour"] = logs_df["date"].dt.hour


# In[13]:


logs_df

# ## Estudiar y  comprobar los datos

# ### ¿Cuántos eventos hay en los registros?

# In[14]:


#Mostramos los valores unicos de evento
logs_df["event"].unique()


# In[15]:


#Los contamos
logs_df["event"].nunique()


# Vemos que hay 5 diferentes eventos:
# MainScreenAppear, PaymentScreenSuccessful, CartScreenAppear, OffersScreenAppear y Tutorial

# In[16]:


logs_df["event"].count()


# Hay un total de 243,713 registros de eventos.

# ### ¿Cuántos usuarios y usuarias hay en los registros?

# In[17]:


logs_df["id"].nunique()


# Tenemos 7,551 usuarios unicos en los datos.

# ### ¿Cuál es el promedio de eventos por usuario?

# In[18]:


# Contar el número de eventos por usuario
events_per_user = logs_df.groupby("id").size()

# Calcular el promedio de eventos por usuario
average_events_per_user = events_per_user.mean().round(2)

average_events_per_user


# La cantidad promedio de eventos que pasan los usuarios son de 32.28

# ### ¿Qué periodo de tiempo cubren los datos? Encuentra la fecha máxima y mínima. Traza un histograma por fecha y hora. 
# ¿Puedes tener seguridad de que tienes datos igualmente completos para todo el periodo? Los eventos más antiguos podrían terminar en los registros de algunos usuarios o usuarias por razones técnicas y esto podría sesgar el panorama general. Encuentra el momento en el que los datos comienzan a estar completos e ignora la sección anterior. ¿Qué periodo representan realmente los datos?

# In[19]:


#Buscamos la fecha minima y maxima
print("El dia maximo fue:", logs_df["date"].max())
print("El dia minimo fue:", logs_df["date"].min())


# La primera fue 2019-07-25 04:43:36   
# La ultima fue 2019-08-07 21:15:17

# In[20]:


#Hacemos un histograma
sns.histplot(data=logs_df,
            x=logs_df["date"],
            kde=True,
            bins=30,
            )

#Agregamos titulo
plt.title("Comportamiento de los datos a traves del tiempo")
#rotamos los labels en X
plt.xticks(rotation=45)



#mostramos la grafica
plt.show()


# In[21]:


# Agrupamos los datos por fecha y contar los registros por día
daily_counts = logs_df.groupby(logs_df["date"].dt.date).size()

# Creamos un plot para ver la frecuencia de datos por fecha
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts.values, marker="o")
plt.title("Frecuencia de Datos por Fecha")
plt.xlabel("Fecha")
plt.ylabel("Número de Registros")
plt.xticks(rotation=45)
plt.show()


# Con la gráfica vemos que la mayoría de los datos son despues del 1 de agosto del 2019.

# In[22]:


# Filtramos el dataframe para datos despues del 1 de agosto del 2019
filtered_logs_df= logs_df[logs_df["date"]> "2019-08-01"]


# In[23]:


# Agrupamos los datos por fecha y contar los registros por día
daily_counts = filtered_logs_df.groupby(logs_df["date"].dt.date).size()

# Creamos un plot para ver la frecuencia de datos por fecha
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts.values, marker="o")
plt.title("Frecuencia de Datos por Fecha")
plt.xlabel("Fecha")
plt.ylabel("Número de Registros")
plt.xticks(rotation=45)
plt.show()


# Ya tenemos los datos en el periodo donde se encuentra la mayoria de los datos. 
# 
# Es del 1 al 7 de agosto del 2019.
# 
# Hay que revisar cuantos datos se excluyeron.

# In[24]:


# Revisemos la cantidad de datos, antes y despues del filtro.
count = logs_df["id"].count()
filtered_count = filtered_logs_df["id"].count()
percentage_data = filtered_count *100 / count
percentage_data.round(2)


# Vemos que nuestros datos filtrados representan el 98.84% del dataframe anterior, por lo tanto solo excluimos el 1.16% de los datos con el nuevo periodo, lo cual son la minoría y no tendrá un impacto significativo el quitar estos datos del análisis.

# ### ¿Perdiste muchos eventos y usuarios al excluir los datos más antiguos?

# In[25]:


#Calculamos la cantidad de eventos y total de usuarios excluidos
loss_count = count-filtered_count

loss_count


# Se perdieron en todal 2,2826 datos del dataframe anterior. Revisemos si se perdieron usarios unicos.

# In[26]:


#Obtenemos los usuarios unicos del filtro
filtered_unique_users = filtered_logs_df["id"].unique()

#Obtenemos la cantidad de usuarios unicos antes del filtro
unique_users_count = logs_df["id"].nunique()

#Obtenemos la cantidad de usuarios unicos despues del filtro
filtered_unique_users_count =filtered_logs_df["id"].nunique()

#Contamos los usuarios que ya no se encuentran en el dataframe filtrado
excluded_users_count = logs_df[~logs_df["id"].isin(filtered_unique_users)]["id"].count()

percentage_user_count_lost = 100-(filtered_unique_users_count*100/unique_users_count)

print("El porcentaje que de los usuarios excluidos es el:", percentage_user_count_lost)
print("La cantidad de usuarios excluidos fue: ", excluded_users_count)


# La cantidad de usuarios que ya no se encuentran en el dataframe es minimo, el 0.22%

# ### Asegúrate de tener usuarios y usuarias de los tres grupos experimentales.

# In[27]:


#Contamos los usurios unicos de cada grupo y obtenemos el total
group_246_count = filtered_logs_df[filtered_logs_df["group"]==246]["id"].nunique()
group_247_count = filtered_logs_df[filtered_logs_df["group"]==247]["id"].nunique()
group_248_count = filtered_logs_df[filtered_logs_df["group"]==248]["id"].nunique()

print("La cantidad de usuarios del grupo 246 es:", group_246_count)
print("La cantidad de usuarios del grupo 247 es:", group_247_count)
print("La cantidad de usuarios del grupo 248 es:", group_248_count)


# Si tenemos usuarios en los tres grupos experimentales, y no están muy desfasados uno del otro.

# ### Observa qué eventos hay en los registros y su frecuencia de suceso. Ordénalos por frecuencia.

# In[28]:


#Revisamos cuales son los diferentes eventos que hay
filtered_logs_df["event"].unique()


# Siguen siendo MainScreenAppear, PaymentScreenSuccessful, CartScreenAppear, OffersScreenAppear y Tutorial

# In[29]:


#Veamos su frecuencia de cada evento
event_count= filtered_logs_df.groupby("event").size().reset_index(name= "count")

#Acomodamos los datos de manera descendente
event_count = event_count.sort_values(by="count", ascending= False)

#Preparamos una grafica tipo bar para mostrar los datos
sns.barplot(data=event_count,
           x="event",
           y="count")

plt.xticks(rotation=45)
plt.title("Frecuencia total de eventos")
plt.xlabel("Eventos")
plt.ylabel("Frecuencia")

#Mostramos la grafica
plt.show()


# 1er Lugar lo tuvo Mainscreen, tiene sentido ya que es la pantalla principal  
# 2do  offersScreen, muestran un interes en la pantalla de las ofertas  
# 3er CartScreen, andan preparando la compra  
# 4to paymentScreenSuccessful, si se ve un desfase al anterior lugar pero tiene sentido que se encuentre en este lugar  
# 5to tutorial, este parece que muy pocos se meten al tutorial

# ### Encuentra la cantidad de usuarios y usuarias que realizaron cada una de estas acciones. Ordena los eventos por el número de usuarios y usuarias. Calcula la proporción de usuarios y usuarias que realizaron la acción al menos una vez

# In[30]:


#Calculemos la cantidad de usuarios unicos por cada evento
event_unique_user_count =(filtered_logs_df.groupby("event")["id"]
                          .nunique().reset_index(name= "count")
                          .sort_values(by="count", ascending= False)
                         )

#Preparamos una grafica tipo bar para mostrar los datos
sns.barplot(data=event_unique_user_count,
           x="event",
           y="count")

plt.xticks(rotation=45)
plt.title("Cantidad de usuarios unicos por evento")
plt.xlabel("Eventos")
plt.ylabel("Cantidad de usuarios unicos")

#Mostramos la grafica
plt.show()


# In[31]:


#Calculemos el total de usuarios unicos
total_unique_users = filtered_logs_df["id"].nunique()

#calculemos las proporciones
event_unique_user_count["percentage"] = (event_unique_user_count["count"] 
                                         *100
                                         / total_unique_users
                                        )
event_unique_user_count["percentage"] = event_unique_user_count["percentage"].round(2)


#Preparamos una grafica tipo bar para mostrar los datos
sns.barplot(data=event_unique_user_count,
           x="event",
           y="percentage")

plt.xticks(rotation=45)
plt.title("¨Porcentaje de usuarios unicos que pasaron por los eventos")
plt.xlabel("Eventos")
plt.ylabel("Porcentaje de usuarios unicos")

#Mostramos la grafica
plt.show()


# Los porcentajes de usuarios unicos que entraron almenos una vez a un evento fue:  
# Parece que hubo un porcentaje que no paso por mainscreen. LLego como al 98%  
# El offerscreen sigue en 2do lugar casi llegando al 60%  
# Cartsscreen no muy atras con alredor del 50%  
# Paymentscreen por debajo del 50%  
# y tutorial menos del 10%
# 
# En el caso del 98% de mainscreen, no se si es el efecto por borrar datos viejos o si nomas entraron a ver el tutorial pero no pasaron por mainscreen. Pero como veo que solo es un 2% no creo que sea necesario invertirle el tiempo para averiguarlo.
# 
# Al menos que me digan lo contrario (:

# ### ¿En qué orden crees que ocurrieron las acciones? ¿Todas son parte de una sola secuencia? No es necesario tenerlas en cuenta al calcular el embudo.

# Viendo los resultados anteriores, el orden sería:  
# 1.- Mainscreen  
# 2.- Offerscreen  
# 3.- Paymentscreen  
# 
# Tutorial quería como un evento opcional que no es necesario para completar los eventos anteriores.

# ### Utiliza el embudo de eventos para encontrar la proporción de usuarios y usuarias que pasan de una etapa a la siguiente. (Por ejemplo, para la secuencia de eventos A → B → C, calcula la proporción de usuarios en la etapa B a la cantidad de usuarios en la etapa A y la proporción de usuarios en la etapa C a la cantidad en la etapa B).

# In[32]:


#Hacemos una tabla tipo pivot de cuando los usuarios entraron por primera vez a un evento
users = filtered_logs_df.pivot_table(
    index="id", 
    columns="event", 
    values="date",
    aggfunc="min")

#Iniciamos con el embudo de eventos

#Excluimos los usuarios que no entraron al mainscreen
step1 = ~users["MainScreenAppear"].isna()

#Consideramos los usuarios que pasaron por el step 1 y que pasaron por offers screen despues de main screen
step2 = step1 & (users["OffersScreenAppear"] > users["MainScreenAppear"])

#Consideramos los usuarios que pasaron por el step 2 y que pasaron por cart screen despues de offers screen
step3 = step2 & (users["CartScreenAppear"] > users["OffersScreenAppear"])

#Consideramos los usuarios que pasaron por el step 3 y que pasaron por payment screen despues de cart screen
step4 = step3 & (users["PaymentScreenSuccessful"] > users["CartScreenAppear"])

#Vemos el tamaño de los datos para obtener las cantidad de usuarios por cada etapa del embudo de eventos
n_main = users[step1].shape[0]
n_offers = users[step2].shape[0]
n_carts = users[step3].shape[0]
n_payments = users[step4].shape[0]


#Imprimos los resultados
print("Entraron a la pantalla principal", n_main, "usuarios unicos")
print("Entraron a la pantalla de ofertas", n_offers, "usuarios unicos")
print("Entraron a la pantalla de carrito", n_carts, "usuarios unicos")
print("Realizaron su compra de manera exitosa", n_payments, "usuarios unicos")


# In[33]:


#Calculamos el porcentaje de main a offer
AB_percentage = n_offers * 100 / n_main

#Calculamos el porcentaje de offer a cart
BC_percentage = n_carts * 100 / n_offers

#Calculamos el porcentaje de cart a payment
CD_percentage = n_payments *100 / n_carts

print("El porcentaje de usuarios que pasaron del main screen al offer screen es:", AB_percentage)
print("El porcentaje de usuarios que pasaron del offer screen al cart screen es:", BC_percentage)
print("El porcentaje de usuarios que pasaron del cart screen al payment screen es:", CD_percentage)


# En cuestión de porcentaje donde menos usuarios pasan es de la pantalla del carrito a la de la compra con un 25.69%  
# Ya con una diferencia el 42.06% pasa de la pantalla de oferta a la de compra,  
# y con el 56.62% pasa de la pantalla principal a la de ofertas

# ### ¿En qué etapa pierdes más usuarios y usuarias?

# In[34]:


print("La cantidad de usuarios perdidos de main screen a offer screen es:", n_main - n_offers)
print("La cantidad de usuarios perdidos de offer screen a cart screen es:", n_offers - n_carts)
print("La cantidad de usuarios perdidos de cart screen a payment screen es:", n_carts - n_payments)


# Si vemos en cuestión de cantidad, donde se pierden más usuarios es del main screen a offer screen.  

# ### ¿Qué porcentaje de usuarios y usuarias hace todo el viaje desde su primer evento hasta el pago?

# In[35]:


print("El porcentaje de usaurios que pasan de la pantalla principal hasta realizar una compra es:", n_payments * 100 / n_main)


# Resultó ser el 6.12%. No se si es mucho o si es poco, se tendría que comparar con análisis anteriores para ver si es mucho es poco, aumento o bajo.  
# A simplemente vista si parece poco pero desconozco como han estado anteriormente.

# In[36]:


import plotly.graph_objects as go

# Crear la figura del embudo con las etapas del embudo y los usuarios de cada etapa
fig = go.Figure(go.Funnel(
    y = ['Pantalla Principal', 'Ofertas', 'Carrito', 'Pago Exitoso'],
    x = [n_main, n_offers, n_carts, n_payments],  # Números que calculaste para cada etapa
    textinfo = "value+percent initial",
    marker = {"color": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]}
))

# Agregar título
fig.update_layout(title="Embudo de Conversión de Usuarios")

# Mostrar el gráfico interactivo
fig.show()


# ## Estudiar los resultados del experimento

# ### ¿Cuántos usuarios y usuarias hay en cada grupo?

# In[37]:


#Contamos los usurios unicos de cada grupo y obtenemos el total
group_246_count = filtered_logs_df[filtered_logs_df["group"]==246]["id"].nunique()
group_247_count = filtered_logs_df[filtered_logs_df["group"]==247]["id"].nunique()
group_248_count = filtered_logs_df[filtered_logs_df["group"]==248]["id"].nunique()

print("La cantidad de usuarios del grupo 246 es:", group_246_count)
print("La cantidad de usuarios del grupo 247 es:", group_247_count)
print("La cantidad de usuarios del grupo 248 es:", group_248_count)


# No se ve una diferencia considerable entre la cantidad de usuarios de un grupo a otro.

# ### Tenemos dos grupos de control en el test A/A, donde comprobamos nuestros mecanismos y cálculos. Observa si hay una diferencia estadísticamente significativa entre las muestras 246 y 247.

# In[38]:


#Filtramos los datos para hacer dataframes respecto a cada grupo
group_246 = filtered_logs_df[filtered_logs_df["group"]==246][["event","id","date"]]
group_247 = filtered_logs_df[filtered_logs_df["group"]==247][["event","id","date"]]

#Hacemos que dimensione cada tipo de evento
counts_246 = group_246.groupby("event").size()
counts_247 = group_247.groupby("event").size()

#Obtenemos los valores de los conteos
data_246 = counts_246.values
data_247 = counts_247.values

#Obtenemos la cantidad de veces que entraron a la pantalla de pagos
payment_count_246= data_246[3]
payment_count_247= data_247[3]

#Obtenemos la suma total
total_246 = sum(data_246)
total_247 = sum(data_247)

# Asegúrate de que tus variables estén correctamente definidas
success_counts = [payment_count_246, payment_count_247]
total_counts = [total_246, total_247]

# Realizamos la prueba Z
z_stat, p_value = sm.stats.proportions_ztest(success_counts, total_counts)

# Imprimimos los resultados
print("\nT-statistic:", z_stat)
print("P-value:", p_value)


# La T-statistic esta lejos de cero, lo cual quiere decir que hay una diferencia significativa entre los datos.
# Mientras el valor de P es menor a 0.05, lo cual confirma que hay una diferencia estadistiva significativa entre ambos grupos. 
# Entonces ambos grupos de control tienen resultados diferentes.


# ### Selecciona el evento más popular. En cada uno de los grupos de control, encuentra la cantidad de usuarios y usuarias que realizaron esta acción. Encuentra su proporción. Comprueba si la diferencia entre los grupos es estadísticamente significativa. Repite el procedimiento para todos los demás eventos (ahorrarás tiempo si creas una función especial para esta prueba). ¿Puedes confirmar que los grupos se dividieron correctamente?

# In[39]:


# Contamos la cantidad de veces que ocurre cada evento
event_counts = filtered_logs_df["event"].value_counts()

# Seleccionamos el evento más popular
most_popular_event = event_counts.idxmax()

# Filtramos el DataFrame para incluir solo el evento más popular
event_df = filtered_logs_df[filtered_logs_df["event"] == most_popular_event]

# Contamos la cantidad de usuarios por grupo que realizaron el evento más popular
group_event_counts = event_df.groupby("group")["id"].nunique()

# Mostramos los resultados
print("El evento más popular fue:")
print(most_popular_event)
print()
print("Cantidad de usuarios por grupo que realizaron el evento más popular:")
print(group_event_counts)


# In[40]:


def prueba_z(group1, group2, event_name, df, alpha):
    """
    Realiza la prueba de Z para comparar dos grupos en cuanto a la realización de un evento.
    
    Parameters:
    - group1: El primer grupo a comparar.
    - group2: El segundo grupo a comparar.
    - event_name: El nombre del evento a analizar.
    - df: DataFrame que contiene los datos.
    """
    total_count_group1 = df[df["group"] == group1]["id"].size
    total_count_group2 = df[df["group"] == group2]["id"].size
    
    # Conteo de éxitos para cada grupo
    success_count_group1 = df[(df["group"] == group1) & (df["event"] == event_name)].shape[0]
    success_count_group2 = df[(df["group"] == group2) & (df["event"] == event_name)].shape[0]
    
    total_counts = [total_count_group1, total_count_group2]
    success_counts = [success_count_group1, success_count_group2]
    
    # Verificamos que no haya ceros en total_counts
    if 0 in total_counts:
        print(f"No hay datos suficientes para realizar la prueba Z entre grupo {group1} y grupo {group2} para el evento '{event_name}'.")
        return
    
    # Realizamos la prueba Z
    stat, p_value = sm.stats.proportions_ztest(success_counts, total_counts)
    
    # Solo imprimiremos valores donde haya una diferencia estadística significativa
    print(f"Comparación entre grupo {group1} y grupo {group2} para el evento '{event_name}':")
    if p_value < alpha:
        print(f"Estadístico Z: {stat}")
        print(f"P-valor: {p_value}")
        print("")
    else:
        print("No hay diferencia estadística significativa")
        print("")

def analyze_event(event_name, df):    
    # Realizamos la prueba Z entre cada par de grupos con alpha = 0.05
    prueba_z(246, 247, event_name, df, 0.05)

# Recorremos todos los eventos y aplicamos la función
for event in filtered_logs_df["event"].unique():
    print(f"Analizando el evento: {event}")
    analyze_event(event, filtered_logs_df)


# No parece que haya alguna diferencia considerable entre los grupos. Los grupos de control son similares.

# In[42]:


#Hacemos las pruebas entre los grupos de control y el alterno con alpha = 0.05
def analyze_event(event_name, df):
    prueba_z(246, 248, event_name, df, 0.05)
    prueba_z(247, 248, event_name, df, 0.05)
# Recorremos todos los eventos y aplicamos la función
for event in filtered_logs_df["event"].unique():
    print(f"Analizando el evento: {event}")
    analyze_event(event, filtered_logs_df)


# El grupo alterno (248) tuvo un efecto positivo en varios eventos, pero también mostró un efecto negativo en el evento "OffersScreenAppear". Esto sugiere que, si bien algunos cambios han mejorado las visitas o la interacción, en otros casos podrían haber llevado a una disminución en la efectividad. Podría ser útil profundizar en las razones detrás de estos resultados y considerar ajustes en la estrategia según los eventos y su desempeño.

# ### ¿Qué nivel de significancia has establecido para probar las hipótesis estadísticas mencionadas anteriormente? Calcula cuántas pruebas de hipótesis estadísticas has realizado. Con un nivel de significancia estadística de 0.1, uno de cada 10 resultados podría ser falso. ¿Cuál debería ser el nivel de significancia? Si deseas cambiarlo, vuelve a ejecutar los pasos anteriores y comprueba tus conclusiones.

# Previamente considere un nivel de significancia de 0.05.  
# Son 3 grupos con 5 eventos, esto da un total de 15 pruebas.

# Nivel de Significancia Ajustado (Bonferroni)
# 
# Nivel de significancia deseado: 0.1
# Número de pruebas: 15
# Nivel de significancia ajustado = Nivel de significancia deseado / Número de pruebas
# Nivel de significancia ajustado = 0.1 / 15 ≈ 0.0067

# In[44]:


#Hacemos las pruebas entre los grupos de control y el alterno con alpha = 0.0067
def analyze_event(event_name, df):
    prueba_z(246, 248, event_name, df, 0.0067)
    prueba_z(247, 248, event_name, df, 0.0067)
# Recorremos todos los eventos y aplicamos la función
for event in filtered_logs_df["event"].unique():
    print(f"Analizando el evento: {event}")
    analyze_event(event, filtered_logs_df)


# El grupo alterno (248) mostró un efecto positivo en varios eventos clave, como "MainScreenAppear" y "PaymentScreenSuccessful" en comparación con el grupo 246, pero también tuvo efectos negativos en otros eventos como "OffersScreenAppear" y "PaymentScreenSuccessful" en comparación con el grupo 247. Esto sugiere que, aunque algunas modificaciones han mejorado la interacción en ciertos aspectos, han llevado a una disminución en otros. Sería útil investigar más a fondo las razones detrás de estas diferencias para ajustar estrategias futuras.


# Nivel de Significancia Ajustado (Bonferroni)
# 
# Nivel de significancia deseado: 0.05
# Número de pruebas: 15
# Nivel de significancia ajustado = Nivel de significancia deseado / Número de pruebas
# Nivel de significancia ajustado = 0.05 / 15 ≈ 0.0033

# In[45]:


#Hacemos las pruebas entre los grupos de control y el alterno con alpha = 0.0067
def analyze_event(event_name, df):
    prueba_z(246, 248, event_name, df, 0.0033)
    prueba_z(247, 248, event_name, df, 0.0033)
# Recorremos todos los eventos y aplicamos la función
for event in filtered_logs_df["event"].unique():
    print(f"Analizando el evento: {event}")
    analyze_event(event, filtered_logs_df)


# El grupo alterno (248) ha tenido un efecto positivo en algunos eventos clave, como "MainScreenAppear" y "PaymentScreenSuccessful" en comparación con el grupo 246. Sin embargo, también se observan efectos negativos en otros eventos, como "OffersScreenAppear" y "PaymentScreenSuccessful" en comparación con el grupo 247. Esto sugiere que, si bien algunos cambios han mejorado la interacción, también han llevado a una disminución en otros aspectos.

# ## Conclusión

# Dependiendo del nivel de significancia que seleccionamos si varía un poco los resultados, pero en la mayoría si hubo un efecto negativo en el OfferScreen y contra un grupo con el PaymentScreen.
# 
# Si hay área de oportunidad para mejorar estas interfaces y ver el feedback de los usuarios para ver que les parecieron los cambios y poder saber que se tiene que mejorar para no tener un efecto negativo con las etapas.

# In[ ]:




