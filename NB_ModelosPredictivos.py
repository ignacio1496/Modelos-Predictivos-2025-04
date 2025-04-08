#!/usr/bin/env python
# coding: utf-8

# ## NB_ModelosPredictivos
# 
# New notebook

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import iplot, plot
from plotly.subplots import make_subplots

from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing

import warnings
warnings.filterwarnings('ignore')

colors = ["#b1e7cd","#854442","#000000","#fff4e6","#3c2f2f",
         "#be9b7b ","#512E5F","#45B39D","#AAB7B8 ","#20B2AA",
         "#FF69B4","#00CED1","#FF7F50","#7FFF00","#DA70D6"]

csv_path = f"{notebookutils.nbResPath}/builtin/Superstore.csv"

# Probar con otro encoding
df = pd.read_csv(csv_path, delimiter=',', encoding='latin1')

display(df)



# In[2]:


print(f'the number of rows is : {df.shape[0]} \nThe number of columns is : {df.shape[1]} '.upper() )


# In[3]:


df.columns


# In[4]:


df.info()


# In[5]:


df.describe().T


# In[6]:


df.isna().sum()


# In[7]:


df.duplicated().any()


# In[8]:


pd.DataFrame({
   'Countd' :df.shape[0] ,
   'Null' : df.isnull().sum() ,
   'Null%' : df.isnull().mean()*100 ,
   'cardinality' : df.nunique()
})


# In[9]:


df.describe(include=object).T


# In[10]:


df[['Order Date','Ship Date']].dtypes


# In[11]:


df['Order Date']=pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])


# # ¿Cuáles son los productos más vendidos?

# In[12]:


top5_selling_products = df.groupby(['Product Name'])['Sales'].sum().sort_values(ascending=False).reset_index().head()
top5_selling_products


# In[13]:


fig = px.bar(top5_selling_products,x= top5_selling_products['Product Name'] ,y='Sales',
            template='plotly_dark',
            color_discrete_sequence=[["#be9b7b","#fff4e6","#3c2f2f","#b1e7cd ","#45B39D"]] ,
            title='Los 5 Productos más vendidos '.upper() ,
            text_auto=True , 
            
            )

fig.update_layout(
   xaxis_title='Sales',
)

iplot(fig)


# # Productos más rentables

# In[14]:


top5_profitable_products = df.groupby(['Product Name'])['Profit'].sum().sort_values(ascending=False).head()
top5_profitable_products


# In[15]:


fig = px.bar(top5_profitable_products,x = top5_profitable_products ,
            template='plotly_dark',
            color= top5_profitable_products,
            title='Los 5 Productos más rentables '.upper() ,
            text_auto=True , 
            
            )

fig.update_layout(
   xaxis_title='total profit ',
)

iplot(fig)


# # ¿Cuál es la tendencia de ventas a lo largo del tiempo (mensual, anual)?

# In[16]:


df['year'] =df['Order Date'].dt.year
df['month'] = df['Order Date'].dt.to_period('M').astype(str)

monthly_sales = df.groupby('month')['Sales'].sum().reset_index()
yearly_sales = df.groupby('year')['Sales'].sum().reset_index()

fig = px.bar(yearly_sales,
            x='year',
            y='Sales',
            template='plotly_dark',
            color_discrete_sequence=['#C0C0C0'],
            text_auto=True)

fig.update_layout(
   bargap = 0.6,
   title='Tendencia de ventas anuales',
   xaxis_title='Años',
   yaxis_title='Total de Ventas',
)
fig.show()


# In[17]:


fig = px.line(monthly_sales,
            x=monthly_sales['month'],
            y=monthly_sales['Sales'],
            template='plotly_dark',
            color_discrete_sequence=['#b1e7cd']
            )
fig.update_layout(
   title = 'Tendencia de ventas mensuales' ,
   yaxis_title = 'Total Ventas' ,
   xaxis_title = 'Meses'
)
fig.show()


# ** La tendencia general de la empresa es la ganancia, pero hay algunos períodos en los que las ganancias disminuyen y luego aumentan
# otra vez. Se concluye que es por los descuentos**

# # ¿Qué categoría de productos genera el mayor beneficio?

# In[18]:


category = df.groupby('Category')['Profit'].sum().sort_values(ascending=False).reset_index()
category

fig = px.bar(category,
            y=category['Category'],
            x=category['Profit'],
            template='plotly_dark',
            color_discrete_sequence=[colors[8]],text_auto=True
            )

fig.update_layout(
   bargap = 0.6,
   title='Beneficio total para cada categoría',
   )

fig.show()


# In[19]:


fig = px.pie(category,
            names=category['Category'],
            template='plotly_dark',
            color_discrete_sequence=px.colors.sequential.RdBu,hole=0.4,
            values=category['Profit']
         )


fig.update_layout(
   
   title='Porcentaje de beneficios para cada categoría',
   )
fig.show()


# # ¿Qué región genera más ventas?

# In[20]:


Regions_most_sales = df.groupby('Region')['Sales'].sum().reset_index()
fig=px.bar(Regions_most_sales ,
         x=Regions_most_sales['Sales'],
         y='Region',
         template='plotly_dark',
         color_discrete_sequence=[colors[7]],text_auto=True)
fig.update_layout(
   bargap = 0.6,
   title='Más ventas por región',
   )

fig.show()


# # ¿Cuál es el impacto de los descuentos y promociones en las ventas?

# In[21]:


fig= px.scatter(df ,x=df['Discount'],
               y=df['Sales'],
               template='plotly_dark',
               color_discrete_sequence=[colors[7]])
fig.update_layout(
      title='Ventas totales con descuentos',
      yaxis_title = 'Total Ventas'
   )
fig.show()


# In[22]:


discount_group = df.groupby('Discount')['Profit'].sum().reset_index()

fig= px.scatter(discount_group ,
               x=discount_group['Discount'],
               y=discount_group['Profit'],
               template='plotly_dark',
            color_discrete_sequence=[colors[7]]
            )
fig.update_layout(
   
   title='Beneficio total con descuento',
   yaxis_title = 'Total Beneficio'
   )
iplot(fig)


# En resumen
# 
# Las mejores ventas son cuando el descuento es del 50%, pero no es rentable.
# 
# El descuento del 10% al 20%. Es la mejor solución, ya que son el segundo y tercer más vendidos y obtienen ganancias

# # ¿Cuál es el margen de beneficio promedio para cada categoría de producto?

# In[23]:


df['Profit_margin'] = df['Profit'] / df['Sales']
profite_margin_avg = df.groupby('Category')['Profit_margin'].mean().reset_index()
profite_margin_avg['Profit_margin'] = round(profite_margin_avg['Profit_margin'],2)
profite_margin_avg

fig = px.bar(profite_margin_avg,x='Category',y='Profit_margin',template='plotly_dark',
         color_discrete_sequence=[colors[8]],text_auto=True) 
fig.update_layout(
   bargap = 0.7,
   title='Promedio de beneficio para cada categoría de productos' ,
   )

iplot(fig)


# In[24]:


# Convertir la columna de fecha
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Agrupar por mes y sumar ventas
monthly_sales = df.resample('M', on='Order Date')['Sales'].sum()

# Graficar ventas mensuales
plt.figure(figsize=(12, 5))
plt.plot(monthly_sales, label='Ventas Mensuales')
plt.title("Serie de Ventas Mensuales")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.grid(True)
plt.legend()
plt.show()


# In[25]:


# --------------------------------------
# 1. PROMEDIO MÓVIL (Ventana de 3 meses)
# --------------------------------------
rolling_avg = monthly_sales.rolling(window=3).mean()

plt.figure(figsize=(12, 5))
plt.plot(monthly_sales, label='Ventas Originales')
plt.plot(rolling_avg, label='Promedio Móvil (3 meses)', linestyle='--')
plt.title("Promedio Móvil")
plt.legend()
plt.show()


# **Promedio Móvil**
# **Ventajas: Muy simple, útil para suavizar ruido.
# 
# Desventajas: No predice hacia el futuro (solo suaviza lo pasado), y responde lento a cambios.
# 
# Cuándo usarlo: Si solo necesitas entender la tendencia general.
# 
# No es bueno para predicción real, solo para visualización de tendencia.**

# In[26]:


# --------------------------------------
# 2. SUAVIZAMIENTO EXPONENCIAL SIMPLE
# --------------------------------------
model_ses = SimpleExpSmoothing(monthly_sales).fit(smoothing_level=0.2, optimized=False)
forecast_ses = model_ses.fittedvalues

plt.figure(figsize=(12, 5))
plt.plot(monthly_sales, label='Ventas Originales')
plt.plot(forecast_ses, label='Suavizamiento Exponencial Simple', linestyle='--')
plt.title("Suavizamiento Exponencial Simple")
plt.legend()
plt.show()


# **Suavizamiento Exponencial Simple (SES)**
# Ventajas: Se adapta mejor que el promedio móvil.
# 
# Desventajas: No captura tendencias ni estacionalidad.
# 
# Cuándo usarlo: Si los datos son estables sin patrón estacional ni tendencia clara.
# 
# En este caso no sería el más adecuado, porque las ventas tienen una tendencia creciente y posible estacionalidad anual.

# In[27]:


# --------------------------------------
# 3. MODELO DE HOLT (con tendencia)
# --------------------------------------
model_holt = Holt(monthly_sales).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
forecast_holt = model_holt.fittedvalues

plt.figure(figsize=(12, 5))
plt.plot(monthly_sales, label='Ventas Originales')
plt.plot(forecast_holt, label='Holt (con tendencia)', linestyle='--')
plt.title("Modelo de Holt")
plt.legend()
plt.show()


# **Modelo de Holt (tendencia lineal)**
# Ventajas: Capta tendencia creciente o decreciente.
# 
# Desventajas: No capta estacionalidad.
# 
# Cuándo usarlo: Datos con tendencia pero sin patrón repetitivo estacional.
# 
# Funciona mejor que SES, pero no ideal si hay estacionalidad, como puede ser en ventas (ej: aumentan en fin de año).

# In[28]:


# --------------------------------------
# 4. HOLT-WINTERS (con estacionalidad)
# --------------------------------------
model_hw = ExponentialSmoothing(
    monthly_sales,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()
forecast_hw = model_hw.fittedvalues

plt.figure(figsize=(12, 5))
plt.plot(monthly_sales, label='Ventas Originales')
plt.plot(forecast_hw, label='Holt-Winters', linestyle='--')
plt.title("Modelo Holt-Winters")
plt.legend()
plt.show()


# **Holt-Winters (tendencia + estacionalidad)**
# Ventajas: Capta tendencia y patrones estacionales.
# 
# Desventajas: Puede sobreajustarse si no hay estacionalidad real.
# 
# Cuándo usarlo: Cuando hay estacionalidad clara (mensual, anual) y tendencia.
# 
# Es el modelo mas completo por que:
# 
# - Las ventas muestran una tendencia general creciente.
# 
# - Es probable que haya estacionalidad anual (ej: meses con más ventas).

# In[29]:


df['Order Date'] = pd.to_datetime(df['Order Date'])
monthly_sales = df.resample('M', on='Order Date')['Sales'].sum()

# === AJUSTE DEL MODELO HOLT-WINTERS ===
model_hw = ExponentialSmoothing(
    monthly_sales,
    trend='add',                # Captura tendencia
    seasonal='add',             # Captura estacionalidad
    seasonal_periods=12         # Ciclo anual (12 meses)
).fit()

# === PREDICCIÓN DE LOS PRÓXIMOS 12 MESES ===
forecast_12 = model_hw.forecast(12)

# === GRAFICAR RESULTADO ===
plt.figure(figsize=(14, 6))
plt.plot(monthly_sales, label='Ventas Históricas')
plt.plot(forecast_12, label='Predicción (12 meses)', linestyle='--', color='red')
plt.title("Predicción de Ventas - Holt-Winters")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === MOSTRAR VALORES PRONOSTICADOS ===
print("Predicción para los próximos 12 meses:")
print(forecast_12)


# In[30]:


# === DIVISIÓN ENTRENAMIENTO / VALIDACIÓN ===
train = monthly_sales[:-12]
test = monthly_sales[-12:]

# === FUNCIÓN PARA MÉTRICAS ===
def calcular_metricas(y_real, y_pred):
    mad = np.mean(np.abs(y_real - y_pred))
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    std = np.std(y_real - y_pred)
    return mad, mape, std

# === MODELO 1: PROMEDIO MÓVIL (3 MESES) ===
rolling_pred = train.rolling(window=3).mean().iloc[-1]
pred_pm = [rolling_pred] * 12  # mismo valor para cada predicción
mad_pm, mape_pm, std_pm = calcular_metricas(test, pred_pm)

# === MODELO 2: SUAVIZAMIENTO EXPONENCIAL SIMPLE ===
model_ses = SimpleExpSmoothing(train).fit(smoothing_level=0.2, optimized=False)
forecast_ses = model_ses.forecast(12)
mad_ses, mape_ses, std_ses = calcular_metricas(test, forecast_ses)

# === MODELO 3: HOLT ===
model_holt = Holt(train).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
forecast_holt = model_holt.forecast(12)
mad_holt, mape_holt, std_holt = calcular_metricas(test, forecast_holt)

# === MODELO 4: HOLT-WINTERS ===
model_hw = ExponentialSmoothing(
    train, trend='add', seasonal='add', seasonal_periods=12
).fit()
forecast_hw = model_hw.forecast(12)
mad_hw, mape_hw, std_hw = calcular_metricas(test, forecast_hw)

# === CONSTRUIR LA TABLA DE RESULTADOS ===
resultados = pd.DataFrame({
    'Método': ['Promedio Móvil', 'Suavización Expo', 'Holt', 'Winter'],
    'MAD': [mad_pm, mad_ses, mad_holt, mad_hw],
    'MAPE': [mape_pm, mape_ses, mape_holt, mape_hw],
    'Desv_Est': [std_pm, std_ses, std_holt, std_hw]
})

# Redondear y mostrar
resultados[['MAD', 'MAPE', 'Desv_Est']] = resultados[['MAD', 'MAPE', 'Desv_Est']].round(2)
print(resultados)

