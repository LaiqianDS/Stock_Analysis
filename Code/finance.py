# API Yahoo
import yfinance as yf

# Scrapping
import requests
from bs4 import BeautifulSoup

# Data Viz
import matplotlib.pyplot as plt
import mplfinance as mpf
from IPython.display import display
import seaborn as sns

# Funcionalidades
#timedelta es para diferencias de tiempo, date es una clase
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import csv
from ipywidgets import widgets, interact
from scipy.stats import ttest_ind # comparacion de medias


###########################
# Definición de funciones #
###########################

###########
# General #
###########

def get_prices(ticker, start=date.today()-timedelta(days=365), end=date.today()):
    """
    Dado un ticker, carga los datos de cotizaciones entre start_date y end_date
    Formato de fecha: YYYY-MM-DD
    Si no se especifica end_date, se inicializa como hoy por defecto.
    Es importante decir que el primer dato puede ser POSTERIOR al dado, pues puede haber empezado a cotizar después de la fecha proporcionada    
    """
    return yf.download(ticker, start=start, end=end, progress=False).dropna()

def get_prices_detailed(ticker, period='1y'):
    """
    Función similar a get_prices, pero utilizando la función yf.Ticker.history()
    Algunos ejemplos de period es 1y, 1w, 1mo, 1m
    """
    return yf.Ticker(ticker).history(period=period)

def get_info(ticker):
    """
    Dado un ticker, carga datos de la empresa.
    """
    return yf.Ticker(ticker).info

def repr_info(info, ticker, description=False):
    """
    Recibe un diccionario resultante de llamar a la función get_info() y  muestra por pantalla los datos más relevantes
    """
    # Datos más relevantes a elleción del programador
    info_d = {
        'general': ['country', 'website', 'industry', 'sector', 'fullTimeEmployees', 'dividendYield', 'beta', 'marketCap',  'currency', 'enterpriseValue', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh'],
        'financials': ['trailingPE', 'forwardPE', 'trailingEps', 'forwardEps', 'targetHighPrice', 'targetLowPrice', 'freeCashflow', 'operatingCashflow', 'earningsGrowth', 'revenueGrowth'],
        'margins': ['profitMargins', 'grossMargins', 'ebitdaMargins', 'operatingMargins'],
        'returns': ['returnOnAssets', 'returnOnEquity']
    }

    # Imprimo toda la info
    print(f'Información de la empresa: {ticker.upper()}')
    
    if description:
        print(f'Resumen: {info["longBusinessSummary"]}')

    not_seen = []
        
    for section in info_d:
        print('#'*40)
        print(f'\t\t-- {section.upper()} --')
        for data in info_d[section]:
            try:
                print(f'{data}: {info[data]}')
            except:
                not_seen.append(data)
        if section == 'general':
            print(f'1 Year Change %: {pct_change(ticker)}')
            print(f'5 Years CAGR: {cagr(ticker, start_date=date.today()-timedelta(days=365*5))}')
        print('\n')

    # Printeo si ha habido algún dato que no ha podido ser mostrado
    if len(not_seen) > 0: print(f'Los siguientes datos no se pudieron mostrar: {not_seen}')

def load_repr_info(ticker, desc=False):
    """
    Función auxiliar que llama a get_info y repr_info.
    Pide el ticker de la empresa y el booleano para la descripción
    IMPORTANTE: No acepta más de un ticker
    """
    repr_info(get_info(ticker), ticker, desc)

############
# Cálculos #
############
    
def pct_change(ticker, start_date=date.today()-timedelta(days=365), end_date=date.today()):
    """
    Dado un ticker, calcula el retorno si hubieramos invertido en la apertura del mercado
    del día start_date hasta el cierre del día actual.
    Formato de fecha: YYYY-MM-DD
    """
    
    # Descargar datos históricos desde start_date hasta end_date
    data = get_prices(ticker, start=start_date, end=end_date)
    
    initial_price = data['Open'].iloc[0]  # Precio de apertura en start_date
    last_price = data['Close'].iloc[-1]  # Precio de cierre actual
    percent_change = ((last_price - initial_price) / initial_price)
    
    return round(percent_change, 4)

def pct_change_from(data):
    """
    Dados unos datos precargados, calcula el retorno si hubieramos invertido en la apertura del mercado
    del primer día en los datos hasta el cierre del último día.
    Formato de fecha: YYYY-MM-DD
    """
    
    initial_price = data['Open'].iloc[0]  # Precio de apertura en start_date
    last_price = data['Close'].iloc[-1]  # Precio de cierre actual
    percent_change = ((last_price - initial_price) / initial_price)
    
    return round(percent_change, 4)

def cagr(ticker, start_date=None, end_date=None):
    """
    Obtiene el CAGR (Compound Annual Growth Rate) de una empresa.
    Para ello, toma los años enteros de diferencia desde el start_date hasta end_date, redondeando hacia abajo.
    Si no se especifica año, solo obtiene el pct de cambio de un año
    Formato de fecha: YYYY-MM-DD
    """
    if start_date is None:
        start_date = str(date.today() - timedelta(days=365))
    if end_date is None:
        end_date = str(date.today())
    
    # Normalizar el formato de las fechas si son cadenas
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date() if len(start_date) == 10 else datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date() if len(end_date) == 10 else datetime.strptime(end_date, "%Y-%m-%d")

    # Descargar datos históricos desde start_date hasta end_date
    data = get_prices(ticker, start=str(start_date), end=str(end_date))

    anyos = int(((end_date - start_date).days / 365))  # Años de diferencia, redondeando hacia abajo
    initial_price = data['Open'].iloc[0]  # Precio de apertura en start_date
    last_price = data['Close'].iloc[-1]  # Precio de cierre actual
    
    cagr = ((last_price / initial_price) ** (1 / anyos) - 1)
    return round(cagr, 4)

def cagr_from(data):
    """
    Obtiene el CAGR dado un dataset de cotizaciones
    Para ello, toma los años enteros de diferencia redondeando hacia abajo.
    """
    start_date = data.index[0]
    end_date = data.index[-1]
    anyos = int(((end_date-start_date).days/365)) # Anyos de diferencia, redondeando hacia abajo

    if anyos == 0:
        raise ZeroDivisionError('El dataset contiene menos de un año de datos, necesitas un mínimo de un 1 año')

    initial_price = data['Open'].iloc[0]  # Precio de apertura en start_date
    last_price = data['Close'].iloc[-1]  # Precio de cierre actual
    cagr = ((last_price / initial_price) ** (1 / anyos) - 1)

    return round(cagr, 4)

def relative_strength(prices, n=14):
    """
    Función que calcula el índice relativo de fuerza o RSI.
    Se ha obtenido ayuda bibliográfica para su diseño. Consulta: 
    https://github.com/matplotlib/mplfinance/blob/master/examples/indicators/mpf_rsi_demo.py
    """
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def bbands(data):
    """
    Las bandas de bollinger son un indicador técnico compuesto de tres bandas: superior, inferior e intermedia.
    Se basan en la media y la desviación típica, y el Teorema de Chebyshev.
    Upper Band = MA + K ⋅ σ
    Middle Band = MA
    Lower Band = MA − K ⋅ σ
    Donde:
    MA es una media movil (generalmente de longitud 20)
    K es el número de desviaciones estándares (typically set to 2)
    σ es la desviación estándar
    Esta función devuelve un dataframe con las bandas (sup. e inf.) y el nivel porcentaje de la cotización dentro de la banda (0% es la media móvil).
    """
    # Calcular la media móvil
    data['SMA'] = data['Close'].rolling(window=20).mean()

    # Calcular la desviación estándar
    data['std_dev'] = data['Close'].rolling(window=20).std()

    # Calcular las bandas superior e inferior de Bollinger
    data['Banda_Superior'] = data['SMA'] + (data['std_dev'] * 2)
    data['Banda_Inferior'] = data['SMA'] - (data['std_dev'] * 2)

    # Calcular el nivel porcentaje 
    data['percentB'] = (data['Close'] - data['Banda_Inferior']) / (data['Banda_Superior'] - data['Banda_Inferior'])
    return data


#################
# Visualización #
#################

def add_rsi(data):
    """
    Crea el objeto make_addplot para añadir el rsi al gráfico
    """
    data['rsi'] = relative_strength(data['Close'],n=7)    
    apd = [mpf.make_addplot(data['rsi'], panel=1, color='lime',ylim=(10,90),secondary_y=True)]
    return apd

def add_macd(data):
    """
    Función auxiliar para añadir el macd en el gráfico de la función plot().
    Recibe el dataframe de las cotizaciones.
    Las medias utilizadas en el macd o Moving Average Convergence Divergence son de longitud 9, 12 y 26.
    En el MACD: 
    Si la línea MACD está por encima de cero, tendencia bullish
    Si la línea MACD está por debajo de cero, tendencia bearish
    La estrategia básica suele combinar histograma y MACD, tal que, si el MACD sobresale del histograma:
    Por debajo del nivel cero, oportunidad de compra
    Por encima del nivel cero, oportunidad de venta
    """

    # Calcular el indicador MACD y el histograma
    exp12 = data['Close'].ewm(span=12, min_periods=0, adjust=False).mean() #ewm es media exponencial
    exp26 = data['Close'].ewm(span=26, min_periods=0, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, min_periods=0, adjust=False).mean()
    histogram = macd-signal

    # Añadimos los addplots
    apds = [mpf.make_addplot(histogram,type='bar',width=0.7,panel=1,
                             color='dimgray',alpha=1,secondary_y=False),
            mpf.make_addplot(signal,panel=1,color='orange',secondary_y=True)
           ]
    return apds

def add_bbands(data):
    """
    Crea el objeto make_addplot para añadir las bandas de bollinger.
    Recibe un dataset de cotizaciones
    """
    new_data = bbands(data)
    ap = [
          mpf.make_addplot(data['Banda_Inferior'], color='red'),
          mpf.make_addplot(data['Banda_Superior'], color='green'),
          mpf.make_addplot(data['SMA'], color='gray')
         ]
    return ap

def plot(ticker, start_date=str(date.today()-timedelta(days=365)), end_date=str(date.today()), volume=False, 
         style='yahoo', title='', ylabel='', ylabel_lower='', savefig=None, type='candle', sma=(), info=False, macd=False, rsi=False, bollinger=False):
    """
    Dada una fecha de inicio y una fecha de fin, realiza el gráfico de la cotización dentro de ese período.
    Si no hay fecha de fin, toma el último día de cotización. 
    Otros atributos:
    - volume: boolean
    - style= consult mpf.available_styles(), 'yahoo' is at default,
    - title= str,
    - ylabel= str,
    - ylabel_lower=str, 
    - savefig=str,
    - type = lines/candles
    - sma = tuple
    - info= boolean, muestra por pantalla información general de la empresa
    - macd= boolean, muestra el MACD
    - rsi = boolean, muestra el RSI.
    - bollinger = boolean, muestra las bandas de bollinger
    NOTA SOBRE INDICADORES: Por convención, la presencia de un indicador excluirá los demás, puedes plotear varios llamando varias veces al método plot
    Orden: macd > rsi 
    Formato de fecha: YYYY-MM-DD
    """
    # Obtener los datos de cotización utilizando yfinance
    data = get_prices(ticker, start=start_date, end=end_date)

    # Normalización
    if title=='': title = f'Cotización de {ticker}'
    if ylabel=='': ylabel='Price'
    if ylabel_lower=='': ylabel_lower='Volume'

    # Adición de ténicos y graficación
    if macd: 
        ap = add_macd(data)
        if bollinger: ap.extend(add_bbands(data))
        if savefig is None:
            mpf.plot(data, type=type, addplot=ap, volume=volume, volume_panel=2, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, mav=(sma))
        else:
            mpf.plot(data, type=type, addplot=ap, volume=volume, volume_panel=2, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, savefig=savefig+'.png', mav=(sma))
    elif rsi:
        ap = add_rsi(data)
        if bollinger: ap.extend(add_bbands(data))
        if savefig is None:
            mpf.plot(data, type=type, addplot=ap, volume=volume, volume_panel=2, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, mav=(sma))
        else:
            mpf.plot(data, type=type, addplot=ap, volume=volume, volume_panel=2, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, savefig=savefig+'.png', mav=(sma))
    else:
        if bollinger: 
            ap = add_bbands(data)
            if savefig is None:
                mpf.plot(data, type=type, volume=volume, addplot=ap, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, mav=(sma))
            else:
                mpf.plot(data, type=type, volume=volume, addplot=ap, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, savefig=savefig+'.png', mav=(sma))
        else:
            if savefig is None:
                mpf.plot(data, type=type, volume=volume, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, mav=(sma))
            else:
                mpf.plot(data, type=type, volume=volume, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, savefig=savefig+'.png', mav=(sma))
    if info:
        load_repr_info(ticker)

def plot_from(data, style='yahoo', title='', ylabel='', ylabel_lower='', savefig=None, type='candle', volume=False, sma=(), macd=False, rsi=False, bollinger=False):
    """
    Dado un dataset de cotizaciones, realiza un gráfico de su cotización.
    Atributos:
    - volume: boolean
    - style= consult mpf.available_styles(), 'yahoo' is at default,
    - title= str,
    - ylabel= str,
    - ylabel_lower=str, 
    - savefig=str,
    - type = lines/candles
    - sma = tuple or list
    - macd= boolean, muestra el MACD
    - rsi = boolean, muestra el RSI
    - bollinger = boolean, muestra las bandas de bollinger
    NOTA SOBRE INDICADORES: Por convención, la presencia de un indicador excluirá los demás, puedes plotear varios llamando varias veces al método plot
    Orden: macd > rsi 
    """
    # Normalización
    if title=='': title = f'Cotización del dataset'
    if ylabel=='': ylabel='Price'
    if ylabel_lower=='': ylabel_lower='Volume'

    # Adición de ténicos y graficación
    if macd: 
        ap = add_macd(data)
        if bollinger: ap.extend(add_bbands(data))
        if savefig is None:
            mpf.plot(data, type=type, addplot=ap, volume=volume, volume_panel=2, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, mav=(sma))
        else:
            mpf.plot(data, type=type, addplot=ap, volume=volume, volume_panel=2, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, savefig=savefig+'.png', mav=(sma))
    elif rsi:
        ap = add_rsi(data)
        if bollinger: ap.extend(add_bbands(data))
        if savefig is None:
            mpf.plot(data, type=type, addplot=ap, volume=volume, volume_panel=2, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, mav=(sma))
        else:
            mpf.plot(data, type=type, addplot=ap, volume=volume, volume_panel=2, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, savefig=savefig+'.png', mav=(sma))
    else:
        if bollinger: 
            ap = add_bbands(data)
            if savefig is None:
                mpf.plot(data, type=type, volume=volume, addplot=ap, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, mav=(sma))
            else:
                mpf.plot(data, type=type, volume=volume, addplot=ap, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, savefig=savefig+'.png', mav=(sma))
        else:
            if savefig is None:
                mpf.plot(data, type=type, volume=volume, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, mav=(sma))
            else:
                mpf.plot(data, type=type, volume=volume, style=style, title=title, ylabel=ylabel, ylabel_lower=ylabel_lower, savefig=savefig+'.png', mav=(sma))

def correlation(dataframes):
    """
    Recibe una lista de dataframes y devuelve las correlaciones entre pares de activos
    """
    nombres_dataframes = [f'DF{i+1}' for i in range(len(dataframes))]

    correlaciones = pd.DataFrame(index=nombres_dataframes, columns=nombres_dataframes)

    for i, df1 in enumerate(dataframes):
        for j, df2 in enumerate(dataframes):
            correlacion = df1['Open'].corr(df2['Open'])
            correlaciones.iloc[i, j] = correlacion

    # Graficar el mapa de calor de las correlaciones
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlaciones.astype(float), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlación entre activos')
    plt.xlabel('DataFrames')
    plt.ylabel('DataFrames')
    plt.show()
    
def compare(names):
    """
    Recibe una lista con los datos (df) de una o varias empresas con el mismo eje temporal y realiza una gráfica de lineas con todas
    EL parámetro deben ser lista
    """
    # Genera los dataframes
    if isinstance(names, list):
        dataframes = [get_prices_detailed(name) for name in names]
    
    # Iterar sobre la lista de DataFrames y generar un nuevo dataframe
    new_df = pd.DataFrame()
    for i in range(len(names)):
        new_df[names[i]] = dataframes[i]['Close']

    added_plots = mpf.make_addplot(new_df[names[1:]], secondary_y=False)

    """
    Ejemplo de addplots (guía oficial)
    tcdf = df[['LowerB','UpperB']]  # DataFrame with two columns
    apd  = mpf.make_addplot(tcdf)
    mpf.plot(df,addplot=apd)
    """

    # Configurar el gráfico principal con el primer DataFrame
    primer_df = dataframes[0]

    # Crear el gráfico con mplfinance y añadir los addplots
    fig, axes = mpf.plot(primer_df, addplot=added_plots, type='line', volume=False,
                         ylabel='Precio', title='Cotizaciones de Empresas', returnfig=True)
    axes[0].legend(names)    

    correlation(dataframes)

#########################
# Impacto de resultados #
#########################
    
def scrap_results(ticker):
    """
    Recibe un ticker y, utilizando beautifulSoup, obtiene los resultados de las 4 últimas presentaciones y devuelve
    una lista de tuplas (diferencia, % diferencia) y una lista con las fechas con formato MM/DD/YYYY. 
    """
    # Establecemos una sesión
    headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"}
    
    # Obtenemos los datos
    src = requests.get(f"https://finance.yahoo.com/quote/{ticker}/analysis?p={ticker}", headers=headers)
    if src.status_code == 200:
        soup = BeautifulSoup(src.content, 'html.parser')
        earning_history = soup.find_all("table", class_="W(100%) M(0) BdB Bdc($seperatorColor) Mb(25px)")[2] # 3a tabla de la web
        dates = earning_history.find_all('th', class_="Fw(400) W(20%) Fz(xs) C($tertiaryColor) Ta(end)") # table headers con las fechas
        l_dates = [date.renderContents().decode()[6:-7] for date in dates] # coge el texto, decodifica y guarda quitando texto sobrante
        data = earning_history.find_all('td', class_="Ta(end)") # coge todos los datos de la tabla
        l_data = list(zip([d.renderContents().decode() for d in data[-8:-4:]], [d.renderContents().decode() for d in data[-4:]])) # lista de tuplas con los que queremos
        
        return l_data, l_dates
    else: raise ValueError('Couldn\'t get stock info')

def results_to_dataframe(ticker):
    """
    Recibe un ticker y carga los datos de los últimos 4 anuncios de resultados llamando a la función
    scrap_results(). Posteriormente, convierte las listas recibidas en un Dataframe.
    Dicho dataframe tiene fechas como índice y las columnas Difference y Difference %
    """
    data, dates = scrap_results(ticker) # Carga los datos

    # Creamos el dataframe
    dic = {'Date': dates, 'Difference': [d[0] for d in data], 'Difference %': [d[1] for d in data]} 
    res = pd.DataFrame(dic)

    # Ajustamos y modificamos el índice
    res['Date'] = pd.to_datetime(res['Date'])
    res = res.set_index('Date')
    
    return res

def results_impact(ticker):
    """
    Función que hace un wrap de scrap_results y results_to_dataframe
    Posteriormente, realiza la prueba t de Student de comparación de medias
    para medir si el impacto de los resultados ha sido significativo
    tal que las medias son diferentes en el período previo frente al posterior
    El resultado es un diccionario con las fechas de los resultados como claves
    y un booleano indicando diferencia de medias como valor
    """
    results = results_to_dataframe(ticker)
    
    # Normalizo las fechas del DataFrame results
    results.index = pd.to_datetime(results.index, format = '%m/%d/%Y').strftime('%Y-%m-%d')
    impact = {}
    
    for fecha in results.index:
        # Obtener datos históricos de precios de una acción (por ejemplo, Apple)
        window_size = 10  # Número de días antes y después del evento

        # Convertir la fecha del evento a timestamp
        fecha = pd.Timestamp(fecha)

        # Crear un rango de fechas hábiles que incluya la ventana de tiempo, pues hay festivos, etc.
        # freq = 'B' es para días laborales
        rango_previo = pd.date_range(end=fecha, periods=window_size, freq='B')  # Para días antes del evento
        rango_posterior = pd.date_range(start=fecha, periods=window_size + 1, freq='B')  # Para días después del evento

        # Obtener los precios de acciones para el rango de fechas ajustado a días hábiles
        pre = get_prices(ticker, start=rango_previo.min(), end=rango_previo.max())
        post = get_prices(ticker, start=rango_posterior.min(), end=rango_posterior.max())
        
        # Realizar la prueba t de Student de comparacion de medias
        t_stat, p_value = ttest_ind(pre['Close'], post['Close'])
        impact[str(fecha)] = p_value < 0.05
    return impact