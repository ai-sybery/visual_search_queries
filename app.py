import streamlit as st
st.set_page_config(page_title="Job Activity", layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1_gIvSsx9ueu1dWHtZnMTS9K5CqXYRJsz/export?format=xlsx"
    df = pd.read_excel(url)
    df['Период'] = pd.to_datetime(df['Период'])
    return df

def calculate_ema_with_season(data, periods=3, span=12):
    # Расчет EMA
    ema = data.ewm(span=span, adjust=False).mean()
    
    # Расчет сезонных коэффициентов
    seasonal_ratios = data / ema
    monthly_seasonality = {}
    
    # Группируем по месяцам и считаем средние коэффициенты
    for month in range(1, 13):
        month_ratios = seasonal_ratios[seasonal_ratios.index.month == month]
        monthly_seasonality[month] = month_ratios.mean()
    
    # Расчет прогноза
    last_ema = ema.iloc[-1]
    future_dates = pd.date_range(start=data.index[-1], periods=periods+1, freq='M')[1:]
    forecast_values = []
    
    for date in future_dates:
        month = date.month
        forecast_values.append(last_ema * monthly_seasonality[month])
    
    return pd.Series(forecast_values, index=future_dates), ema

def calculate_polynomial_with_season(data, periods=3, degree=4):
    # Создаем числовой индекс для регрессии
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values
    
    # Создаем полиномиальные признаки
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    # Обучаем модель
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Расчет исторических значений
    historical_values = model.predict(X_poly)
    
    # Расчет сезонных коэффициентов (как в EMA V1)
    ema = data.ewm(span=12, adjust=False).mean()
    seasonal_ratios = data / ema
    monthly_seasonality = {}
    
    for month in range(1, 13):
        month_ratios = seasonal_ratios[seasonal_ratios.index.month == month]
        monthly_seasonality[month] = month_ratios.mean()
    
    # Прогноз
    future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
    future_X_poly = poly_features.transform(future_X)
    base_predictions = model.predict(future_X_poly)
    
    future_dates = pd.date_range(start=data.index[-1], periods=periods+1, freq='M')[1:]
    forecast_values = []
    
    for date, base_pred in zip(future_dates, base_predictions):
        month = date.month
        forecast_values.append(base_pred * monthly_seasonality[month])
    
    return pd.Series(forecast_values, index=future_dates), pd.Series(historical_values, index=data.index)

def forecast_trends(data, periods=3, method='prophet'):
    # Расчет EMA
    _, ema_historical = calculate_ema_with_season(data)
    # Расчет Polynomial
    _, poly_historical = calculate_polynomial_with_season(data)

    if method == 'prophet':
        df_prophet = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        model = Prophet(yearly_seasonality=True)
        model.fit(df_prophet)
        future_dates = pd.date_range(start=data.index[-1], periods=periods+1, freq='M')[1:]
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        return forecast['yhat'].values, future_dates, ema_historical, poly_historical
    
    elif method == 'ema_season':
        forecast_values, _ = calculate_ema_with_season(data, periods)
        return forecast_values.values, forecast_values.index, ema_historical, poly_historical
    
    elif method == 'polynomial':
        forecast_values, _ = calculate_polynomial_with_season(data, periods)
        return forecast_values.values, forecast_values.index, ema_historical, poly_historical

try:
    st.title("Активность соискателей", anchor=False)
    
    df = load_data()
    professions = df.columns[1:].tolist()
    
    selected_prof = st.selectbox('Выберите профессию:', professions)
    
    forecast_method = st.radio(
        "Выберите метод прогнозирования:",
        ['Prophet', 'EMA with season', 'Polynomial with season'],
        horizontal=True
    )
    
    data = df.set_index('Период')[selected_prof]
    
    method = {
        'Prophet': 'prophet',
        'EMA with season': 'ema_season',
        'Polynomial with season': 'polynomial'
    }[forecast_method]
    
    forecast_values, future_dates, ema_historical, poly_historical = forecast_trends(data, method=method)
    
    fig = go.Figure()
    
    # Исторические данные
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data.values,
        name='Факт',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Исторические значения EMA
    fig.add_trace(go.Scatter(
        x=data.index,
        y=ema_historical,
        name='EMA',
        line=dict(color='#666666', width=1),
        visible='legendonly'
    ))
    
    # Исторические значения Polynomial
    fig.add_trace(go.Scatter(
        x=data.index,
        y=poly_historical,
        name='Polynomial',
        line=dict(color='#b19cd9', width=1),
        visible='legendonly'
    ))
    
    # Соединительная линия
    fig.add_trace(go.Scatter(
        x=[data.index[-1], future_dates[0]],
        y=[data.iloc[-1], forecast_values[0]],
        name='Соединение',
        line=dict(color='#ff7f0e', width=1, dash='dot'),
        showlegend=False
    ))
    
    # Прогноз
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_values,
        name='Прогноз активности',
        line=dict(color='#ff7f0e', width=1),
        mode='lines+markers',
        marker=dict(size=6, color='#ff7f0e')
    ))
    
    fig.update_layout(
        title={
            'text': f'Прогноз активности: {selected_prof}',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title='Период',
        yaxis_title='Кол-во запросов в месяц',
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=14),
        xaxis=dict(gridcolor='#E5E5E5'),
        yaxis=dict(gridcolor='#E5E5E5')
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Таблица с прогнозом
    last_actual = data.iloc[-1]
    changes = [(v/last_actual - 1)*100 for v in forecast_values]
    
    forecast_df = pd.DataFrame({
        'Месяц': future_dates.strftime('%B %Y'),
        'Прогноз активности по отношению к текущему месяцу, %': [f"{x:.1f}%" for x in changes]
    })

    def color_negative_red(val):
        try:
            num = float(val.strip('%'))
            color = '#ff4b4b' if num < 0 else '#28a745'
            return f'color: {color}'
        except:
            return ''

    st.dataframe(
        forecast_df.style.applymap(
            color_negative_red, 
            subset=['Прогноз активности по отношению к текущему месяцу, %']
        ),
        use_container_width=True,
        hide_index=True
    )

except Exception as e:
    st.error(f"Ошибка: {str(e)}")
