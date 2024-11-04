# app.py
import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# Конфигурация API
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-exp-0827")
except Exception as e:
    st.error("Ошибка инициализации API Gemini. Проверьте ключ.")

# Конфигурация страницы
st.set_page_config(page_title="Job Trends", layout="wide")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_excel('data/job_trends.xlsx')
    df['Период'] = pd.to_datetime(df['Период'], format='%B %Y')
    return df

# Прогнозирование
def forecast_trends(data, periods=3):
    df_prophet = pd.DataFrame({
        'ds': data.index,
        'y': data.values
    })
    model = Prophet(yearly_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast

# Основной код
try:
    st.title("Тренды поисковых запросов")
    
    df = load_data()
    professions = df.columns[1:].tolist()
    
    selected_prof = st.selectbox('Выберите профессию:', professions)
    
    # Подготовка данных
    data = df.set_index('Период')[selected_prof]
    
    # Прогноз
    forecast = forecast_trends(data)
    
    # График
    fig = go.Figure()
    
    # Исторические данные
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data.values,
        name='Факт',
        line=dict(color='blue')
    ))
    
    # Прогноз
    future_dates = pd.date_range(
        start=data.index[-1],
        periods=4,
        freq='M'
    )[1:]
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast['yhat'].tail(3),
        name='Прогноз',
        line=dict(color='red', dash='dash')
    ))
    
    # Доверительный интервал
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast['yhat_upper'].tail(3),
        fill=None,
        mode='lines',
        line=dict(color='rgba(255,0,0,0)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast['yhat_lower'].tail(3),
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(255,0,0,0)'),
        name='Доверительный интервал'
    ))
    
    fig.update_layout(
        title=f'Тренд и прогноз: {selected_prof}',
        xaxis_title='Период',
        yaxis_title='Доля запросов (%)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Ошибка: {str(e)}")
