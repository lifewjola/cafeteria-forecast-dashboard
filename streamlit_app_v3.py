import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os
import requests
from io import BytesIO
import numpy as np
import random
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
import subprocess
import sys
import time

st.set_page_config(
    page_title="Babcock University Cafeteria Forecast", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="image.png"
)

BABCOCK_BLUE = "#003087"
BABCOCK_GOLD = "#FFD700"  
BABCOCK_LIGHT_BLUE = "#E6EEF8" 
BABCOCK_DARK_BLUE = "#001F54" 

st.markdown("""
<style>
    .main-header {
        color: #003087;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 0.5rem 0;
        border-bottom: 2px solid #FFD700;
        margin-bottom: 1.5rem;
    }
    
    .subheader {
        color: #003087;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 0.3rem 0;
        border-left: 4px solid #FFD700;
        padding-left: 10px;
        margin: 1.5rem 0 1rem 0;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-top: 3px solid #003087;
        margin-bottom: 1rem;
    }
    
    .popular-meal {
        border-top: 3px solid #FFD700;
    }
    
    .css-1d391kg, .css-12oz5g7 {
        background-color: #E6EEF8;
    }
    
    .stButton>button {
        background-color: #003087;
        color: white;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #001F54;
        color: #FFD700;

    }
    
    .info-box {
        background-color: #E6EEF8;
        border-left: 4px solid #003087;
        padding: 1rem;
        border-radius: 0 5px 5px 0;
    }
    
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        font-size: 0.8rem;
        color: #666;
    }
    
    .positive-delta {
        color: green;
    }
    
    .negative-delta {
        color: red;
    }
    
    .number-input-container {
        background-color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    
    .update-button {
        background-color: #003087;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .update-button:hover {
        background-color: #001F54;
        color: white;

    }
    
    .stProgress > div > div {
        background-color: #FFD700;
    }
</style>
""", unsafe_allow_html=True)


meal_options = {
    "Monday": {
        "Breakfast": ["Akara & Pap", "Bread & Tea", "Yam & Egg"],
        "Lunch": ["Jollof Rice", "Beans & Plantain", "Spaghetti & Sauce"],
        "Dinner": ["Eba & Egusi", "Rice & Stew", "Yam Porridge"]
    },
    "Tuesday": {
        "Breakfast": ["Moi Moi & Custard", "Bread & Akamu", "Boiled Yam & Sauce"],
        "Lunch": ["Fried Rice", "White Rice & Stew", "Beans & Garri"],
        "Dinner": ["Amala & Ewedu", "Noodles & Egg", "Eba & Ogbono"]
    },
    "Wednesday": {
        "Breakfast": ["Akara & Custard", "Bread & Tea", "Sweet Potato & Egg"],
        "Lunch": ["Jollof Spaghetti", "Rice & Beans", "Okra Soup & Eba"],
        "Dinner": ["Fufu & Egusi", "Indomie Special", "Yam Porridge"]
    },
    "Thursday": {
        "Breakfast": ["Boiled Plantain & Egg", "Bread & Akamu", "Moi Moi & Pap"],
        "Lunch": ["Ofada Rice", "White Rice & Banga", "Beans & Plantain"],
        "Dinner": ["Pounded Yam & Ogbono", "Jollof Rice", "Stir Fry Noodles"]
    },
    "Friday": {
        "Breakfast": ["Custard & Akara", "Tea & Bread", "Fried Yam & Egg"],
        "Lunch": ["Fried Rice", "Jollof Rice", "Okra & Eba"],
        "Dinner": ["Eba & Egusi", "Rice & Beans", "Spaghetti & Sauce"]
    }
}

def generate_data(num_students=200, num_weeks=4):
    # Constants
    students = [f"Student_{i}" for i in range(1, num_students + 1)]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    meals = ["Breakfast", "Lunch", "Dinner"]
    
    def generate_meal_attendance():
        choices = [1, 0] 
        weights = [0.65, 0.35]  
        return random.choices(choices, weights=weights, k=3)  
    
    historical_data = []
    
    for week in range(num_weeks):
        for day in days:
            for student in students:
                attendance_pattern = generate_meal_attendance() 
                for i, meal in enumerate(meals):
                    if attendance_pattern[i]:
                        chosen_meal = random.choice(meal_options[day][meal])
                    else:
                        chosen_meal = 'None'  
                    historical_data.append({
                        "Week": week + 1,
                        "Day": day,
                        "Student": student,
                        "Meal": meal,
                        "Meal_Option": chosen_meal
                    })
    
    historical_df = pd.DataFrame(historical_data)
    
    historical_df.to_csv("cafeteria_data.csv", index=False)
    
    return historical_df

def add_new_week_data(num_students=200):
    try:
        existing_df = pd.read_csv("cafeteria_data.csv")
        
        last_week = existing_df['Week'].max()
        new_week = last_week + 1
        
        students = [f"Student_{i}" for i in range(1, num_students + 1)]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        meals = ["Breakfast", "Lunch", "Dinner"]
        
        def generate_meal_attendance():
            choices = [1, 0]  
            weights = [0.65, 0.35]  
            return random.choices(choices, weights=weights, k=3) 
        
        new_week_data = []
        
        for day in days:
            for student in students:
                attendance_pattern = generate_meal_attendance()
                for i, meal in enumerate(meals):
                    if attendance_pattern[i]:
                        chosen_meal = random.choice(meal_options[day][meal])
                    else:
                        chosen_meal = 'None'
                    new_week_data.append({
                        "Week": new_week,
                        "Day": day,
                        "Student": student,
                        "Meal": meal,
                        "Meal_Option": chosen_meal
                    })
        
        new_week_df = pd.DataFrame(new_week_data)
        
        combined_df = pd.concat([existing_df, new_week_df], ignore_index=True)
        
        combined_df.to_csv("cafeteria_data.csv", index=False)
        
        return combined_df, new_week
    
    except FileNotFoundError:
        historical_df = generate_data()
        return historical_df, 4  

def generate_forecast(next_week=None):
    df = pd.read_csv("cafeteria_data.csv")
    
    if next_week is None:
        next_week = df['Week'].max() + 1
    
    meal_counts = df.groupby(['Week', 'Day', 'Meal', 'Meal_Option']).size().reset_index(name='Count')
    
    X_raw = meal_counts[['Week', 'Day', 'Meal', 'Meal_Option']]
    y = meal_counts['Count']
    
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_cat = encoder.fit_transform(X_raw[['Day', 'Meal', 'Meal_Option']])
    X_num = meal_counts[['Week']].values
    X_final = np.hstack([X_num, X_cat])
    
    X_final = sm.add_constant(X_final)
    poisson_model = sm.GLM(y, X_final, family=sm.families.Poisson())
    poisson_results = poisson_model.fit()
    
    valid_meal_options = []
    for day, meals_dict in meal_options.items():
        for meal, options in meals_dict.items():
            for option in options:
                valid_meal_options.append((day, meal, option))
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    meals = ["Breakfast", "Lunch", "Dinner"]
    
    pred_input = []
    for day in days:
        for meal in meals:
            for option in meal_options[day][meal]:
                if (day, meal, option) in valid_meal_options:
                    pred_input.append((next_week, day, meal, option))
    
    pred_df = pd.DataFrame(pred_input, columns=['Week', 'Day', 'Meal', 'Meal_Option'])
    
    X_pred_cat = encoder.transform(pred_df[['Day', 'Meal', 'Meal_Option']])
    X_pred_num = pred_df[['Week']].values
    X_pred_final = sm.add_constant(np.hstack([X_pred_num, X_pred_cat]), has_constant='add')
    
    pred_counts = poisson_results.predict(X_pred_final)
    pred_df['Predicted_Count'] = np.round(pred_counts).astype(int)
    
    pred_df.to_csv("forecast_output.csv", index=False)
    
    return pred_df

@st.cache_data
def load_data():
    try:
        forecast_df = pd.read_csv("forecast_output.csv")
        try:
            cafeteria_df = pd.read_csv("cafeteria_data.csv")
            return forecast_df, cafeteria_df
        except:
            return forecast_df, None
    except:
        cafeteria_df = generate_data()
        forecast_df = generate_forecast()
        return forecast_df, cafeteria_df

def update_data_and_forecasts():
    with st.spinner("Updating data and generating new forecasts..."):
        progress_bar = st.progress(0)
        
        cafeteria_df, last_week = add_new_week_data()
        progress_bar.progress(50)
        time.sleep(0.5)  
        
        forecast_df = generate_forecast(last_week + 1)
        progress_bar.progress(100)
        time.sleep(0.5)  
        
        st.cache_data.clear()
        
        st.success(f"Added Week {last_week} data and generated forecasts for Week {last_week + 1}!")
        return True


df, cafeteria_df = load_data()


logo = Image.open("babcock-logo.png")
st.sidebar.image(logo, width=200)
   
st.sidebar.markdown("<h2 style='color:#003087;'>Dashboard Controls</h2>", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='margin: 0.5rem 0; border-color: #FFD700;'>", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='color:#003087; font-size: 1.2rem;'>Meal Selection</h3>", unsafe_allow_html=True)
day = st.sidebar.selectbox("Select Day", sorted(df['Day'].unique()))
meal = st.sidebar.selectbox("Select Meal", sorted(df['Meal'].unique()))

st.sidebar.markdown("<h3 style='color:#003087; font-size: 1.2rem;'>School Population</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='number-input-container'>", unsafe_allow_html=True)
students_in_school = st.sidebar.number_input("Number of Students in School", min_value=1, value=300)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='color:#003087; font-size: 1.2rem;'>Data Management</h3>", unsafe_allow_html=True)
if st.sidebar.button("Update Records & Forecasts", key="update_button"):
    update_success = update_data_and_forecasts()
    if update_success:
        st.rerun(scope="app")

st.sidebar.markdown("<hr style='margin: 1.5rem 0; border-color: #FFD700;'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='background-color: #E6EEF8; padding: 1rem; border-radius: 5px; border-left: 4px solid #003087;'>
    <h4 style='color: #003087; margin-top: 0;'>About This Dashboard</h4>
    <p style='font-size: 0.9rem;'>This dashboard helps Babcock University's cafeteria staff plan meal preparation by forecasting student turnout for each meal option.</p>
</div>
""", unsafe_allow_html=True)

week_now = df['Week'].max()  
previous_week = week_now - 1  

df_week_now = df[(df['Week'] == week_now) & (df['Day'] == day) & (df['Meal'] == meal)]

if cafeteria_df is not None:
    actual_counts_all_weeks = cafeteria_df.groupby(['Week', 'Day', 'Meal', 'Meal_Option']).size().reset_index(name='Actual_Count')
    
    actual_counts_prev_week = actual_counts_all_weeks[
        (actual_counts_all_weeks['Week'] == previous_week) & 
        (actual_counts_all_weeks['Day'] == day) & 
        (actual_counts_all_weeks['Meal'] == meal)
    ].copy()  
    
    if not actual_counts_prev_week.empty:
        total_students_prev_week = actual_counts_prev_week['Actual_Count'].sum()
        if total_students_prev_week > 0:  
            actual_counts_prev_week.loc[:, 'Scaled_Actual'] = (
                actual_counts_prev_week['Actual_Count'] / total_students_prev_week * students_in_school
            ).round().astype(int)
        else:
            actual_counts_prev_week.loc[:, 'Scaled_Actual'] = 0
else:
    actual_counts_all_weeks = pd.DataFrame(columns=['Week', 'Day', 'Meal', 'Meal_Option', 'Actual_Count'])
    actual_counts_prev_week = pd.DataFrame(columns=['Week', 'Day', 'Meal', 'Meal_Option', 'Actual_Count', 'Scaled_Actual'])

total_predicted = df_week_now['Predicted_Count'].sum()
df_week_now = df_week_now.copy()  
if total_predicted > 0:
    df_week_now.loc[:, 'Scaled_Count'] = (df_week_now['Predicted_Count'] / total_predicted * students_in_school).round().astype(int)
else:
    df_week_now.loc[:, 'Scaled_Count'] = 0

df_merged = pd.merge(
    df_week_now,
    actual_counts_prev_week[['Meal_Option', 'Scaled_Actual']],
    on='Meal_Option', 
    how='left'
)

df_merged['Scaled_Actual'] = df_merged['Scaled_Actual'].fillna(0).astype(int)

df_merged['Delta'] = df_merged['Scaled_Count'] - df_merged['Scaled_Actual']

max_count = df_merged['Scaled_Count'].max()

st.markdown("<h1 class='main-header'>🍛 Babcock University Cafeteria Meal Forecast</h1>", unsafe_allow_html=True)

st.markdown(f"<h2 class='subheader'>Forecast for {meal} on {day} (Week {week_now})</h2>", unsafe_allow_html=True)

cols = st.columns(len(df_merged))

sample_images = {}
    
def get_image(meal_option):
    
    img_dir = "images"
    if os.path.exists(img_dir):
        possible_filenames = [
            f"{meal_option.replace(' & ', '-').replace(' ', '-').lower()}.jpg",
            f"{meal_option.replace(' & ', '_').replace(' ', '_').lower()}.jpg", 
            f"{meal_option.lower().replace(' & ', '-').replace(' ', '-')}.png",
            f"{'-'.join(meal_option.lower().split())}.jpg",
            f"{meal_option.lower().replace(' & ', '').replace(' ', '')}.jpg",
            f"{meal_option.lower().replace(' & ', 'and').replace(' ', '-')}.jpg"
        ]
        
        
        for filename in possible_filenames:
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                try:
                    return Image.open(img_path).resize((300, 200))
                except Exception as e:
                    print(f"Error opening image {img_path}: {e}")
         
        print(f"No image found for: {meal_option}")
    return None

for i, row in enumerate(df_merged.itertuples()):
    if i < len(cols):  
        is_popular = row.Scaled_Count == max_count
        
        cols[i].markdown(f"""
        <div class='metric-card {"popular-meal" if is_popular else ""}'>
            <h3 style='color: #003087; margin-top: 0;'>{row.Meal_Option}</h3>
            <h2 style='color: #003087; margin: 0.5rem 0;'>{row.Scaled_Count} students</h2>
            <p style='{"color: green;" if row.Delta >= 0 else "color: red;"}'>
                {'+' if row.Delta > 0 else ''}{row.Delta} from last week
            </p>
            {f"<span style='background-color: #FFD700; color: #003087; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem;'>Most Popular</span>" if is_popular else ""}
        </div>
        """, unsafe_allow_html=True)
        
        img = get_image(row.Meal_Option)
        if img:
            cols[i].image(img, use_container_width=True)
        else:
            cols[i].markdown(f"""
            <div style='background-color: #E6EEF8; height: 150px; display: flex; align-items: center; justify-content: center; border-radius: 5px;'>
                <p style='text-align: center; color: #666;'>No image available</p>
            </div>
            """, unsafe_allow_html=True)

if cafeteria_df is not None:
    trend_actual = actual_counts_all_weeks[
        (actual_counts_all_weeks['Day'] == day) & 
        (actual_counts_all_weeks['Meal'] == meal)
    ]
    
    trend_weeks = []
    for week in trend_actual['Week'].unique():
        week_data = trend_actual[trend_actual['Week'] == week].copy()
        total_students = week_data['Actual_Count'].sum()
        if total_students > 0:
            week_data.loc[:, 'Scaled_Count'] = (week_data['Actual_Count'] / total_students * students_in_school).round().astype(int)
            week_data.loc[:, 'Data_Type'] = 'Actual'
            trend_weeks.append(week_data[['Week', 'Day', 'Meal', 'Meal_Option', 'Scaled_Count', 'Data_Type']])
    
    forecast_trend = df_week_now[['Week', 'Day', 'Meal', 'Meal_Option', 'Scaled_Count']].copy()
    forecast_trend.loc[:, 'Data_Type'] = 'Forecast'
    
    if trend_weeks:
        trend_data = pd.concat([pd.concat(trend_weeks), forecast_trend])
    else:
        trend_data = forecast_trend
else:
    trend_data = df[(df['Day'] == day) & (df['Meal'] == meal)].copy()
    trend_data.loc[:, 'Data_Type'] = 'Forecast'
    
    for week in trend_data['Week'].unique():
        week_data = trend_data[trend_data['Week'] == week]
        total_predicted = week_data['Predicted_Count'].sum()
        if total_predicted > 0:
            trend_data.loc[trend_data['Week'] == week, 'Scaled_Count'] = (
                week_data['Predicted_Count'] / total_predicted * students_in_school
            ).round().astype(int)
        else:
            trend_data.loc[trend_data['Week'] == week, 'Scaled_Count'] = 0

meal_options_list = trend_data['Meal_Option'].unique()
trend_charts = []

for option in meal_options_list:
    option_data = trend_data[trend_data['Meal_Option'] == option].copy()
    
    option_data = option_data.sort_values('Week')
    
    trend_charts.append(option_data)

combined_trend = pd.concat(trend_charts)
combined_trend.loc[:, 'Line_Style'] = combined_trend.apply(
    lambda x: 'Actual' if x['Data_Type'] == 'Actual' else 'Forecast', axis=1
)

st.markdown("<h2 class='subheader'>📈 Meal Option Trends Over Weeks</h2>", unsafe_allow_html=True)

trend_data = trend_data.sort_values(['Meal_Option', 'Week'])

fig = px.line(
    trend_data, 
    x='Week', 
    y='Scaled_Count', 
    color='Meal_Option',
    markers=True, 
    labels={'Scaled_Count': 'Student Count', 'Week': 'Week Number'},
    title=f"Trend for {meal} on {day} (scaled to {students_in_school} students)"
)

color_sequence = [BABCOCK_BLUE, BABCOCK_GOLD, "#4682B4", "#8B4513", "#2E8B57", "#800080"]
fig.update_traces(line=dict(width=3)) 

fig.update_layout(
    legend_title_text='Meal Options',
    xaxis_title="Week",
    yaxis_title="Student Count",
    plot_bgcolor='white',
    font=dict(family="Segoe UI, Arial, sans-serif"),
    title_font=dict(family="Segoe UI, Arial, sans-serif", color=BABCOCK_BLUE),
    colorway=color_sequence,  
    xaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        tickmode='linear',
        dtick=1, 
        title_font=dict(family="Segoe UI, Arial, sans-serif", color=BABCOCK_BLUE)
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        title_font=dict(family="Segoe UI, Arial, sans-serif", color=BABCOCK_BLUE)
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor=BABCOCK_BLUE,
        borderwidth=1
    )
)

fig.add_vline(
    x=week_now - 0.5, 
    line_width=1, 
    line_dash="dash", 
    line_color=BABCOCK_BLUE,
    annotation_text="Forecast",
    annotation_position="top right",
    annotation_font=dict(color=BABCOCK_BLUE)
)

fig.add_vrect(
    x0=week_now - 0.5,
    x1=week_now + 0.5,
    fillcolor=BABCOCK_LIGHT_BLUE,
    opacity=0.3,
    layer="below",
    line_width=0,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class='info-box'>
    <h3 style='color: #003087; margin-top: 0;'>How to Use This Dashboard</h3>
    <ul style='margin-bottom: 0;'>
        <li>The dashboard shows forecasted meal counts for the current week scaled to the on-campus student population.</li>
        <li>Delta values show the change between the current week forecast and the previous week's actual counts.</li>
        <li>Trend lines display actual and forecasted counts for each meal option over time.</li>
        <li>All counts are scaled to the current school population for consistent comparison.</li>
        <li>The most popular meal option is highlighted with a gold border.</li>
        <li>Click the "Update Records & Forecasts" button to generate new data and forecasts for the next week.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
    <p>© 2024 Babcock University Cafeteria Management System</p>
    <p style='color: #003087;'>Excellence. Integrity. Service.</p>
</div>
""", unsafe_allow_html=True)
