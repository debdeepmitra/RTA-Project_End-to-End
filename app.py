import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prediction import ordinal_encoder, get_prediction
import shap

model = joblib.load(r'C:\Users\Debdeep\Desktop\TMLC\Project-1\xgb.joblib')

st.set_page_config(page_title ='Accident severity prediction', page_icon = '🚨', layout ='wide')

options_day = ['Monday', 'Sunday', 'Friday', 'Wednesday', 'Saturday', 'Thursday', 'Tuesday']
options_driver_age = ['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown']
options_driver_sex = ['Male', 'Female', 'Unknown']
options_driver_education_level = ['Above high school', 'Junior high school', 'Elementary school',
 'High school', 'Unknown', 'Illiterate', 'Writing & reading']
options_driving_experience = ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence', 'Below 1yr', 'NA']
options_vehicle_type = ['Automobile', 'Public (> 45 seats)', 'Lorry (41-100Q)',
 'Public (13-45 seats)', 'Lorry (11-40Q)','Long lorry', 'Public (12 seats)',
 'Taxi', 'Pick up upto 10Q', 'Stationwagen', 'Ridden horse', 'Other', 'Bajaj',
 'Turbo', 'Motorcycle', 'Special vehicle', 'Bicycle']
options_vehicle_owner = ['Owner', 'Governmental', 'Organization', 'Other']
options_service_year = ['Above 10yr', '5-10yrs', 'Unknown', '1-2yr', '2-5yrs', 'Below 1yr']
options_accident_areas = ['Residential areas', 'Office areas', '  Recreational areas',
 ' Industrial areas', 'Other', ' Church areas', '  Market areas', 'Unknown',
 'Rural village areas', ' Outside rural areas', ' Hospital areas',
 'School areas', 'Rural village areasOffice areas', 'Recreational areas']
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
 'other', 'Double carriageway (median)', 'One way',
 'Two-way (divided with solid lines road marking)', 'Unknown']
options_road_allignment = ['Tangent road with flat terrain',
 'Tangent road with mild grade and flat terrain', 'Escarpments',
 'Tangent road with rolling terrain', 'Gentle horizontal curve',
 'Tangent road with mountainous terrain',
 'Steep grade downward with mountainous terrain', 'Sharp reverse curve',
 'Steep grade upward with mountainous terrain']
options_junction_type = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape', 'X Shape']
options_surface_types = ['Asphalt roads', 'Earth roads', 'Asphalt roads with some distress', 'Gravel roads', 'Other']
options_road_surface_conditions = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']
options_light_condition = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit']
options_weather_condition = ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy', 'Snow', 'Unknown', 'Fog or mist']
options_collision_type = ['Collision with roadside-parked vehicles',
 'Vehicle with vehicle collision', 'Collision with roadside objects',
 'Collision with animals', 'Other', 'Rollover', 'Fall from vehicles',
 'Collision with pedestrians', 'With Train', 'Unknown']


options_vehicle_movement = ['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go',
 'Getting off', 'Reversing', 'Unknown', 'Parked', 'Stopping', 'Overtaking',
 'Other', 'Entering a junction']
options_casualty_class = ['NA', 'Driver or rider', 'Pedestrian', 'Passenger']
options_casualty_sex = ['NA', 'Male', 'Female']
options_casualty_age = ['NA', '31-50', '18-30', 'Under 18', 'Over 51', '5']
options_pedestrian_movement = ['Not a Pedestrian', "Crossing from driver's nearside",
 'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle',
 'Unknown or other',
 'Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle',
 'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)',
 'Walking along in carriageway, back to traffic',
 'Walking along in carriageway, facing traffic',
 'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle']
options_accident_cause = ['Moving Backward', 'Overtaking', 'Changing lane to the left',
 'Changing lane to the right', 'Overloading', 'Other',
 'No priority to vehicle', 'No priority to pedestrian', 'No distancing',
 'Getting off the vehicle improperly', 'Improper parking', 'Overspeed',
 'Driving carelessly', 'Driving at high speed', 'Driving to the left',
 'Unknown', 'Overturning', 'Turnover', 'Driving under the influence of drugs',
 'Drunk driving']







feats = ['day_of_week', 'driver_age', 'driver_sex', 'driver_education_level', 'driver_experience', 'vehicles_type', 'vehicles_owner', 'service_year', 'accident_areas', 'lanes', 'road_allignment', 'junction_type', 'surface_types', 'road_condition', 'light_condition', 'weather_condition', 'collision_type', 'vehicles_involved', 'casualties', 'vehicle_movement', 'casualty_class', 'casulty_sex', 'casualty_age', 'pedestrian_movement', 'accident_cause', 'hour', 'minute']

st.markdown("<h1 style = 'text-align: center;'>Accident Severity Prediction Application 🚨</h1>", unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):

        st.subheader('Enter the input to following features:')

        hour = st.slider('Hour of accident: ', 0, 23, value=0, format='%d')
        day_of_week = st.selectbox('Day of week: ', options=options_day)
        driver_age = st.selectbox('Driver age: ', options=options_driver_age)
        driver_sex = st.selectbox('Driver sex: ', options=options_driver_sex)
        driver_education_level = st.selectbox('Driver education level: ', options=options_driver_education_level)
        driver_experience = st.selectbox('Driver experience: ', options=options_driving_experience)
        vehicles_type = st.selectbox('Type of vehicle: ', options=options_vehicle_type)
        vehicles_owner = st.selectbox('Vehicle owner: ', options=options_vehicle_owner)
        service_year = st.selectbox('Vehicle service year: ', options=options_service_year)
        accident_areas = st.selectbox('Area of accident: ', options=options_accident_areas)
        lanes = st.selectbox('Lanes: ', options=options_lanes)
        road_allignment = st.selectbox('Allignment of road: ', options=options_road_allignment)
        junction_type = st.selectbox('Type of junction: ', options=options_junction_type)
        surface_types = st.selectbox('Type of surface: ', options=options_surface_types)
        road_condition = st.selectbox('Road condition: ', options=options_road_surface_conditions)
        light_condition = st.selectbox('Light condition: ', options=options_light_condition)
        weather_condition = st.selectbox('Weather condition: ', options=options_weather_condition)
        collision_type = st.selectbox('Type of collision: ', options=options_collision_type)
        vehicles_involved = st.slider('No of vehicles involved: ', 1, 7, value=0, format='%d')
        casualties = st.slider('No of casualties: ', 0, 7, value=0, format='%d')
        vehicle_movement = st.selectbox('Type of collision: ', options=options_vehicle_movement)
        casualty_class = st.selectbox('Class of casualty: ', options=options_casualty_class)
        casulty_sex = st.selectbox('Sex of casualty: ', options=options_casualty_sex)
        casualty_age = st.selectbox('Age of casualty: ', options=options_casualty_age)
        pedestrian_movement = st.selectbox('Movement of pedestrian: ', options=options_pedestrian_movement)
        accident_cause = st.selectbox('Cause of accident: ', options=options_accident_cause)                 
        

        submit = st.form_submit_button('Predict Injury')

    if submit:

        day_of_week = ordinal_encoder(day_of_week, options_day)
        driver_age = ordinal_encoder(driver_age, options_driver_age)
        driver_sex = ordinal_encoder(driver_sex, options_driver_sex)
        driver_education_level = ordinal_encoder(driver_education_level, options_driver_education_level)
        driver_experience = ordinal_encoder(driver_experience, options_driving_experience)
        vehicles_type = ordinal_encoder(vehicles_type, options_vehicle_type)
        vehicles_owner = ordinal_encoder(vehicles_owner, options_vehicle_owner)
        service_year = ordinal_encoder(service_year, options_service_year)
        accident_areas = ordinal_encoder(accident_areas, options_accident_areas)
        lanes = ordinal_encoder(lanes, options_lanes)
        road_allignment = ordinal_encoder(road_allignment, options_road_allignment)
        junction_type = ordinal_encoder(junction_type, options_junction_type)
        surface_types = ordinal_encoder(surface_types, options_surface_types)
        road_condition = ordinal_encoder(road_condition, options_road_surface_conditions)
        light_condition = ordinal_encoder(light_condition, options_light_condition)
        weather_condition = ordinal_encoder(weather_condition, options_weather_condition)
        collision_type = ordinal_encoder(collision_type, options_collision_type)       
        vehicle_movement = ordinal_encoder(vehicle_movement, options_vehicle_movement)
        casualty_class = ordinal_encoder(casualty_class, options_casualty_class)
        casulty_sex = ordinal_encoder(casulty_sex, options_casualty_sex)
        casualty_age = ordinal_encoder(casualty_age, options_casualty_age)
        pedestrian_movement = ordinal_encoder(pedestrian_movement, options_pedestrian_movement)
        accident_cause = ordinal_encoder(accident_cause, options_accident_cause)

        

        data = np.array([day_of_week, driver_age, driver_sex, driver_education_level, driver_experience, vehicles_type, vehicles_owner, service_year, accident_areas, lanes, road_allignment, junction_type, surface_types, road_condition, light_condition, weather_condition, collision_type, vehicles_involved, casualties, vehicle_movement, casualty_class, casulty_sex, casualty_age, pedestrian_movement, accident_cause, hour, 0]).reshape(1,-1)
        pred = get_prediction(data, model)

        st.write(f'The predicted severity is:  {pred}')




main()