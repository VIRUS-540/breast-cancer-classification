import streamlit as st
import joblib
import pandas as pd
import sqlite3
from datetime import datetime

def get_db():
    connection = sqlite3.connect('prediction.db')
    return connection

def init_db():
    connection = get_db()
    cursor = connection.cursor()
    cursor.execute(
        '''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                radius_mean REAL,
                texture_mean REAL,
                perimeter_mean REAL,
                area_mean REAL,
                smoothness_mean REAL,
                compactness_mean REAL,
                concavity_mean REAL,
                concave_points_mean REAL,
                symmetry_mean REAL,
                fractal_dimension_mean REAL,
            
                radius_se REAL,
                texture_se REAL,
                perimeter_se REAL,
                area_se REAL,
                smoothness_se REAL,
                compactness_se REAL,
                concavity_se REAL,
                concave_points_se REAL,
                symmetry_se REAL,
                fractal_dimension_se REAL,
            
                radius_worst REAL,
                texture_worst REAL,
                perimeter_worst REAL,
                area_worst REAL,
                smoothness_worst REAL,
                compactness_worst REAL,
                concavity_worst REAL,
                concave_points_worst REAL,
                symmetry_worst REAL,
                fractal_dimension_worst REAL,

                prediction TEXT,
                probability REAL
            )
        '''
    )
    connection.commit()
    connection.close()

def save_prediction(input_data,result,prob, timestamp):
    connection = get_db()
    cursor = connection.cursor()
    cursor.execute(
        '''
            INSERT INTO predictions (
            timestamp,
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean,
            symmetry_mean,
            fractal_dimension_mean,
        
            radius_se,
            texture_se,
            perimeter_se,
            area_se,
            smoothness_se,
            compactness_se,
            concavity_se,
            concave_points_se,
            symmetry_se,
            fractal_dimension_se,
        
            radius_worst,
            texture_worst,
            perimeter_worst,
            area_worst,
            smoothness_worst,
            compactness_worst,
            concavity_worst,
            concave_points_worst,
            symmetry_worst,
            fractal_dimension_worst,
            prediction, 
            probability)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''',
        (*input_data,result,prob,timestamp)
    )
    connection.commit()
    connection.close()

init_db()
st.title('Breast Cancer Classification')
model = joblib.load('breast_cancer_model.pkl')

#Mean Measurements
st.subheader('Mean Measurements')
radius_mean = st.number_input(
    label='Radius Mean',
    placeholder = 'Enter radius mean: ',
    value=None
)
texture_mean = st.number_input(
    label='Texture Mean',
    placeholder = 'Enter texture mean: ',
    value=None
)
perimeter_mean = st.number_input(
    label='Perimeter Mean',
    placeholder = 'Enter perimeter mean: ',
    value=None
)
area_mean = st.number_input(
    label='Area Mean',
    placeholder = 'Enter area mean: ',
    value=None
)
smoothness_mean = st.number_input(
    label='Smoothness Mean',
    placeholder = 'Enter Smoothness mean: ',
    value=None,
    format = '%.4f'
)
compactness_mean = st.number_input(
    label='Compactness Mean',
    placeholder = 'Enter compactness mean: ',
    value=None,
    format = '%.4f'
)
concavity_mean = st.number_input(
    label='Concavity Mean',
    placeholder = 'Enter concavity mean: ',
    value=None,
    format = '%.4f'
)
concave_points_mean = st.number_input(
    label='Concave Points Mean',
    placeholder = 'Enter concave points mean: ',
    value=None,
    format = '%.4f'
)
symmetry_mean = st.number_input(
    label='Symmetry Mean',
    placeholder = 'Enter symmetry mean: ',
    value=None,
    format = '%.4f'
)
fractal_dimension_mean = st.number_input(
    label='Fractal Dimension Mean',
    placeholder = 'Enter Fractal Dimension mean:',
    value=None,
    format = '%.5f'
)

#Standard Error Measurements
st.subheader('Standard Error Measurements')
radius_se = st.number_input(
    label='Radius SE',
    placeholder = 'Enter radius se: ',
    value=None,
    format = '%.4f'
)
texture_se = st.number_input(
    label='Texture SE',
    placeholder = 'Enter texture se: ',
    value=None,
    format = '%.4f'
)
perimeter_se = st.number_input(
    label='Perimeter SE',
    placeholder = 'Enter perimeter se: ',
    value=None,
    format = '%.3f'
)
area_se = st.number_input(
    label='Area SE',
    placeholder = 'Enter area se: ',
    value=None,
    format = '%.4f'    
)
smoothness_se = st.number_input(
    label='Smoothness SE',
    placeholder = 'Enter Smoothness se: ',
    value=None,
    format = '%.6f'
)
compactness_se = st.number_input(
    label='Compactness SE',
    placeholder = 'Enter compactness se: ',
    value=None,
    format = '%.5f'
)
concavity_se = st.number_input(
    label='Concavity SE',
    placeholder = 'Enter concavity se: ',
    value=None,
    format = '%.5f'
)
concave_points_se = st.number_input(
    label='Concave Points SE',
    placeholder = 'Enter concave points se: ',
    value=None,
    format = '%.5f'
)
symmetry_se = st.number_input(
    label='Symmetry SE',
    placeholder = 'Enter symmetry se: ',
    value=None,
    format = '%.5f'
)
fractal_dimension_se = st.number_input(
    label='Fractal Dimension SE',
    placeholder = 'Enter Fractal Dimension se:',
    value=None,
    format = '%.6f'
)

#Worst Measurements
st.subheader('Worst Measurements')
radius_worst = st.number_input(
    label='Radius Worst',
    placeholder = 'Enter radius worst: ',
    value=None
)
texture_worst = st.number_input(
    label='Texture Worst',
    placeholder = 'Enter texture worst: ',
    value=None
)
perimeter_worst = st.number_input(
    label='Perimeter Worst',
    placeholder = 'Enter perimeter worst: ',
    value=None
)
area_worst = st.number_input(
    label='Area Worst',
    placeholder = 'Enter area worst: ',
    value=None
)
smoothness_worst = st.number_input(
    label='Smoothness Worst',
    placeholder = 'Enter Smoothness worst: ',
    value=None,
    format = '%.4f'
)
compactness_worst = st.number_input(
    label='Compactness Worst',
    placeholder = 'Enter compactness worst: ',
    value=None,
    format = '%.4f'
)
concavity_worst = st.number_input(
    label='Concavity Worst',
    placeholder = 'Enter concavity Worst: ',
    value=None,
    format = '%.5f'
)
concave_points_worst = st.number_input(
    label='Concave Points Worst',
    placeholder = 'Enter concave points Worst: ',
    value=None,
    format = '%.4f'
)
symmetry_worst = st.number_input(
    label='Symmetry Worst',
    placeholder = 'Enter symmetry Worst: ',
    value=None,
    format = '%.4f'
)
fractal_dimension_worst = st.number_input(
    label='Fractal Dimension Worst',
    placeholder = 'Enter Fractal Dimension worst:',
    value=None,
    format = '%.5f'
)

input_data = [
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    concave_points_mean,
    symmetry_mean,
    fractal_dimension_mean,

    radius_se,
    texture_se,
    perimeter_se,
    area_se,
    smoothness_se,
    compactness_se,
    concavity_se,
    concave_points_se,
    symmetry_se,
    fractal_dimension_se,

    radius_worst,
    texture_worst,
    perimeter_worst,
    area_worst,
    smoothness_worst,
    compactness_worst,
    concavity_worst,
    concave_points_worst,
    symmetry_worst,
    fractal_dimension_worst
]

feature_names = [
    'radius_mean',                
    'texture_mean',                 
    'perimeter_mean',               
    'area_mean',                   
    'smoothness_mean',          
    'compactness_mean',             
    'concavity_mean',               
    'concave points_mean',          
    'symmetry_mean',                
    'fractal_dimension_mean',  
    
    'radius_se',                    
    'texture_se',                   
    'perimeter_se',                 
    'area_se',                      
    'smoothness_se',                
    'compactness_se',              
    'concavity_se',                 
    'concave points_se',            
    'symmetry_se',                  
    'fractal_dimension_se',  
    
    'radius_worst',                 
    'texture_worst',               
    'perimeter_worst',              
    'area_worst',                  
    'smoothness_worst',             
    'compactness_worst',            
    'concavity_worst',             
    'concave points_worst',         
    'symmetry_worst',               
    'fractal_dimension_worst'      
]

input_df = pd.DataFrame(
    [input_data],
    columns=feature_names
)
st.dataframe(input_df)

if st.button('Predict'):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if prediction[0] == 0:
        result = 'Benign'
        st.success(result)
        prob = probability[0][0] * 100
        st.write('Benign Probability: ',prob)
    else:
        result = 'Malignant'
        st.error(result)
        prob = probability[0][1] * 100
        st.write('Malignant Probability: ',prob)

    save_prediction(input_data, result, prob, timestamp)

connection = get_db()

df = pd.read_sql_query(
    'SELECT * FROM predictions',
    connection
)

st.dataframe(df)

connection.close()