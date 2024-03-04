import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def load_data():
    df = pd.read_csv('full_cohort_data.csv')
    
    # Impute numeric columns with median
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    imputer_numeric = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])

    # Impute categorical/binary columns with the mode
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer_categorical.fit_transform(df[categorical_columns])

    # Create new column types for future analysis
    df['gender'] = df['gender_num'].map({1: 'M', 0: 'F'})
    
    df['age_range'] = pd.cut(df['age'].astype(int), bins=[-float('inf'), 20, 29, 39, 49, 59, 69, 79, 89, float('inf')], 
                             labels=['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '>=90'], 
                             right=False)

    df['weight_range'] = pd.cut(df['weight_first'].astype(int), bins=[-float('inf'), 50, 79, 109, float('inf')], 
                                labels=['<50kg', '50-79kg', '80-109kg', '>=110kg'])
   
    df['bmi_range'] = pd.cut(df['bmi'].round(1), bins=[-float('inf'), 18.5, 24.9, 29.9, float('inf')], 
                                labels=['<18.5', '18.5-24.9', '25-29.9', '>=30'])

    return df

df = load_data()

### Main Page ###
def main_page():
    st.title("Welcome to Vision Pro Project👋")
    st.write("This is the main page. **👈 Select a page from the dropdown in the sidebar** to navigate this project.")
    st.markdown(
    """
    ### Dataset Info
    - Clinical data from the MIMIC-II database for a case study on indwelling arterial catheters [dataset](https://physionet.org/content/mimic2-iaccd/1.0/)
    - 1776 patients with 46 variables: 27 quantitative variables, 16 categorical/binary variables, 1 ordinal variable, and 2 nominal variables
    - Key statistics: the number of patients, their distinct clinical measurements, lab test results, and medication orders
    ### Target Audience
    - Clinicians, medical researchers, hospital administrators, and quality improvement teams
    - Enhance understanding of the clinical use of arterial catheters
    - Inform evidence-based improvements in patient care
    - Support research into the prevention of catheter-related complications
    """
    )


### Page 1 ###
## Question: What is the relationship between the patient’s demographic and the outcomes of arterial catheterization?
def page1():
    st.title("The relationship between the patient’s demographic and the outcomes of arterial catheterization")
    
    # Add radio buttons for gender
    gender = st.radio("Gender", ["M", "F"])
    subset = df[(df["gender"] == gender) & (df["aline_flg"] == 1)]
    
    # Add multiselect options for age range
    age_default = [
        "40-49",
        "50-59",
        "60-69",
        "70-79",
    ]
    age = st.multiselect("Age", df["age_range"].unique(), age_default)
    subset = subset[subset["age_range"].isin(age)]

    # Add multiselect options for weight range
    weight_default = [
        "<50kg",
        "50-79kg",
    ]
    weight = st.multiselect("Weight", df["weight_range"].unique(), weight_default)
    subset = subset[subset["weight_range"].isin(weight)]

    # Create bar chart for BMI vs. mortality rate
    bmis =  [
        '<18.5', 
        '18.5-24.9', 
        '25-29.9', 
        '>=30'
    ]
    mortality_rate = subset.groupby(['bmi_range', 'service_unit'])['hosp_exp_flg'].mean().reset_index()
    mortality_rate['mortality_rate'] = mortality_rate['hosp_exp_flg'] * 100
    mortality_bmi_bar_chart = alt.Chart(mortality_rate).mark_bar().encode(
        x=alt.X('bmi_range', sort=bmis, title='BMI Range'),
        y=alt.Y('mortality_rate', title='Mortality Rate'), 
        color=alt.Color('service_unit:N', title='Service Unit', scale=alt.Scale(scheme='blues')),
        tooltip=[alt.Tooltip('mortality_rate:Q', title='Mortality Rate'), alt.Tooltip('service_unit:N', title='Service Unit')]
    ).properties(
        width=600,
        height=400,
        title=f"BMI vs. mortality rate in hospital"
    )

    # Create pie chart for BMI distribution
    bmi_pie_chart = alt.Chart(subset).mark_arc().encode(
        theta=alt.Theta(field='bmi_range', type='quantitative', aggregate='count'),
        color=alt.Color(field='bmi_range', type='nominal', sort=bmis, title='BMI Range', scale=alt.Scale(scheme='greenblue')),
        tooltip=[alt.Tooltip('count()', title='Count')]
    ).properties(
        width=300,
        height=300,
        title=f"BMI count for {'males' if gender == 'M' else 'females'} in the specific age range and weight range"
    )

    st.altair_chart(mortality_bmi_bar_chart, use_container_width=True)
    st.altair_chart(bmi_pie_chart, use_container_width=True)

### Page 2 ###
## Question: How do initial clinical assessments and vital signs correlate with the length of stay in ICU or hospital?

def page2():
    st.title("The correlation of initial clinical assessments and vital signs with the length of stay in ICU or hospital")
    
    # Add radio buttons for choosing patient initial score
    score_option = st.radio("Patient Initial Score", ["SAPS I Score", "SOFA Score"])
    if score_option == "SAPS I Score":
        score_column = "sapsi_first"
    elif score_option == "SOFA Score":
        score_column = "sofa_first"

    # Add radio buttons for choosing patient length of stay
    los_option = st.radio("Patient Length of Stay", ["Days in ICU", "Days in Hospital"])
    if los_option == "Days in ICU":
        los_column = "icu_los_day"
    elif los_option == "Days in Hospital":
        los_column = "hospital_los_day"

    # Add multiselect options for patient initial clinical observation
    obs_option = ['map_1st', 'hr_1st', 'temp_1st', 'spo2_1st', 'abg_count', 'wbc_first', 'hgb_first', 'platelet_first', 'sodium_first', 
                  'potassium_first', 'tco2_first', 'chloride_first', 'bun_first', 'creatinine_first', 'po2_first', 'pco2_first']
    obs_default = [
        "hr_1st",
        "temp_1st",
        "potassium_first", 
        "tco2_first",
    ]
    obs = st.multiselect("Patient Initial Clinical Observation", obs_option, obs_default)

    # Create scatter plot for selected patient initial score and length of stay
    scatter_chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(score_column, title="Patient Initial Score"),
        y=alt.Y(los_column, title="Patient Length of Stay"),
        tooltip=[alt.Tooltip(score_column, title='Score'), alt.Tooltip(los_column, title='LOS')]
    ).properties(
        width=600,
        height=400,
        title=f"Scatter Plot of Patient {score_option} vs. {los_option}"
    )
    st.altair_chart(scatter_chart, use_container_width=True)
    
    # Perform machine learning feature selection if there are too many variables chosen
    if obs:
        selected_features = obs + [score_column, los_column]
        subset_df = df[selected_features]

        if len(obs) > 3:
            X = subset_df[obs]
            y = subset_df[los_column]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            indices = np.argsort(importances)[-3:]
            top_vars = [obs[i] for i in indices]
        else:
            top_vars = obs

        top_vars += [score_column, los_column]
        corr_df = subset_df[top_vars].corr()

        corr_df.reset_index(inplace=True)
        corr_df = corr_df.melt(id_vars='index', var_name='Variable', value_name='Correlation')
        
        # Create heatmap for the correlation between clinical variables and length of stay
        clinical_LOS_chart = alt.Chart(corr_df).mark_rect().encode(
            x=alt.X('Variable:N', title='Variable'),
            y=alt.Y('index:N', title=''),
            color='Correlation:Q',
            tooltip=['Variable', 'index', 'Correlation']
        ).properties(
            width=600, 
            height=400, 
            title="Correlation matrix of selected clinical measurements and length of stay")

        st.altair_chart(clinical_LOS_chart, use_container_width=True)


### Page 3 ###
## Question: Does the time of day or week of ICU admission affect patients’ outcomes or complications?
def page3():
    st.title("The relationship between ICU admission time (24h), length (days) and patients’ outcomes")

    mortality = df[['icu_los_day', 'hour_icu_intime', 'censor_flg']].copy()
    
    brush = alt.selection_interval(encodings=['x'])

    days_bins = [0, 5, 10, 15, 20, 25, float('inf')]
    days = ['Day 0-5', 'Day 6-10', 'Day 11-15', 'Day 16-20', 'Day 21-25', 'Day 26+']
    mortality['icu_los_day_group'] = pd.cut(mortality['icu_los_day'], bins=days_bins, labels=days, right=False)

    intime_bins = [-0.01, 6, 12, 18, 24]
    intime_labels = ['Hour 0-6', 'Hour 7-12', 'Hour 13-18', 'Hour 19-24']
    mortality['hour_icu_int_group'] = pd.cut(mortality['hour_icu_intime'], bins=intime_bins, labels=intime_labels, right=True)

    mortality = mortality.groupby(['icu_los_day_group', 'hour_icu_int_group']).agg(
        death_count=('censor_flg', lambda x: (x==0).sum()),
        patient_count=('censor_flg', 'size'),
        mortality_rate=('censor_flg', lambda x: (x==0).mean())
    ).reset_index()

    chart = alt.Chart(mortality).mark_rect().encode(
        x=alt.X('icu_los_day_group:O', sort=days, title='ICU Length of Stay (days)'), 
        y=alt.Y('hour_icu_int_group:O', sort=intime_labels, title='ICU Admission Hour (24h)'), 
        color=alt.Color('mortality_rate:Q',
                        legend=alt.Legend(title="Mortality rate")),
        tooltip=['mortality_rate:Q'] 
    ).add_selection(
        brush
    ).properties(
        title='Mortality Rate of ICU Length of Stay vs. Admission Hour',
        width=500,
        height=150
    )

    bar_chart = alt.Chart(mortality).mark_bar().encode(
        x=alt.X('sum(death_count):Q', title='Sum of Deaths'),
        y=alt.Y('icu_los_day_group:O', sort='-x', title='ICU Length of Stay (days)'),
        tooltip=['icu_los_day_group:O', 'sum(death_count):Q']
    ).transform_filter(
        brush
    ).properties(
        title='Deaths Counts by ICU Length of Stay',
        height=150,
        width=500
    )

    st.altair_chart(chart & bar_chart, use_container_width=True)

# Dictionary of pages
pages = {
    "Main Page": main_page,
    "Question 1: What is the relationship between the patient’s demographic and the outcomes of arterial catheterization?": page1,
    "Question 2: How do initial clinical assessments and vital signs correlate with the length of stay in ICU or hospital?": page2,
    "Question 3: What are the relationships between ICU admission time (24h), length (days) and patients’ outcomes?": page3,
}

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.selectbox("Go to", list(pages.keys()))

# Display the selected page using the dictionary
page = pages[selection]
page()
