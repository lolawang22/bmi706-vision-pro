import altair as alt
import pandas as pd
import streamlit as st

### Main Page ###
def main_page():
    st.title("Welcome to Vision Pro Project👋")
    st.write("This is the main page. Select a page from the dropdown in the sidebar to navigate.")


### Page 1 ###
## Question: What is the relationship between the patient’s demographic and the outcomes of arterial catheterization?
def page1():
    st.title("Question 1: What is the relationship between the patient’s demographic and the outcomes of arterial catheterization?")
    st.write("This is the content of the first page.")

### Page 2 ###
## Question: How do initial clinical assessments and vital signs correlate with the length of stay in the ICU and hospital?

def page2():
    st.title("Question 2: How do initial clinical assessments and vital signs correlate with the length of stay in the ICU and hospital?")
    st.write("This is the content of the second page.")

### Page 3 ###
## Question: Does the time of day or week of ICU admission affect patients’ outcomes or complications?
def page3():
    st.title("Question 3: Does the time of day or week of ICU admission affect patients’ outcomes or complications?")
    st.write("This is the content of the third page.")


# Dictionary of pages
pages = {
    "Main Page": main_page,
    "Question 1: What is the relationship between the patient’s demographic and the outcomes of arterial catheterization?": page1,
    "Question 2: How do initial clinical assessments and vital signs correlate with the length of stay in the ICU and hospital?": page2,
    "Question 3: Does the time of day or week of ICU admission affect patients’ outcomes or complications?": page3,
}

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.selectbox("Go to", list(pages.keys()))

# Display the selected page using the dictionary
page = pages[selection]
page()

