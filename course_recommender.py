"""
COURSE RECOMMENDATION SYSTEMS

Authors: Sathiya Narayan Chakravarthy, Sanjana Ganesh, Sanika Hadatgune, Prasiddha Sudhakar

Sources referred :
https://medium.com/@prateekgaurav/step-by-step-content-based-recommendation-system-823bbfd0541c
https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243
https://medium.com/@srang992/making-a-movie-recommendation-app-using-streamlit-and-docker-part-1-8dee8983cea9
"""

from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

import matplotlib.pyplot as plt
import plotly.express as px
import requests
import json, collections, time, re, string
from datetime import datetime


@st.cache_data
def load_data():
    df = pd.concat([pd.read_csv(f"data/coursera-courses-overview.csv"),
                    pd.read_csv(f"data/coursera-individual-courses.csv")], axis=1)

    # preprocess data
    new = []
    for c in df.columns:
        new.append(c.lower().replace(' ', '_'))
    df.columns = new

    df['instructors'] = df['instructors'].fillna('Missing')
    df['skills'] = df['skills'].fillna('Missing')

    # split by skills
    df['skills'] = df['skills'].apply(lambda x: x.split(','))

    df['enrolled_student_count'] = df['enrolled_student_count'].apply(
        lambda x: float(x.replace('k', '')) * 1000 if 'k' in x else (
            float(x.replace('m', '')) * 1000000 if 'm' in x else np.nan))

    # making number features numeric
    df['course_rating'] = df['course_rating'].apply(lambda x: np.nan if x == 'Missing' else float(x))
    df['course_rated_by'] = df['course_rated_by'].apply(lambda x: np.nan if x == 'Missing' else float(x))
    df['percentage_of_new_career_starts'] = df['percentage_of_new_career_starts'].apply(
        lambda x: np.nan if x == 'Missing' else float(x))
    df['percentage_of_pay_increase_or_promotion'] = df['percentage_of_pay_increase_or_promotion'].apply(
        lambda x: np.nan if x == 'Missing' else float(x))

    # input Approx 10 months to complete -> return '10 months'
    def find_time(x):
        l = x.split(' ')
        idx = 0
        for i in range(len(l)):
            if l[i].isdigit():
                idx = i
        try:
            return l[idx] + ' ' + l[idx + 1]
        except:
            return l[idx]

    df['estimated_time_to_complete'] = df['estimated_time_to_complete'].apply(find_time)

    return df


@st.cache_data
def filter_records(dataframe, chosen_options, feature, id):
    selected_records = [dataframe[id][i] for i in range(1000) for op in chosen_options if op in dataframe[feature][i]]
    return selected_records


def extract_keywords(df, feature):
    keyword_lists = []
    r = Rake()
    for i in range(len(df)):
        descr = df[feature][i]
        r.extract_keywords_from_text(descr)
        key_words_dict_scores = r.get_word_degrees()
        keywords_string = " ".join(list(key_words_dict_scores.keys()))
        keyword_lists.append(keywords_string)

    return keyword_lists


def recommendations(df, input_course, cosine_sim, num_courses, find_similar=True):
    # find the index of the input course
    idx = df.index[df['course_name'] == input_course].tolist()[0]

    # create a series of similarity scores
    score_series = pd.Series(cosine_sim[idx])

    # sort the series by similarity score and get the top recommended courses
    if find_similar:
        top_sugg = list(score_series.sort_values(ascending=False).iloc[1:num_courses + 1].index)
    else:
        top_sugg = list(score_series.sort_values(ascending=True).iloc[:num_courses].index)

    # create a list of recommended courses
    recommended = df.loc[top_sugg, 'course_name'].tolist()

    return recommended


def content_based_recommendations(df, input_course, courses, num_courses):
    # filter out the courses
    df = df[df['course_name'].isin(courses)].reset_index()

    # create description keywords
    df['desc_keywords'] = extract_keywords(df, 'description')

    count = CountVectorizer()
    count_matrix = count.fit_transform(df['desc_keywords'])
    # cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # make the recommendation
    rec_courses_similar = recommendations(df, input_course, cosine_sim, num_courses, True)
    temp_sim = df[df['course_name'].isin(rec_courses_similar)]
    # rec_courses_dissimilar = recommendations(df, input_course, cosine_sim, num_courses, False)
    # temp_dissim = df[df['course_name'].isin(rec_courses_dissimilar)]

    st.header(":mag: :blue[RECOMMENDED COURSES] :books:")
    # top 5 courses
    st.header("Top " + str(num_courses) + " most similar courses")
    st.write(temp_sim)


def build_recommendation_model(df):

    st.header("Content-based Recommendation")
    st.sidebar.header("PREFERENCES SELECTION")

    st.write("CONTENT BASED RECOMMENDATION identifies courses that are comparable to a chosen course. "
             "Every course that has been filtered based on the learner's abilities in the preceding section is "
             "available to them.")
    st.write("Choose course from 'Select Course' dropdown on the sidebar")

    # filter by skills
    skills_avail = []
    for i in range(1000):
        skills_avail = skills_avail + df['skills'][i]
    skills_avail = list(set(skills_avail))
    skills_select = st.sidebar.multiselect("Select Skills", skills_avail)

    temp = filter_records(df, skills_select, 'skills', 'course_url')
    skill_filtered = df[df['course_url'].isin(temp)].reset_index()
    # update filtered courses
    courses = skill_filtered['course_name']

    st.header("Courses based on Skills selected")
    st.write(skill_filtered)

    # some more info
    st.write("**Number of programs filtered:**", skill_filtered.shape[0])
    st.write("**Number of professional degrees:**",
             skill_filtered[skill_filtered['learning_product_type'] == 'PROFESSIONAL CERTIFICATE'].shape[0])
    st.write("**Number of specializations:**",
             skill_filtered[skill_filtered['learning_product_type'] == 'SPECIALIZATION'].shape[0])
    st.write("**Number of courses:**",
             skill_filtered[skill_filtered['learning_product_type'] == 'COURSE'].shape[0])

    input_course = st.sidebar.selectbox("Select Course", courses, key='courses')
    num_courses_max = st.sidebar.slider('Select Max Courses to recommend', 0, 10, 1)

    # use button to initiate content-based recommendations
    rec_radio = st.sidebar.radio("Recommend Similar Courses", ('no', 'yes'), index=0)

    # basic plots
    chart = alt.Chart(skill_filtered).mark_bar().encode(
        y='course_provided_by:N',
        x='count(course_provided_by):Q',
    ).properties(
        title='Univs/Orgs providing these courses'
    )
    st.altair_chart(chart, use_container_width=True, theme='streamlit')

    # # there should be more than 2 courses
    # if len(courses) <= 2:
    #     st.write("*There should be at least 3 courses. Please add more.*")

    # Recommendation Engine when Radio Button is turned ON
    if rec_radio == 'yes':
        content_based_recommendations(df, input_course, courses, num_courses_max)


if __name__ == "__main__":
    st.title(":violet[Course Recommender Systems - DFP Capstone]")
    st.write("Exploring Courses on Coursera")
    st.sidebar.title("Set your Parameters")
    st.sidebar.header("Toggle this to show Raw data")
    st.header("About:")
    st.write("The Course Recommender is an easy-to-use tool designed to assist students in finding courses offered by "
             "Coursera using data-driven methods. Users can study and engage with the dataset's different features "
             "using the technology to choose their best options. Moreover, Course Recommender can recommend courses to "
             "students depending on their chosen learning styles.")

    # load data
    df = load_data()

    # toggle button to display raw data
    if st.sidebar.checkbox("Display raw data", key='disp_data'):
        st.write(df)
    else:
        pass

    build_recommendation_model(df)
