# Data Focused Python
DFP final project - Course Recommender Systems

# Course Recommender System using Coursera Data

-> Introduction

This is a course recommender system designed to help learners navigate through the courses on Coursera, aided by a data-driven strategy. The system is designed to help learners identify suitable courses for their learning preferences based on data-driven insights.

The system uses Coursera data to provide recommendations. It includes features for exploratory data analysis, visualization, scraping, and a recommendation algorithm based on the RAKE (NLTK) algorithm.

# Features

Data collection: Data is collected from a curated list of courses from coursera already present in the form of CSV. 
Exploratory Data Analysis (EDA): The data collected is also explored to identify patterns and relationships between the data points.
Data visualization: Visualization is used to communicate insights and to make the recommendations user-friendly and also tells some relationships between a few attributes in the dataset.
RAKE (NLTK) algorithm: The RAKE algorithm is used to extract keywords from course descriptions and identify the most relevant courses based on learner preferences.

# Installation - TBD 

Clone the repository: git clone https://github.com//course-recommender.git
Install the required dependencies using the following command: pip install -r requirements.txt
Run the app: streamlit run course_recommender.py

# Usage

When this command is run, it opens the application on localhost with port number 8501 
Specify the preferred course categories, select the skills and give the max no of courses to be taken and other relevant parameters.
Click on the "Yes" button under Recommendation on sidebar to generate a list of recommended courses.
