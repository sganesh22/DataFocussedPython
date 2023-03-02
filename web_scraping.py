# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:08:19 2023

@author: sanjana ganesh
@author: prasiddha sudhakar
@author: sanika hadatgune
@author: sathiya narayan chakravarthy
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://www.hostinger.com/tutorials/learn-coding-online-for-free"

# Make a GET request to the URL
response = requests.get(url)

# Create a BeautifulSoup object from the response text
soup = BeautifulSoup(response.text, "html.parser")

# Find all the h3 tags in the HTML
h3_tags = soup.find_all("h3")
website = []
url = []
description = []
courseNames = []
courses = []
    
def getWebsites():
    i = 0
    # Loop through the h3 tags and print the text content
    for tag in h3_tags:
        if (i < 108):
            website.append(tag.text.split(".")[1].strip())
            i += 1
            
def getURL():
    i = 0
    # Loop through the h3 tags and find the <a> tag inside each tag
    for tag in h3_tags:
        if (i != len(h3_tags)-1):
            a_tag = tag.find("a")
            i += 1
            if a_tag is not None:
                url.append(a_tag["href"])

def getDesc():
    i = 0
    # Loop through the h3 tags and find the first <p> tag after each tag
    for tag in h3_tags:
        if (i < 108):
            p_tag = tag.find_next_sibling("p")
            if p_tag is not None:
                description.append(p_tag.text.strip())
                i += 1

def getCourseNames():
    # Loop through the h3 tags and find the first <ul> tag after each tag
    for tag in h3_tags:
        ul_tag = tag.find_next_sibling("ul")
        if ul_tag is not None:
            courses = []
            for li_tag in ul_tag.find_all("li"):
                courses.append(li_tag.text.strip())
            courseNames.append(courses)
    
getWebsites()
getURL()       
getDesc()
getCourseNames()
df = pd.DataFrame({'Course URL': url,
                           'Websites': website,
                          'Description': description,
                          'Course names': courseNames})

df.to_csv('courses.csv')
