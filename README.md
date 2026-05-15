# QAC387-Group-Repo
QAC387 group repository; Members: Shital, Anh, Nirvaan

Our agent is a data analysis assistant that supports individuals' in understanding the relationship between their smartphone usage, sleep and stress. It is able to handle responding to commands such as schema, ask...or generate code based on a command. 

# Motivation
Inspired by the research done at the Sleep and Psychological Adjustment Lab (S.P.A. Lab) at Wesleyan, we wanted to create an agent that can help people without a background in data analysis or sleep research to understand how their phones may be impacting their sleep and therefore their quality of life. With the help of our agent, the hopeful aim is that you do not need a background in research or data analysis to understand research-based outcomes. 

# Dataset
Our dataset is publicly available through Kaggle, called the Screen Time, Sleep & Stress Analysis Dataset with 15,000 entries and 13 columns. The dataset has already been cleaned and processed, specifically with no missing values, a clean structured format, and balanced categorical features.

# Tools
Our agent has a wide collection of tools for exploring the dataset (summarize, basic profile), running regressions (multi linear regression, predict) and generating plots (bar chart, histogram)

# Instructions 
To use, clone this repo and create a virtual Python environment. Download dependencies from the requirements.txt file and set up the OpenAI API key in the .env file (delete before committing changes!). Then run the agent. 

# Example prompts
- "schema"
- "help me analyze this dataset"
- "run a multiple linear regression for sleep quality using sleep duration and stress level as predictors"

