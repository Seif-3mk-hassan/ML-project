import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
DATA_PATH = 'E:\\ML project\\ML-project\\E-Learning Student Perfromance Prediction'
ASSESSMENT_DATA_PATH = os.path.join(DATA_PATH, 'assessments.csv')
COURSES_DATA_PATH = os.path.join(DATA_PATH, 'courses.csv')
STUDENTS_ASSESSMENTS_DATA_PATH = os.path.join(DATA_PATH, 'StudentAssesments.csv')
STUDENTS_INFO_DATA_PATH = os.path.join(DATA_PATH, 'studentinfo.csv')
STUDENTS_REGISTRATION_DATA_PATH = os.path.join(DATA_PATH, 'studentRegistration.csv')
STUDENTS_VLE_DATA_PATH = os.path.join(DATA_PATH, 'studentVle.csv')
VLE_DATA_PATH = os.path.join(DATA_PATH, 'vle.csv')

assessments = pd.read_csv(ASSESSMENT_DATA_PATH)
courses = pd.read_csv(COURSES_DATA_PATH)
students_assessments = pd.read_csv(STUDENTS_ASSESSMENTS_DATA_PATH)
students_info = pd.read_csv(STUDENTS_INFO_DATA_PATH)
students_registration = pd.read_csv(STUDENTS_REGISTRATION_DATA_PATH)
students_vle = pd.read_csv(STUDENTS_VLE_DATA_PATH)
vle = pd.read_csv(VLE_DATA_PATH)