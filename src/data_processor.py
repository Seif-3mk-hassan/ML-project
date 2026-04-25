import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
import os
warnings.filterwarnings('ignore')
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



# ===========================================================
# PART 1: PREPROCESSING + ADVANCED FEATURE ENGINEERING
# ===========================================================
print("=" * 70)
print("  PART 1: PREPROCESSING + ADVANCED FEATURES")
print("=" * 70)

# -- Load --
print("\n[1.1] Loading...")
SA = pd.read_csv(STUDENTS_ASSESSMENTS_DATA_PATH)
SI = pd.read_csv(STUDENTS_INFO_DATA_PATH)
ASS = pd.read_csv(ASSESSMENT_DATA_PATH)

SA['score'] = pd.to_numeric(SA['score'].replace('?', np.nan), errors='coerce')
SA = SA.dropna(subset=['score'])
print(f"  Assessments: {SA.shape[0]} rows")

# -- Merge assessment metadata --
print("[1.2] Merging metadata...")
df = SA.merge(
    ASS[['id_assessment', 'code_module', 'code_presentation',
         'date', 'weight', 'assessment_type']],
    on='id_assessment', how='left')
df.rename(columns={'date': 'deadline_date'}, inplace=True)
df['deadline_date'] = pd.to_numeric(df['deadline_date'], errors='coerce')
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

# -- Timing --
print("[1.3] Timing features...")
df['days_before_deadline'] = df['deadline_date'] - df['date_submitted']
df['is_late'] = (df['days_before_deadline'] < 0).astype(int)
df['abs_days_from_deadline'] = df['days_before_deadline'].abs()

# -- Encode StudentInfo --
print("[1.4] Encoding StudentInfo...")
si = SI.copy()
edu_order = ['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent',
             'HE Qualification', 'Post Graduate Qualification']
si['highest_education_ord'] = si['highest_education'].astype(
    pd.CategoricalDtype(categories=edu_order, ordered=True)).cat.codes.replace(-1, np.nan)

imd_order = ['0-10%', '10-20', '10-20%', '20-30%', '30-40%',
             '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
si['imd_band_ord'] = si['imd_band'].astype(
    pd.CategoricalDtype(categories=imd_order, ordered=True)).cat.codes.replace(-1, np.nan)

si['age_band_ord'] = si['age_band'].astype(
    pd.CategoricalDtype(categories=['0-35', '35-55', '55<='], ordered=True)).cat.codes.replace(-1, np.nan)
si['disability_bin'] = si['disability'].map({'Y': 1, 'N': 0})
si['gender_bin'] = si['gender'].map({'M': 1, 'F': 0})

si_cols = ['code_module', 'code_presentation', 'id_student',
           'highest_education_ord', 'imd_band_ord', 'age_band_ord',
           'disability_bin', 'gender_bin', 'num_of_prev_attempts', 'studied_credits']
df = df.merge(si[si_cols], on=['code_module', 'code_presentation', 'id_student'], how='left')

# -- VLE total features --
print("[1.5] VLE total features...")
vle_parts = []
for i, chunk in enumerate(pd.read_csv(STUDENTS_VLE_DATA_PATH, chunksize=500_000)):
    agg = chunk.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        total_clicks=('sum_click', 'sum'),
        total_vle_interactions=('sum_click', 'count'),
        unique_active_days=('date', 'nunique'),
        first_date=('date', 'min'),
        last_date=('date', 'max'),
    ).reset_index()
    vle_parts.append(agg)
    if (i + 1) % 10 == 0:
        print(f"  chunk {i+1}")

vle = pd.concat(vle_parts).groupby(
    ['code_module', 'code_presentation', 'id_student']).agg(
    total_clicks=('total_clicks', 'sum'),
    total_vle_interactions=('total_vle_interactions', 'sum'),
    unique_active_days=('unique_active_days', 'sum'),
    first_date=('first_date', 'min'),
    last_date=('last_date', 'max'),
).reset_index()
vle['activity_span_days'] = vle['last_date'] - vle['first_date']
vle['avg_clicks_per_day'] = vle['total_clicks'] / vle['unique_active_days'].replace(0, np.nan)
vle.drop(columns=['first_date', 'last_date'], inplace=True)

df = df.merge(vle, on=['code_module', 'code_presentation', 'id_student'], how='left')
for c in ['total_clicks', 'total_vle_interactions', 'unique_active_days',
          'activity_span_days', 'avg_clicks_per_day']:
    df[c] = df[c].fillna(0)

# -- Per-assessment VLE clicks --
print("[1.6] Per-assessment VLE clicks (before deadline)...")
vle_daily_parts = []
for i, chunk in enumerate(pd.read_csv(STUDENTS_VLE_DATA_PATH, chunksize=500_000)):
    daily = chunk.groupby(['code_module', 'code_presentation', 'id_student', 'date']).agg(
        day_clicks=('sum_click', 'sum')).reset_index()
    vle_daily_parts.append(daily)
    if (i + 1) % 10 == 0:
        print(f"  chunk {i+1}")

vle_daily = pd.concat(vle_daily_parts).groupby(
    ['code_module', 'code_presentation', 'id_student', 'date']).agg(
    day_clicks=('day_clicks', 'sum')).reset_index()

keys = df[['code_module', 'code_presentation', 'id_student', 'deadline_date']].drop_duplicates()
joined = keys.merge(vle_daily, on=['code_module', 'code_presentation', 'id_student'])
pre = joined[joined['date'] <= joined['deadline_date']]
pre_agg = pre.groupby(['code_module', 'code_presentation', 'id_student', 'deadline_date']).agg(
    clicks_before_deadline=('day_clicks', 'sum'),
    active_days_before_deadline=('date', 'nunique'),
).reset_index()

df = df.merge(pre_agg, on=['code_module', 'code_presentation', 'id_student', 'deadline_date'], how='left')
df['clicks_before_deadline'] = df['clicks_before_deadline'].fillna(0)
df['active_days_before_deadline'] = df['active_days_before_deadline'].fillna(0)

# -- Prior score features --
print("[1.7] Prior score features...")
df = df.sort_values(['code_module', 'code_presentation', 'id_student',
                     'deadline_date', 'date_submitted']).reset_index(drop=True)

g = df.groupby(['code_module', 'code_presentation', 'id_student'])
df['assessment_number'] = g.cumcount()
df['prior_mean_score'] = g['score'].transform(lambda x: x.expanding().mean().shift(1))
df['prior_min_score'] = g['score'].transform(lambda x: x.expanding().min().shift(1))
df['prior_max_score'] = g['score'].transform(lambda x: x.expanding().max().shift(1))
df['prior_last_score'] = g['score'].shift(1)
df['prior_std_score'] = g['score'].transform(lambda x: x.expanding().std().shift(1))

score_median = df['score'].median()
for c in ['prior_mean_score', 'prior_min_score', 'prior_max_score',
          'prior_last_score', 'prior_std_score']:
    df[c] = df[c].fillna(score_median if 'std' not in c else 0)

# -- NEW: Interaction features --
print("[1.8] Interaction features...")
df['clicks_x_days_before'] = df['clicks_before_deadline'] * df['days_before_deadline'].clip(lower=0)
df['active_ratio'] = df['active_days_before_deadline'] / df['deadline_date'].replace(0, np.nan).fillna(1)
df['click_intensity'] = df['clicks_before_deadline'] / df['active_days_before_deadline'].replace(0, np.nan).fillna(1)
df['late_x_clicks'] = df['is_late'] * df['total_clicks']
df['edu_x_clicks'] = df['highest_education_ord'].fillna(0) * df['clicks_before_deadline']
df['prior_score_x_clicks'] = df['prior_mean_score'] * df['clicks_before_deadline']
df['submission_consistency'] = df['prior_std_score']  # already computed

# -- NEW: Log-transform skewed features --
print("[1.9] Log-transforming skewed features...")
for c in ['total_clicks', 'total_vle_interactions', 'clicks_before_deadline',
          'click_intensity', 'clicks_x_days_before']:
    df[f'{c}_log'] = np.log1p(df[c].clip(lower=0))

# -- One-hot encode --
print("[1.10] One-hot encoding...")
atype_dum = pd.get_dummies(df['assessment_type'], prefix='atype', dtype=int)
mod_dum = pd.get_dummies(df['code_module'], prefix='mod', dtype=int)
pres_dum = pd.get_dummies(df['code_presentation'], prefix='pres', dtype=int)
df = pd.concat([df, atype_dum, mod_dum, pres_dum], axis=1)

# -- Handle missing --
print("[1.11] Handling missing...")
for c in ['imd_band_ord', 'highest_education_ord', 'age_band_ord']:
    df[c] = df[c].fillna(df[c].median())
df['days_before_deadline'] = df['days_before_deadline'].fillna(0)
df['weight'] = df['weight'].fillna(0)

# -- Define features --
print("[1.12] Defining features...")

feature_cols = (
    # Demographic
    ['num_of_prev_attempts', 'studied_credits', 'disability_bin',
     'gender_bin', 'imd_band_ord', 'highest_education_ord', 'age_band_ord']
    # VLE total
    + ['total_clicks', 'total_vle_interactions', 'unique_active_days',
       'activity_span_days', 'avg_clicks_per_day']
    # VLE per-assessment
    + ['clicks_before_deadline', 'active_days_before_deadline']
    # Prior scores
    + ['assessment_number', 'prior_mean_score', 'prior_min_score',
       'prior_max_score', 'prior_last_score', 'prior_std_score']
    # Interactions
    + ['clicks_x_days_before', 'active_ratio', 'click_intensity',
       'late_x_clicks', 'edu_x_clicks', 'prior_score_x_clicks']
    # Log-transforms
    + ['total_clicks_log', 'total_vle_interactions_log',
       'clicks_before_deadline_log', 'click_intensity_log',
       'clicks_x_days_before_log']
    # Timing
    + ['days_before_deadline', 'is_late', 'abs_days_from_deadline']
    # Assessment
    + ['weight']
    + list(atype_dum.columns)
    # Module & presentation
    + list(mod_dum.columns)
    + list(pres_dum.columns)
)

target_col = 'score'