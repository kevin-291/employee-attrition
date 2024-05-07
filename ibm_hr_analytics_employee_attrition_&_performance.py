import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import metrics
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.express as px
import plotly.figure_factory as ff
from warnings import filterwarnings

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

filterwarnings("ignore")

"""# Loading Dataset"""

df = pd.read_csv('/content/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()

"""# Data Analysis"""

df.info()

"""* There are a total of 1470 entries in the dataset.
* The dataset consists of 26 numeric columns and 9 categorical columns
"""

df.isnull().sum()

"""*  We can see that the data has no null values"""

categorical_columns = df.select_dtypes('object')
categorical_columns

numeric_columns = df.select_dtypes('int64')
numeric_columns

categorical_columns.describe().T

numeric_columns.describe().T

"""From the analysis, we can remove
* EmployeeCount, as all the values are 1
* StandardHours, as all the values are 1
* EmployeeNumber, as it is the number of the employee and wont matter much
* Over18, as all the values are Y
"""

col_to_drop = ['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18']
df = df.drop(col_to_drop, axis=1)

df.head()

cat = df.select_dtypes(include='object')
num = df.select_dtypes(include='number')

# For getting unique values of categorical columns
for i in cat:
    print(f'Unique values of {i} are {set(df[i])}')

"""# Data Visualization"""

hr = df.copy(deep=True)

"""### What is the relationship between attrition and age?"""

# Grouping by age and attrition status, counting daily rates, and resetting index

age_att = hr.groupby(['Age','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')

line_chart = px.line(age_att, x='Age', y='Counts', color='Attrition',
                     title='<b>Age Distribution within Organization with Attrition</b>', height=500,
                     color_discrete_map={'Yes': '#C53E4A', 'No': '#419D9D'})  # Specify line colors here

line_chart.update_traces(texttemplate='%{text:.2f}%',
                  textposition='top center',
                  textfont_size=14,
                  textfont_color='black',
                  marker=dict(line=dict(color='#28221D', width=1)))

line_chart.update_layout(title_x=0.5,
                  legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
                  font_color='#28221D',
                  paper_bgcolor='#EADFC7',
                  plot_bgcolor='#EADFC7')


line_chart.show()

"""The age group between 28-32 witnesses the highest attrition rate, indicating a critical phase where individuals may reassess their career paths. This trend gradually declines with advancing age, reflecting a growing emphasis on job stability and long-term commitments.
Conversely, the early career stages, notably between 18-20, often see heightened attrition as individuals explore different opportunities. This pattern reaches a turning point around the age of 21, marking a transition towards more stable employment decisions.

### Exploring the Influence of Income on Employee Attrition Rates
"""

# Grouping by monthly income and attrition status, counting occurrences, and resetting index
rate_att = hr.groupby(['MonthlyIncome', 'Attrition']).size().reset_index(name='Counts')

# Rounding monthly income to the nearest thousand
rate_att['MonthlyIncome'] = round(rate_att['MonthlyIncome'], -3)

# Grouping again by rounded monthly income and attrition status, counting occurrences, and resetting index
rate_att = rate_att.groupby(['MonthlyIncome', 'Attrition']).size().reset_index(name='Counts')

# Creating the line chart
line_chart = px.line(rate_att, x='MonthlyIncome', y='Counts', color='Attrition',
                     title='<b>Monthly Income-based Counts of People in an Organization</b>', height=500,
                     color_discrete_map={'Yes': '#C53E4A', 'No': '#419D9D'})  # Specify line colors here

line_chart.update_traces(texttemplate='%{text:.2f}%',
                  textposition='top center',
                  textfont_size=14,
                  textfont_color='black',
                  marker=dict(line=dict(color='#28221D', width=1)))

line_chart.update_layout(title_x=0.5,
                  legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
                  font_color='#28221D',
                  paper_bgcolor='#EADFC7',
                  plot_bgcolor='#EADFC7')
line_chart.show()

"""Examining the above chart reveals a significant increase in attrition rates at very low income levels, specifically below 5000 per month. This trend gradually decreases, with a slight increase observed around the 10000 mark, indicative of the middle-class livelihood.
Individuals in this income bracket often aspire to enhance their standards of living, leading them to seek new job opportunities. Conversely, as monthly income reaches a more comfortable level, the likelihood of employee turnover diminishes, as evidenced by the stable, flat line.

### Impact of Job Satisfaction on Employee Attrition with Average Monthly Salary
"""

# Map job satisfaction level to corresponding text labels
job_satisfaction_labels = {
    1: 'Low',
    2: 'Medium',
    3: 'High',
    4: 'Very High'
}

# Convert 'JobSatisfaction' column to text labels
hr['JobSatisfaction'] = hr['JobSatisfaction'].map(job_satisfaction_labels)

# Group data by 'JobSatisfaction' and 'Attrition', calculate average monthly income
avg_inc = hr.groupby(['JobSatisfaction', 'Attrition'])['MonthlyIncome'].mean().reset_index()

# Create Plotly plot
fig = px.bar(avg_inc, x='JobSatisfaction', y='MonthlyIncome', color='Attrition', barmode='group',
             title='<b>Average Income and Job Satisfaction by Attrition Status</b>',
             labels={'JobSatisfaction': 'Level of Job Satisfaction', 'MonthlyIncome': 'Average Monthly Income'},
             category_orders={'JobSatisfaction': sorted(hr['JobSatisfaction'].unique())}, height=500,
            color_discrete_map={'Yes': '#C53E4A', 'No': '#419D9D'})  # Specify custom colors

# Add data labels with adjusted positions
for trace in fig.data:
    if trace.name == 'Yes':
        for i, label in enumerate(trace.y):
            fig.add_annotation(x=trace.x[i], y=label, text=str(round(label, 2)),
                               showarrow=False, font=dict(color='black', size=12), yshift=10, xshift=45)
    else:
        for i, label in enumerate(trace.y):
            fig.add_annotation(x=trace.x[i], y=label, text=str(round(label, 2)),
                               showarrow=False, font=dict(color='black', size=12), yshift=10, xshift=-45)

# Add percentage labels to the bars
fig.update_traces(textposition='outside',
                  textfont_size=14,
                  textfont_color='black',
                  marker=dict(line=dict(color='#28221D', width=1)))

fig.update_layout(title_x=0.5,
                  legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.49),
                  font_color='#28221D',
                  paper_bgcolor='#EADFC7',
                  plot_bgcolor='#EADFC7')

fig.show()

"""In analyzing attrition trends, it's evident that individuals with lower levels of job satisfaction are more inclined to leave their positions. This trend is particularly noticeable among those earning an average monthly salary of $4596, indicating that dissatisfaction may drive employee turnover.

Conversely, individuals with higher satisfaction levels, especially those earning an average monthly salary of $6853, are more likely to remain with the company. This suggests that job satisfaction plays a crucial role in employee retention, with higher satisfaction levels mitigating the risk of attrition.

Overall, this insight underscores the importance of fostering a positive work environment and addressing factors contributing to job dissatisfaction to reduce attrition rates and retain valuable talent within the organization.

### Analysis of Attrition Rates Across Departments
"""

# Group data by Department and Attrition, count occurrences, and reset index
dept_att = hr.groupby(['Department', 'Attrition']).size().reset_index(name='Counts')

# Calculate total counts per department
dept_total_counts = dept_att.groupby('Department')['Counts'].transform('sum')

# Calculate percentage within each department
dept_att['Percentage'] = (dept_att['Counts'] / dept_total_counts) * 100

# Create a bar plot
fig = px.bar(dept_att, x='Department', y='Counts', color='Attrition',
             title='<b>Department-wise Distribution of Employees by Attrition Status</b>',
             text='Percentage', # Use 'Percentage' column as text
             labels={'Counts': 'Count', 'Percentage': 'Percentage'}, height = 500,
            color_discrete_map={'Yes': '#C53E4A', 'No': '#419D9D'})  # Specify custom colors

# Add percentage labels to the bars
fig.update_traces(texttemplate='%{text:.2f}%',
                  textposition='outside',
                  textfont_size=14,
                  textfont_color='black',
                  marker=dict(line=dict(color='#28221D', width=1)))

fig.update_layout(title_x=0.5,
                  legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.49),
                  font_color='#28221D',
                  paper_bgcolor='#EADFC7',
                  plot_bgcolor='#EADFC7')

fig.show()

"""This dataset includes three primary departments. Among them, the Sales department exhibits the highest attrition rate at 20.63%, trailed by the Human Resources Department at 19.05%. Conversely, the Research and Development department demonstrates the lowest attrition rate, indicating stability and contentment within the department, as depicted in the chart (13.84%).

### Attrition Rates Across Job Roles: A Hierarchy of Stability
"""

# Group data by JobRole and Attrition, calculate count and percentage
attr_job = hr.groupby(['JobRole', 'Attrition']).size().unstack(fill_value=0)
attr_job['Total'] = attr_job.sum(axis=1)
attr_job['Attrition_Rate'] = attr_job['Yes'] / attr_job['Total'] * 100

attr_job = attr_job.sort_values('Attrition_Rate')

fig = go.Figure()

# Add 'Yes' Attrition bars
fig.add_trace(go.Bar(y=attr_job.index,
                     x=attr_job['Attrition_Rate'],
                     name='Attrition: Yes',
                     orientation='h',
                     marker_color='#C53E4A'))

# Add 'No' Attrition bars
fig.add_trace(go.Bar(y=attr_job.index,
                     x=100 - attr_job['Attrition_Rate'],
                     name='Attrition: No',
                     orientation='h',
                     marker_color='#419D9D'))

fig.update_layout(
    title='<b>Attrition by Job Role</b>',
    title_x=0.5,
    xaxis_title='Attrition Rate (%)',
    yaxis_title='Job Role',
    barmode='relative',
    bargap=0.1,
    legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center',  x=0.50),
    height=500,
    font_color='#28221D',
    paper_bgcolor='#EADFC7',
    plot_bgcolor='#EADFC7'
)

fig.show()

"""The analysis suggests that higher-level job roles within the organization demonstrate lower attrition rates compared to lower-level roles. Positions such as manufacturing directors, healthcare representatives, managers, and research directors exhibit notably lower attrition rates, indicating that individuals in these roles are less likely to leave the company.

Conversely, roles at lower organizational levels, such as sales representatives, laboratory technicians, and human resources personnel, demonstrate higher attrition rates. This insight implies that individuals in higher-level job roles tend to stay with the company more consistently, contributing to a more stable workforce.

### The Impact of Salary Hikes on Employee Retention
"""

hike_att = hr.groupby(['PercentSalaryHike', 'Attrition']).apply(lambda x: x['DailyRate'].count()).reset_index(name='Counts')

line_plot = px.line(hike_att, x='PercentSalaryHike', y='Counts', color='Attrition',
                    title='<b>Distribution of Salary Hike Percentages Among Employees</b>', height=500,
                    color_discrete_map={'Yes': '#C53E4A', 'No': '#419D9D'})

line_plot.update_traces(texttemplate='%{text:.2f}%',
                  textposition='top center',
                  textfont_size=14,
                  textfont_color='black',
                  marker=dict(line=dict(color='#28221D', width=1)))

line_plot.update_layout(title_x=0.5,
                  legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
                  font_color='#28221D',
                  paper_bgcolor='#EADFC7',
                  plot_bgcolor='#EADFC7')

line_plot.show()

"""Enhanced salary increments serve as a significant incentive for employees, encouraging improved performance and fostering retention within the organization. Consequently, the likelihood of an employee departing from an organization offering lower salary hikes is considerably higher compared to one providing substantial salary increments.

### Examining Work Experience Diversity Among Employees
"""

exp_att = hr.groupby(['NumCompaniesWorked', 'Attrition']).size().reset_index(name='Counts')

area_plot = px.area(exp_att, x='NumCompaniesWorked', y='Counts', color='Attrition',
                    title='<b>Distribution of Work Experience Levels Among Employees</b>', line_shape='spline', height=500,
                    color_discrete_map={'Yes': '#C53E4A', 'No': '#419D9D'})

area_plot.update_traces(texttemplate='%{text:.2f}%',
                  textposition='top center',
                  textfont_size=14,
                  textfont_color='black',
                  marker=dict(line=dict(color='#28221D', width=1)))

area_plot.update_layout(title_x=0.5,
                  legend=dict(orientation='h', yanchor='bottom', y=-0.4, xanchor='center', x=0.5),
                  font_color='#28221D',
                  paper_bgcolor='#EADFC7',
                  plot_bgcolor='#EADFC7')

area_plot.update_layout(title_x=0.5, legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5))

area_plot.show()

"""The chart illustrates a notable trend: individuals who begin their careers with the company or transition to it early on are more likely to seek opportunities elsewhere. Conversely, those with extensive experience across multiple companies tend to exhibit greater loyalty to their current employer.

### Effect of Salary Hikes on Employee Retention and Motivation
"""

promotion_att = hr.groupby(['PercentSalaryHike', 'Attrition']).apply(lambda x: x['DailyRate'].count()).reset_index(name='Counts')

fig = px.line(promotion_att, x='PercentSalaryHike', y='Counts', color='Attrition',
              title='<b>Percent Salary Hike of Employees in the Organization</b>', height=500,
              color_discrete_map={'Yes': '#C53E4A', 'No': '#419D9D'})  # Specify line colors here

fig.update_traces(texttemplate='%{text:.2f}%',
                  textposition='top center',
                  textfont_size=14,
                  textfont_color='black',
                  marker=dict(line=dict(color='#28221D', width=1)))

fig.update_layout(title_x=0.5,
                  legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
                  font_color='#28221D',
                  paper_bgcolor='#EADFC7',
                  plot_bgcolor='#EADFC7')

fig.show()

"""Increased salary raises inspire individuals to perform more effectively and remain committed to the organization. Consequently, we observe that the likelihood of an employee departing from a company with lower salary increments is significantly higher compared to one that offers substantial raises.

### Attrition Rates Across Education Levels
"""

import plotly.express as px

education_labels = {
    1: 'Below College',
    2: 'College',
    3: 'Bachelor',
    4: 'Master',
    5: 'Doctor'
}

# Group data by Education and Attrition, count occurrences, and reset index
dept_att = hr.groupby(['Education', 'Attrition']).size().reset_index(name='Counts')

# Calculate total counts per Education
dept_total_counts = dept_att.groupby('Education')['Counts'].transform('sum')

# Calculate percentage within each Education
dept_att['Percentage'] = (dept_att['Counts'] / dept_total_counts) * 100

# Replace education codes with labels
dept_att['Education'] = dept_att['Education'].map(education_labels)

# Create a bar plot
fig = px.bar(dept_att, x='Education', y='Counts', color='Attrition',
             title='<b>Education-wise Distribution of Employees by Attrition Status</b>',
             text='Percentage', # Use 'Percentage' column as text
             labels={'Counts': 'Count', 'Percentage': 'Percentage'}, height=550, # Update axis labels
             color_discrete_map={'Yes': '#C53E4A', 'No': '#419D9D'})  # Specify custom colors

# Add percentage labels to the bars
fig.update_traces(texttemplate='%{text:.2f}%',
                  textposition='outside',
                  textfont_size=14,
                  textfont_color='black',
                  marker=dict(line=dict(color='#28221D', width=1)))

fig.update_layout(title_x=0.5,
                  legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.49),
                  font_color='#28221D',
                  paper_bgcolor='#EADFC7',
                  plot_bgcolor='#EADFC7')

fig.show()

"""The graph indicates that individuals with education levels below college have the highest attrition rate at 18.24%. This may be because they face limited opportunities for career advancement or feel dissatisfied with roles that don't fully utilize their educational background.

Following closely, employees with bachelor's degrees experience a 17.31% attrition rate. This could be due to aspirations for career growth, seeking better compensation, or exploring opportunities in other organizations.

Similarly, employees with college diplomas face a 15.60% attrition rate. Factors such as job fit, workplace culture, or external market conditions may contribute to their decision to leave.

In contrast, individuals with master's degrees demonstrate a lower attrition rate of 14.57%. Their specialized skills and advanced qualifications make them valuable to their employers, reducing their inclination to seek opportunities elsewhere.

Lastly, employees with doctorate degrees exhibit the lowest attrition rate at 10.42%. Their extensive expertise and deep knowledge in their field contribute to high job satisfaction and a strong commitment to their work. Additionally, the limited availability of positions matching their expertise in the job market further reduces attrition among this group.
"""

# univariate analysis of categorical data:
sns.set(rc={"axes.facecolor":"white","figure.facecolor":"#9ed9cd"})
sns.set_palette("pastel")
for i, col in enumerate(cat):

    fig, axes = plt.subplots(1,2,figsize=(10,5))

    # count of col (countplot)

    ax=sns.countplot(data=df, x=col, ax=axes[0])
    activities = [var for var in df[col].value_counts().sort_index().index]
    ax.set_xticklabels(activities,rotation=90)
    for container in axes[0].containers:
        axes[0].bar_label(container)

    #count of col (pie chart)

    index = df[col].value_counts().index
    size = df[col].value_counts().values
    explode = (0.05, 0.05)

    axes[1].pie(size, labels=index,autopct='%1.1f%%', pctdistance=0.85)

    # Inner circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.suptitle(col,backgroundcolor='black',color='white',fontsize=15)

    plt.show()

plt.figure(figsize = (15,25))
for idx, i in enumerate(num):
    plt.subplot(12, 2, idx + 1)
    sns.boxplot(x = i, data = df)
    plt.title(i,backgroundcolor='black',color='white',fontsize=15)
    plt.xlabel(i, size = 12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,10))
sns.relplot(data=df, y="MonthlyIncome", x="Age", hue="Attrition",col='Gender')
plt.show()

plt.figure(figsize=(5,10))
sns.relplot(data=df, y="MonthlyIncome", x="MaritalStatus", hue="Attrition",col='Gender')
plt.show()

plt.figure(figsize=(5,10))
sns.relplot(data=df, y="MonthlyIncome", x="Department", hue="Attrition",col='Gender')
plt.show()

plt.figure(figsize=(5,10))
sns.relplot(data=df, y="MonthlyIncome", x="Education", hue="Attrition",col='Gender')
plt.show()

plt.figure(figsize=(5,10))
ax=sns.relplot(data=df, y="MonthlyIncome", x="JobRole", hue="Attrition",col='Gender')
rotation = 90
for i, ax in enumerate(ax.fig.axes):   ## getting all axes of the fig object
     ax.set_xticklabels(ax.get_xticklabels(), rotation = rotation)
#plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(5,10))
sns.relplot(data=df, y="MonthlyIncome", x="BusinessTravel", hue="Attrition",col='Gender')
plt.xticks(rotation=90)
plt.show()

"""From the graphs, we can analyse that
* Attrition is the highest for both men and women from 18 to 35 years of age and gradually decreases.
* As income increases, attrition decreases.
* Attrition is much, much less in divorced women.
* Attrition is higher for employees who usually travel than others, and this rate is higher for women than for men.
* Attrition is the highest for those in level 1 jobs.
* Women with the job position of manager, research director and technician laboratory have almost no attrition.
* Men with the position of sales expert have a lot of attrition.

# Data Preprocessing
"""

df_copy = df.copy()

#convert category attributes with only 2 distinct values to numeric by assigning labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_copy['Attrition'] = le.fit_transform(df['Attrition'])
df_copy['OverTime'] = le.fit_transform(df['OverTime'])
df_copy['Gender'] = le.fit_transform(df['Gender'])

#convert category attributes with more than 2 distinct values to numeric using one-hot encoding
df_copy = pd.get_dummies(df_copy, columns=['BusinessTravel', 'Department', 'EducationField',
                               'JobRole', 'MaritalStatus'])

plt.figure(figsize=(20,10))
correlations=df_copy.corr()
correlations['Attrition'].sort_values(ascending = False).plot(kind='bar')

df2 = df.select_dtypes('int64')
df2_copy = df2.copy()


corr = df2_copy.corr(method = "spearman")
sns.set(style="white")

mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12, 10), dpi = 100)
sns.heatmap(corr, mask = mask, cmap= "winter", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot = True, fmt = ".2f")
plt.show()

"""There are high correlation between some features:

* Monthly Income & Job Level
* year in current role, year at company & year with current manager

# Model Development
"""

DF = df.copy()

# Performing ordinal encoding to categorical variables

DF['BusinessTravel'] = DF['BusinessTravel'].replace('Travel_Rarely',2)
DF['BusinessTravel'] = DF['BusinessTravel'].replace('Travel_Frequently',3)
DF['BusinessTravel'] = DF['BusinessTravel'].replace('Non-Travel',4)

DF['Attrition'] = DF['Attrition'].replace('Yes',2)
DF['Attrition'] = DF['Attrition'].replace('No',3)

DF['OverTime'] = DF['OverTime'].replace('Yes',2)
DF['OverTime'] = DF['OverTime'].replace('No',3)

DF['Gender'] = DF['Gender'].replace('Male',2)
DF['Gender'] = DF['Gender'].replace('Female',3)

DF['MaritalStatus'] = DF['MaritalStatus'].replace('Single',2)
DF['MaritalStatus'] = DF['MaritalStatus'].replace('Married',3)
DF['MaritalStatus'] = DF['MaritalStatus'].replace('Divorced',4)

DF['Department'] = DF['Department'].replace('Sales',2)
DF['Department'] = DF['Department'].replace('Human Resources',3)
DF['Department'] = DF['Department'].replace('Research & Development',4)

DF['EducationField'] = DF['EducationField'].replace('Life Sciences',2)
DF['EducationField'] = DF['EducationField'].replace('Medical',3)
DF['EducationField'] = DF['EducationField'].replace('Marketing',4)
DF['EducationField'] = DF['EducationField'].replace('Technical Degree',2)
DF['EducationField'] = DF['EducationField'].replace('Human Resources',3)
DF['EducationField'] = DF['EducationField'].replace('Other',4)

DF['JobRole'] = DF['JobRole'].replace('Sales Executive',2)
DF['JobRole'] = DF['JobRole'].replace('Manufacturing Director',3)
DF['JobRole'] = DF['JobRole'].replace('Healthcare Representative',4)
DF['JobRole'] = DF['JobRole'].replace('Manager',2)
DF['JobRole'] = DF['JobRole'].replace('Research Director',3)
DF['JobRole'] = DF['JobRole'].replace('Laboratory Technician',4)
DF['JobRole'] = DF['JobRole'].replace('Sales Representative',2)
DF['JobRole'] = DF['JobRole'].replace('Research Scientist',3)
DF['JobRole'] = DF['JobRole'].replace('Human Resources',4)

DF = DF.drop(['MonthlyIncome' ,'YearsInCurrentRole' , 'YearsAtCompany', 'YearsWithCurrManager'],axis=1)

# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
DF1 = DF.drop(columns=['Attrition'])
norm = scaler.fit_transform(DF)
norm_df = pd.DataFrame(norm,columns=DF.columns)

X = pd.DataFrame(norm_df.drop(columns='Attrition'))
y = pd.DataFrame(norm_df.Attrition).values.reshape(-1, 1)

# Splitting into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""In the given problem, we found out that the number of females and males are greatly imbalanced. This will lead to a major problem while building the model. To overcome this, we use oversampling. Here, we will be using SMOTE, a popular technique for oversampling."""

#SMOTE
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_resample(X_train,y_train)

# Checking shape of original data and data after SMOTE
print(X_train.shape)
print(smote_train.shape)

# Helper function to plot confusion matrix and display accuracy
def plot(predictions):
    fig, ax = plt.subplots(figsize=(10,5))
    cm = metrics.confusion_matrix(y_test,predictions)
    sns.heatmap(metrics.confusion_matrix(y_test,predictions),annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix', y=1.1)
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.xlabel('y prediction')
    plt.ylabel('y actual')
    plt.show()

    print(metrics.classification_report(y_test, predictions))

"""## 1. Logistic Regression"""

from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(C=1000,max_iter=10000)
log_reg.fit(smote_train, smote_target)
y_pred_lg = log_reg.predict(X_test)

log_reg_acc = metrics.accuracy_score(y_test, y_pred_lg)
print ('accuracy',log_reg_acc)

plot(y_pred_lg)

"""## 2. Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rfc = RandomForestClassifier()
rfc = rfc.fit(smote_train, smote_target)
y_pred = rfc.predict(X_test)

rfc_acc = metrics.accuracy_score(y_test, y_pred)
print ('accuracy',rfc_acc)

plot(y_pred)

"""## 3. Gradient Boosting Classifier"""

seed=0
gb_params ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.25,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0}

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(**gb_params)
gb.fit(smote_train, smote_target)
gb_predictions = gb.predict(X_test)

gb_acc = metrics.accuracy_score(y_test, gb_predictions)
print('accuracy', gb_acc)

plot(gb_predictions)

"""## 4. AdaBoost Classifier"""

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada = ada.fit(smote_train, smote_target)
ada_predictions = ada.predict(X_test)

ada_acc = metrics.accuracy_score(y_test, ada_predictions)
print('accuracy',ada_acc)

plot(ada_predictions)

"""## 5. Gaussian Process Classifier"""

from sklearn.gaussian_process import GaussianProcessClassifier
gpc = GaussianProcessClassifier()
gpc = gpc.fit(smote_train, smote_target)
gpc_pred = gpc.predict(X_test)

gpc_acc = metrics.accuracy_score(y_test, gpc_pred)
print('accuracy',gpc_acc)

plot(gpc_pred)

# Adding the scores to a dataframe

model_scores = {
    "Model" : ['Logistic Regression', 'Random Forest Classifier', 'Gradient Boosting Classifier', 'AdaBoost Classifier', 'Gaussian Process Classifier'],
    "Accuracy" : [log_reg_acc, rfc_acc, gb_acc, ada_acc, gpc_acc]
}

model_details = pd.DataFrame(model_scores)
model_details_sorted = model_details.sort_values('Accuracy', ascending=False)
model_details_sorted.head()

"""Hence, we can see that both Random Forest and Gradient Boosting Classifiers provide a high level of accuracy

### Key Findings
* Gender Disparity: Males exhibit a higher attrition rate compared to females, hinting at potential disparities in job satisfaction, career opportunities, and workplace environment.

* Age Dynamics: Attrition rates vary across different age groups, with individuals between 28-32 experiencing the highest attrition. This trend declines with advancing age, indicating a shift towards job stability and long-term commitments as individuals progress in their careers.

* Income Levels: Attrition rates are influenced by income levels, with significant spikes observed at very low income levels and a gradual decrease as income rises. This underscores the importance of financial stability in employee retention.

* Job Satisfaction: Lower levels of job satisfaction correlate with higher attrition rates, particularly among employees with average monthly salaries of 4596. Conversely, higher satisfaction levels, especially among those earning 6853, contribute to employee retention.

* Departmental Differences: The Sales department exhibits the highest attrition rate, followed by Human Resources, while Research and Development demonstrate lower rates. This suggests variations in work culture, opportunities, and satisfaction levels across departments.

* Job Role Impact: Higher-level job roles show lower attrition rates compared to lower-level roles, indicating the importance of career advancement opportunities and job stability in retaining talent.

* Salary Increment Influence: Enhanced salary increments serve as a significant incentive for retention, motivating employees to perform better and remain committed to the organization.

* Educational Background: Individuals with higher education levels, such as master's and doctorate degrees, demonstrate lower attrition rates, highlighting the value of specialized skills and advanced qualifications in job satisfaction and retention.

* Salary and Stock Options: Salary and stock options serve as significant motivators for employees, leading to higher loyalty and reduced attrition rates. Employees who receive higher pay and more stock options are more likely to remain committed to their organization, highlighting the importance of competitive compensation packages in retaining talent.

* Work-Life Balance: Work-life balance emerges as a crucial factor influencing employee motivation and retention. While a good work-life balance is often considered a motivation factor, it can also lead employees to seek better opportunities and a higher standard of living elsewhere. Balancing work demands with personal life priorities is essential for maintaining employee satisfaction and reducing turnover.

### Other observations include:

* Single employees demonstrate a higher rate of departure compared to their married and divorced counterparts.

* Approximately 10% of employees leave upon reaching their 2-year anniversary with the company.

* Employees who are loyal, hold higher salaries, and assume more responsibilities exhibit a lower likelihood of leaving compared to their peers.

* Individuals residing farther away from their workplace exhibit a higher likelihood of leaving compared to those who live closer.

* Employees who frequently travel for work display a higher propensity to leave compared to their counterparts.

* Those required to work overtime demonstrate a higher likelihood of leaving compared to those who do not.

* Sales representatives comprise a significant proportion of leavers within the dataset.

* Employees with a history of working at multiple companies in the past exhibit a higher likelihood of leaving compared to their counterparts.

### Recommendations

1. Ensure equal opportunities for all genders, addressing potential disparities in job satisfaction and career advancement.

2. Provide support and development opportunities for employees in their late twenties to early thirties to improve retention during this pivotal career phase.

3. Adjust salary structures to offer competitive compensation, particularly for lower income levels, enhancing financial stability and reducing turnover.

4. Prioritize initiatives to enhance job satisfaction, such as recognition programs and skill development opportunities.

5. Assess and address factors contributing to higher attrition in departments like Sales and Human Resources, improving workload management and work culture.

6. Provide clear career paths and skill development opportunities for lower-level roles to increase job stability.

7. Review and optimize salary increment policies and offer competitive compensation packages, including stock options, to motivate loyalty.

8. Implement policies promoting work-life balance, such as flexible work arrangements and wellness programs, to improve satisfaction and reduce turnover.

9. Address specific factors such as support for single employees, workload management for anniversary dates, and challenges faced by employees who live far or travel frequently.

10. Implementing these strategies will help mitigate attrition rates and retain valuable talent, contributing to long-term organizational success.
"""

