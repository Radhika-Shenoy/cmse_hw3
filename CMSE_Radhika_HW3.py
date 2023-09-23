import pandas as pd
import streamlit as st 
import plotly as plt
diabetes_df = pd.read_csv('diabetes.csv')
st.title("Diabetes dataset")
import matplotlib.pyplot as plt
st.header('Histogram')
class Histogram():
    def draw_histogram(self,data=None):
      fig = plt.figure(figsize=(10,6)); ax = fig.gca()
      diabetes_df.hist(bins=30, ax=ax)
      plt.suptitle('Student Performance', y=1.03)    # this adds a "super" title and places it well
      plt.tight_layout()
      return fig
f = Histogram()
st.pyplot(f.draw_histogram(diabetes_df))

import seaborn as sns
dd = diabetes_df[['Glucose','Insulin','Age','Pregnancies','Outcome']]

st.header("Pairplot")
st.pyplot(sns.pairplot(dd,hue = 'Outcome'))

st.header("Relationship plot")
st.pyplot(sns.relplot(data=diabetes_df, x="Pregnancies", y="Glucose", hue="Outcome"))

st.header("Scatter plot 1")
st.pyplot(sns.scatterplot(data=diabetes_df, x="Pregnancies", y="BMI", hue="Outcome").figure)

st.header("Scatter plot 2")
st.pyplot(sns.scatterplot(data=diabetes_df, x="Outcome", y="Insulin", hue="Pregnancies").figure)


# In[64]:


import seaborn as sns
import pandas as pd
plt.figure(figsize=(13,13))
sns.set(font_scale=1.0)
st.header("Heatmap")
st.pyplot(sns.heatmap(diabetes_df.corr(),annot=True,cmap = 'coolwarm').figure)


st.header("Distribution plot")
st.pyplot(sns.displot(diabetes_df, x="Pregnancies", hue="Outcome").figure)

st.header("Categorical plot")
st.pyplot(sns.catplot(x='Pregnancies',y='Insulin',data=diabetes_df, hue="Outcome").figure)

st.header("Box plot")
st.pyplot(sns.boxplot(x='Pregnancies',y='Age',hue='Outcome',data=diabetes_df).figure)


st.title("Student Performance dataset")


student_df = pd.read_csv('student_data.csv')
student_df.head(10)

st.header('Histogram')

class Histogram():
    def draw_histogram(self,data=None):
      fig = plt.figure(figsize=(10,6)); ax = fig.gca()
      student_df.hist(bins=30, ax=ax)
      plt.suptitle('Student Performance', y=1.03)    # this adds a "super" title and places it well
      plt.tight_layout()
      return fig
f = Histogram()
st.pyplot(f.draw_histogram(student_df))


import seaborn as sns
dd = student_df[['absences','age','health','sex','G3']]
st.title('Pair plot')
st.pyplot(sns.pairplot(dd,hue = 'sex'))

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(13,13))
sns.set(font_scale=1.0)
s = student_df.select_dtypes(include=['number'])
st.title('Heatmap')
st.pyplot(sns.heatmap(s.corr(),annot=True,cmap = 'coolwarm').figure)

st.title('Relationship plot')
st.pyplot(sns.relplot(data=student_df, x="G2", y="failures", hue="age").figure)

st.title('Relationship plot')
st.pyplot(sns.relplot(data=student_df, x="freetime", y="age", hue="failures").figure)

st.title('Relationship plot')
st.pyplot(sns.relplot(data=student_df, x="traveltime", y="age", hue="failures").figure)

st.title('Distribution plot')
st.pyplot(sns.displot(student_df, x="freetime", hue="sex").figure)

st.title('Categorical plot')
st.pyplot(sns.catplot(x='age',y='failures',data=student_df, hue="sex").figure)

st.title('Box plot')
st.pyplot(sns.boxplot(x='freetime',y='traveltime',hue='age',data=student_df).figure)

# ## Breast Cancer dataset
st.title("Breast Cancer Dataset")

import pandas as pd
cancer_df = pd.read_csv('data.csv')
st.header('Histogram')
class Histogram():
    def draw_histogram(self,data=None):
      fig = plt.figure(figsize=(25,15)); ax = fig.gca()
      cancer_df.hist(bins=30, ax=ax)
      plt.suptitle('Breast Cancer', y=1.03)    # this adds a "super" title and places it well
      plt.tight_layout()
      return fig
f = Histogram()
st.pyplot(f.draw_histogram(cancer_df))

df_new=cancer_df[['radius_mean','texture_mean','concavity_mean','symmetry_se','diagnosis']]
st.title('Pair plot')
st.pyplot(sns.pairplot(df_new, hue="diagnosis"))

st.title('Relationship plot')
st.pyplot(sns.relplot(data=cancer_df, x="area_mean", y="radius_mean", hue="diagnosis").figure)

st.title('Scatter plot')
st.pyplot(sns.scatterplot(data=cancer_df, x="area_mean", y="radius_mean", hue="diagnosis").figure)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(30, 25))
sns.set(font_scale=1.0) 
c = cancer_df.select_dtypes(include=['number'])
st.title('Heatmap')
st.pyplot(sns.heatmap(c.corr(),annot=True,cmap = 'coolwarm').figure)

st.title('Distribution plot')
st.pyplot(sns.displot(cancer_df, x="radius_mean", hue="diagnosis").figure)
st.title('Distribution plot')
st.pyplot(sns.displot(cancer_df,x='symmetry_worst', y='radius_mean',kind='kde',rug=True).figure)
st.title('Categorical plot')
st.pyplot(sns.catplot(x='diagnosis',y='area_mean',data=cancer_df,order=["M", "B"], hue="diagnosis").figure)


st.title("MPG Dataset")
import seaborn as sns
mpg_df = sns.load_dataset('mpg')
numeric_mpg_df = mpg_df.select_dtypes(include=['number'])
st.title('Distribution plot')
st.pyplot(sns.displot(mpg_df,x='horsepower', y='mpg',kind='kde',rug=True).figure)

import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.0) 
st.title('Heatmap')
st.pyplot(sns.heatmap(numeric_mpg_df.corr(),annot=True,cmap = 'coolwarm').figure)

color_palette = ['#FFD700', '#800080']
st.title('Regular plot')
st.pyplot(sns.regplot(data = mpg_df, lowess=True, line_kws={'color': color_palette[0]},scatter_kws={'color': color_palette[1]},x='horsepower',y='mpg').figure)

st.header('Facet Grid plot')
class FG():
    def draw_facet_grid(self):
      g = sns.FacetGrid(mpg_df, col="origin", hue='displacement',height=2.5, col_wrap=3)
      g.map(sns.scatterplot, 'acceleration','weight',alpha = .6)
      return g


f = FG()
st.pyplot(f.draw_facet_grid())
