import pandas as pd
import seaborn as sns
from scipy import stats as sts
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('insurance.csv')
feature = data.drop('charges', axis = 'columns')
outcome = data['charges'].values

# number of male and female in dataset
gender_values = data['sex'].unique().tolist()
gender_quants = {}
for i in gender_values:
    gender_quants[i] = round(data['sex'].tolist().count(i) / len(data['sex']) * 100, 2)
print(gender_quants)

gpl = [(gender_values[i] + '(' + str(gender_quants[gender_values[i]]) + '%)') for i in range(len(gender_values))]
plt.title('Number of Males & Females')
plt.pie(list(gender_quants.values()), labels = gpl, colors = ['#f0a03c','#fff023'])
plt.show()

# number of people from different regions in dataset
region_values = data['region'].unique().tolist()
region_quants = {}
for i in region_values:
    region_quants[i] = round(data['region'].tolist().count(i) / len(data['region']) * 100, 2)
print(region_quants)

rpl = [(region_values[i] + '(' + str(region_quants[region_values[i]]) + '%)') for i in range(len(region_values))]
plt.title('Number of People from different Region')
plt.pie(list(region_quants.values()), labels = rpl, colors = ['#5a91c8','#46cacf','#3ce034','#a4e643'])
plt.show()

# number of people who smoke / do not smoke in dataset
smoker_values = data['smoker'].unique().tolist()
smoker_quants = {}
for i in smoker_values:
    smoker_quants[i] = round(data['smoker'].tolist().count(i) / len(data['smoker']) * 100, 2)
print(smoker_quants)

spl = [(smoker_values[i] + '(' + str(smoker_quants[smoker_values[i]]) + '%)') for i in range(len(smoker_values))]
plt.title('Number of People who Smokes / Do not Smoke')
plt.pie(list(smoker_quants.values()), labels = spl, colors = ['#ab62e3','#6899e3'])
plt.show()

# creating a dataframe excluding discrete features to check correlation between features and that with outcome
continuous_ftr = data.drop(['sex','smoker','region'], axis = 'columns')
# heatmap showing collinearity between continuous features and outcome
plt.title('Correlation between Continuous Features and Insurance Charges')
sns.heatmap(
    data = continuous_ftr.corr(),
    annot = True,
    cmap = 'Blues',
    xticklabels = list(continuous_ftr.head(0)),
    yticklabels = list(continuous_ftr.head(0))
)
plt.show()

#printing the correlation matrix
print(continuous_ftr.corr())

# fetching list of charges corresponding to a male or female
charges_by_gender = []
for i in gender_values:
    charges_by_gender.append(data.query(f"sex == '{i}'")['charges'].tolist())

# boxplot showing insurance charges range for male or female
plt.title('Relation between Gender and Insurance Charges')
plt.boxplot(
    x = charges_by_gender,
    showmeans = True,
    showbox = True
)
plt.xticks(list(range(1, len(gender_values) + 1)), gender_values)
plt.grid(axis = 'x')
plt.show()

# fetching regions and list of charges corresponding to a particular region
charges_by_region = []
for i in region_values:
    charges_by_region.append(data.query(f"region == '{i}'")['charges'].tolist())

# boxplot showing insurance charges range for different types of region
plt.title('Relation between Region and Insurance Charges')
plt.boxplot(
    x = charges_by_region,
    showmeans = True,
    showbox = True
)
plt.xticks(list(range(1, len(region_values) + 1)), region_values)
plt.grid(axis = 'x')
plt.show()

# fetching list of charges corresponding to a smoker or non-smoker
charges_by_smoker = []
for i in smoker_values:
    charges_by_smoker.append(data.query(f"smoker == '{i}'")['charges'].tolist())

# boxplot showing insurance charges range for smoker or non-smoker
plt.title('Relation between Smoking and Insurance Charges')
plt.boxplot(
    x = charges_by_smoker,
    showmeans = True,
    showbox = True
)
plt.xticks(list(range(1, len(smoker_values) + 1)), smoker_values)
plt.grid(axis = 'x')
plt.show()

# using one-way anova test for testing importance of gender in determining insurance
gender_imp = sts.f_oneway(
    data.query(f"sex == 'male'")['charges'].tolist(),
    data.query(f"sex == 'female'")['charges'].tolist()
)
print('gender f-statistic =' ,list(gender_imp)[0])
print('gender p-value =' ,list(gender_imp)[1])

# using one-way anova test for testing importance of region in determining insurance
region_imp = sts.f_oneway(
    data.query(f"region == 'southwest'")['charges'].tolist(),
    data.query(f"region == 'southeast'")['charges'].tolist(),
    data.query(f"region == 'northwest'")['charges'].tolist(),
    data.query(f"region == 'northeast'")['charges'].tolist()
)
print('region f-statistic =' ,list(region_imp)[0])
print('region p-value =' ,list(region_imp)[1])

# using one-way anova test for testing importance of smoking in determining insurance
smoker_imp = sts.f_oneway(
    data.query(f"smoker == 'yes'")['charges'].tolist(),
    data.query(f"smoker == 'no'")['charges'].tolist()
)
print('smoker f-statistic =' ,list(smoker_imp)[0])
print('smoker p-value =' ,list(smoker_imp)[1])

# feature cleaning - eliminating unimportant features and encoding categorical features
feature = feature.drop(['sex','children','region'], axis = 'columns')
smoker_col_loc = list(feature.head(0)).index('smoker')
transformed_bmi = LabelEncoder().fit_transform(feature['smoker'].values)
feature = feature.drop('smoker', axis = 'columns')
feature.insert(loc = smoker_col_loc, column = 'smoker', value = transformed_bmi)
feature = feature.iloc[:,:].values

# splitting train-test data, training model
xtrain, xtest, ytrain, ytest = train_test_split(feature, outcome, test_size = 0.2)
regressor = RandomForestRegressor(random_state = 0, n_estimators = 100)
regressor.fit(xtrain, ytrain)

# predicting the outcome and comparaing with actual to get accuracy
ypred = regressor.predict(xtest)
print(round(r2_score(ytest, ypred), 2))