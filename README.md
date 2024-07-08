# Insurance_Prediction
<h3>Dataset</h3>
<p><b>Source</b>: Kaggle (https://www.kaggle.com/datasets/mirichoi0218/insurance)</p>
<p><b>Columns</b>: Age, Sex, BMI, Children, Smoker, Region, Charges</p>
<p><b>Records</b>: 1338</p>

<h3>Task</h3>
<ol type="1">
  <li>Finding number of males & females in data</li>
  <li>Finding number of people from different regions</li>
  <li>Finding number of smokers / non-smokers</li>
  <li>Calculating feature correlation of continuous features and with outcome using pandas.DataFrame.corr()</li>
  <li>Calculating importance of categorical feature with outcome using One-Way-Anova Test</li>
  <li>Selecting Features and predicting charges using RandomForestRegressor()</li>
</ol>

<h3>Feature Selection</h3>
<ul>
  <li>Continuous features with correlation >= 0.10</li>
  <li>Categorical features with p_value >= 0.05</li>
  <li><b>Selected Features</b> - Age, BMI, Smoker</li>
</ol>
