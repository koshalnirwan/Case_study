## Case Study is solved using stats model


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv('ionosphere.csv')
```


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>V29</th>
      <th>V30</th>
      <th>V31</th>
      <th>V32</th>
      <th>V33</th>
      <th>V34</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0.99539</td>
      <td>-0.05889</td>
      <td>0.85243</td>
      <td>0.02306</td>
      <td>0.83398</td>
      <td>-0.37708</td>
      <td>1.00000</td>
      <td>0.03760</td>
      <td>...</td>
      <td>-0.51171</td>
      <td>0.41078</td>
      <td>-0.46168</td>
      <td>0.21266</td>
      <td>-0.34090</td>
      <td>0.42267</td>
      <td>-0.54487</td>
      <td>0.18641</td>
      <td>-0.45300</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1.00000</td>
      <td>-0.18829</td>
      <td>0.93035</td>
      <td>-0.36156</td>
      <td>-0.10868</td>
      <td>-0.93597</td>
      <td>1.00000</td>
      <td>-0.04549</td>
      <td>...</td>
      <td>-0.26569</td>
      <td>-0.20468</td>
      <td>-0.18401</td>
      <td>-0.19040</td>
      <td>-0.11593</td>
      <td>-0.16626</td>
      <td>-0.06288</td>
      <td>-0.13738</td>
      <td>-0.02447</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1.00000</td>
      <td>-0.03365</td>
      <td>1.00000</td>
      <td>0.00485</td>
      <td>1.00000</td>
      <td>-0.12062</td>
      <td>0.88965</td>
      <td>0.01198</td>
      <td>...</td>
      <td>-0.40220</td>
      <td>0.58984</td>
      <td>-0.22145</td>
      <td>0.43100</td>
      <td>-0.17365</td>
      <td>0.60436</td>
      <td>-0.24180</td>
      <td>0.56045</td>
      <td>-0.38238</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1.00000</td>
      <td>-0.45161</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.71216</td>
      <td>-1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>...</td>
      <td>0.90695</td>
      <td>0.51613</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.20099</td>
      <td>0.25682</td>
      <td>1.00000</td>
      <td>-0.32382</td>
      <td>1.00000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1.00000</td>
      <td>-0.02401</td>
      <td>0.94140</td>
      <td>0.06531</td>
      <td>0.92106</td>
      <td>-0.23255</td>
      <td>0.77152</td>
      <td>-0.16399</td>
      <td>...</td>
      <td>-0.65158</td>
      <td>0.13290</td>
      <td>-0.53206</td>
      <td>0.02431</td>
      <td>-0.62197</td>
      <td>-0.05707</td>
      <td>-0.59573</td>
      <td>-0.04608</td>
      <td>-0.65697</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0.02337</td>
      <td>-0.00592</td>
      <td>-0.09924</td>
      <td>-0.11949</td>
      <td>-0.00763</td>
      <td>-0.11824</td>
      <td>0.14706</td>
      <td>0.06637</td>
      <td>...</td>
      <td>-0.01535</td>
      <td>-0.03240</td>
      <td>0.09223</td>
      <td>-0.07859</td>
      <td>0.00732</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>-0.00039</td>
      <td>0.12011</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0.97588</td>
      <td>-0.10602</td>
      <td>0.94601</td>
      <td>-0.20800</td>
      <td>0.92806</td>
      <td>-0.28350</td>
      <td>0.85996</td>
      <td>-0.27342</td>
      <td>...</td>
      <td>-0.81634</td>
      <td>0.13659</td>
      <td>-0.82510</td>
      <td>0.04606</td>
      <td>-0.82395</td>
      <td>-0.04262</td>
      <td>-0.81318</td>
      <td>-0.13832</td>
      <td>-0.80975</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>-1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>...</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>0.96355</td>
      <td>-0.07198</td>
      <td>1.00000</td>
      <td>-0.14333</td>
      <td>1.00000</td>
      <td>-0.21313</td>
      <td>1.00000</td>
      <td>-0.36174</td>
      <td>...</td>
      <td>-0.65440</td>
      <td>0.57577</td>
      <td>-0.69712</td>
      <td>0.25435</td>
      <td>-0.63919</td>
      <td>0.45114</td>
      <td>-0.72779</td>
      <td>0.38895</td>
      <td>-0.73420</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>-0.01864</td>
      <td>-0.08459</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.11470</td>
      <td>-0.26810</td>
      <td>...</td>
      <td>-0.01326</td>
      <td>0.20645</td>
      <td>-0.02294</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.16595</td>
      <td>0.24086</td>
      <td>-0.08208</td>
      <td>0.38065</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 35 columns</p>
</div>




```python
df.Class.unique()
```




    array([1, 0], dtype=int64)



# This is a Case Study of Classification and we have "Class" as our target variable with only 2 classes i.e  - 0 and 1

# Data Cleaning


```python
df.shape
```




    (351, 35)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 351 entries, 0 to 350
    Data columns (total 35 columns):
    V1       351 non-null int64
    V2       351 non-null int64
    V3       351 non-null float64
    V4       351 non-null float64
    V5       351 non-null float64
    V6       351 non-null float64
    V7       351 non-null float64
    V8       351 non-null float64
    V9       351 non-null float64
    V10      351 non-null float64
    V11      351 non-null float64
    V12      351 non-null float64
    V13      351 non-null float64
    V14      351 non-null float64
    V15      351 non-null float64
    V16      351 non-null float64
    V17      351 non-null float64
    V18      351 non-null float64
    V19      351 non-null float64
    V20      351 non-null float64
    V21      351 non-null float64
    V22      351 non-null float64
    V23      351 non-null float64
    V24      351 non-null float64
    V25      351 non-null float64
    V26      351 non-null float64
    V27      351 non-null float64
    V28      351 non-null float64
    V29      351 non-null float64
    V30      351 non-null float64
    V31      351 non-null float64
    V32      351 non-null float64
    V33      351 non-null float64
    V34      351 non-null float64
    Class    351 non-null int64
    dtypes: float64(32), int64(3)
    memory usage: 96.1 KB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>V29</th>
      <th>V30</th>
      <th>V31</th>
      <th>V32</th>
      <th>V33</th>
      <th>V34</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>351.000000</td>
      <td>351.0</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>...</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
      <td>351.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.891738</td>
      <td>0.0</td>
      <td>0.641342</td>
      <td>0.044372</td>
      <td>0.601068</td>
      <td>0.115889</td>
      <td>0.550095</td>
      <td>0.119360</td>
      <td>0.511848</td>
      <td>0.181345</td>
      <td>...</td>
      <td>-0.071187</td>
      <td>0.541641</td>
      <td>-0.069538</td>
      <td>0.378445</td>
      <td>-0.027907</td>
      <td>0.352514</td>
      <td>-0.003794</td>
      <td>0.349364</td>
      <td>0.014480</td>
      <td>0.641026</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.311155</td>
      <td>0.0</td>
      <td>0.497708</td>
      <td>0.441435</td>
      <td>0.519862</td>
      <td>0.460810</td>
      <td>0.492654</td>
      <td>0.520750</td>
      <td>0.507066</td>
      <td>0.483851</td>
      <td>...</td>
      <td>0.508495</td>
      <td>0.516205</td>
      <td>0.550025</td>
      <td>0.575886</td>
      <td>0.507974</td>
      <td>0.571483</td>
      <td>0.513574</td>
      <td>0.522663</td>
      <td>0.468337</td>
      <td>0.480384</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>...</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.472135</td>
      <td>-0.064735</td>
      <td>0.412660</td>
      <td>-0.024795</td>
      <td>0.211310</td>
      <td>-0.054840</td>
      <td>0.087110</td>
      <td>-0.048075</td>
      <td>...</td>
      <td>-0.332390</td>
      <td>0.286435</td>
      <td>-0.443165</td>
      <td>0.000000</td>
      <td>-0.236885</td>
      <td>0.000000</td>
      <td>-0.242595</td>
      <td>0.000000</td>
      <td>-0.165350</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.871110</td>
      <td>0.016310</td>
      <td>0.809200</td>
      <td>0.022800</td>
      <td>0.728730</td>
      <td>0.014710</td>
      <td>0.684210</td>
      <td>0.018290</td>
      <td>...</td>
      <td>-0.015050</td>
      <td>0.708240</td>
      <td>-0.017690</td>
      <td>0.496640</td>
      <td>0.000000</td>
      <td>0.442770</td>
      <td>0.000000</td>
      <td>0.409560</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.194185</td>
      <td>1.000000</td>
      <td>0.334655</td>
      <td>0.969240</td>
      <td>0.445675</td>
      <td>0.953240</td>
      <td>0.534195</td>
      <td>...</td>
      <td>0.156765</td>
      <td>0.999945</td>
      <td>0.153535</td>
      <td>0.883465</td>
      <td>0.154075</td>
      <td>0.857620</td>
      <td>0.200120</td>
      <td>0.813765</td>
      <td>0.171660</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 35 columns</p>
</div>




```python
df.isna().sum()
```




    V1       0
    V2       0
    V3       0
    V4       0
    V5       0
    V6       0
    V7       0
    V8       0
    V9       0
    V10      0
    V11      0
    V12      0
    V13      0
    V14      0
    V15      0
    V16      0
    V17      0
    V18      0
    V19      0
    V20      0
    V21      0
    V22      0
    V23      0
    V24      0
    V25      0
    V26      0
    V27      0
    V28      0
    V29      0
    V30      0
    V31      0
    V32      0
    V33      0
    V34      0
    Class    0
    dtype: int64




```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>V29</th>
      <th>V30</th>
      <th>V31</th>
      <th>V32</th>
      <th>V33</th>
      <th>V34</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>V1</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.302034</td>
      <td>-0.006529</td>
      <td>0.156152</td>
      <td>0.127606</td>
      <td>0.221867</td>
      <td>0.027079</td>
      <td>0.189242</td>
      <td>-0.051883</td>
      <td>...</td>
      <td>0.149789</td>
      <td>-0.203100</td>
      <td>-0.010725</td>
      <td>0.133632</td>
      <td>-0.121415</td>
      <td>0.167031</td>
      <td>-0.100914</td>
      <td>0.162962</td>
      <td>0.010788</td>
      <td>0.465614</td>
    </tr>
    <tr>
      <th>V2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>V3</th>
      <td>0.302034</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.143365</td>
      <td>0.476587</td>
      <td>0.025768</td>
      <td>0.440254</td>
      <td>0.008717</td>
      <td>0.471614</td>
      <td>0.047916</td>
      <td>...</td>
      <td>-0.073202</td>
      <td>0.077911</td>
      <td>0.123345</td>
      <td>0.344459</td>
      <td>0.057890</td>
      <td>0.246653</td>
      <td>-0.009332</td>
      <td>0.263343</td>
      <td>0.000584</td>
      <td>0.519145</td>
    </tr>
    <tr>
      <th>V4</th>
      <td>-0.006529</td>
      <td>NaN</td>
      <td>0.143365</td>
      <td>1.000000</td>
      <td>0.001152</td>
      <td>-0.190308</td>
      <td>-0.054030</td>
      <td>0.255003</td>
      <td>-0.302317</td>
      <td>0.207697</td>
      <td>...</td>
      <td>-0.236987</td>
      <td>-0.046910</td>
      <td>0.000743</td>
      <td>-0.041090</td>
      <td>0.342301</td>
      <td>-0.172276</td>
      <td>-0.122788</td>
      <td>-0.153964</td>
      <td>0.034608</td>
      <td>0.125884</td>
    </tr>
    <tr>
      <th>V5</th>
      <td>0.156152</td>
      <td>NaN</td>
      <td>0.476587</td>
      <td>0.001152</td>
      <td>1.000000</td>
      <td>0.038323</td>
      <td>0.597075</td>
      <td>-0.029794</td>
      <td>0.450454</td>
      <td>-0.034236</td>
      <td>...</td>
      <td>-0.032254</td>
      <td>0.140899</td>
      <td>0.184517</td>
      <td>0.257646</td>
      <td>0.051068</td>
      <td>0.399840</td>
      <td>0.025681</td>
      <td>0.383467</td>
      <td>-0.099478</td>
      <td>0.516477</td>
    </tr>
    <tr>
      <th>V6</th>
      <td>0.127606</td>
      <td>NaN</td>
      <td>0.025768</td>
      <td>-0.190308</td>
      <td>0.038323</td>
      <td>1.000000</td>
      <td>-0.010227</td>
      <td>0.274747</td>
      <td>-0.120712</td>
      <td>0.200080</td>
      <td>...</td>
      <td>0.041787</td>
      <td>-0.175433</td>
      <td>-0.068775</td>
      <td>-0.029392</td>
      <td>-0.158090</td>
      <td>-0.100240</td>
      <td>0.316802</td>
      <td>0.016899</td>
      <td>0.185215</td>
      <td>0.149099</td>
    </tr>
    <tr>
      <th>V7</th>
      <td>0.221867</td>
      <td>NaN</td>
      <td>0.440254</td>
      <td>-0.054030</td>
      <td>0.597075</td>
      <td>-0.010227</td>
      <td>1.000000</td>
      <td>-0.150424</td>
      <td>0.461889</td>
      <td>-0.090268</td>
      <td>...</td>
      <td>0.087734</td>
      <td>0.097566</td>
      <td>0.109391</td>
      <td>0.300632</td>
      <td>-0.015158</td>
      <td>0.415216</td>
      <td>-0.008323</td>
      <td>0.545881</td>
      <td>-0.076460</td>
      <td>0.450429</td>
    </tr>
    <tr>
      <th>V8</th>
      <td>0.027079</td>
      <td>NaN</td>
      <td>0.008717</td>
      <td>0.255003</td>
      <td>-0.029794</td>
      <td>0.274747</td>
      <td>-0.150424</td>
      <td>1.000000</td>
      <td>-0.336013</td>
      <td>0.373567</td>
      <td>...</td>
      <td>-0.133023</td>
      <td>-0.254130</td>
      <td>0.072373</td>
      <td>-0.139725</td>
      <td>0.078585</td>
      <td>-0.166682</td>
      <td>0.152381</td>
      <td>-0.200860</td>
      <td>0.360610</td>
      <td>0.207544</td>
    </tr>
    <tr>
      <th>V9</th>
      <td>0.189242</td>
      <td>NaN</td>
      <td>0.471614</td>
      <td>-0.302317</td>
      <td>0.450454</td>
      <td>-0.120712</td>
      <td>0.461889</td>
      <td>-0.336013</td>
      <td>1.000000</td>
      <td>-0.251950</td>
      <td>...</td>
      <td>0.107478</td>
      <td>0.172210</td>
      <td>0.146817</td>
      <td>0.329813</td>
      <td>-0.031983</td>
      <td>0.316021</td>
      <td>-0.067499</td>
      <td>0.344814</td>
      <td>-0.095597</td>
      <td>0.294852</td>
    </tr>
    <tr>
      <th>V10</th>
      <td>-0.051883</td>
      <td>NaN</td>
      <td>0.047916</td>
      <td>0.207697</td>
      <td>-0.034236</td>
      <td>0.200080</td>
      <td>-0.090268</td>
      <td>0.373567</td>
      <td>-0.251950</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.043680</td>
      <td>-0.250947</td>
      <td>0.072018</td>
      <td>-0.123296</td>
      <td>-0.008578</td>
      <td>-0.155661</td>
      <td>-0.015640</td>
      <td>-0.203629</td>
      <td>0.098104</td>
      <td>0.120634</td>
    </tr>
    <tr>
      <th>V11</th>
      <td>0.034138</td>
      <td>NaN</td>
      <td>0.325016</td>
      <td>-0.190090</td>
      <td>0.449829</td>
      <td>-0.291447</td>
      <td>0.412876</td>
      <td>-0.364003</td>
      <td>0.670813</td>
      <td>-0.337374</td>
      <td>...</td>
      <td>0.131849</td>
      <td>0.292281</td>
      <td>0.197369</td>
      <td>0.396851</td>
      <td>0.074600</td>
      <td>0.294646</td>
      <td>0.023922</td>
      <td>0.339506</td>
      <td>-0.152225</td>
      <td>0.167908</td>
    </tr>
    <tr>
      <th>V12</th>
      <td>0.072216</td>
      <td>NaN</td>
      <td>0.169981</td>
      <td>0.315877</td>
      <td>0.042896</td>
      <td>0.163933</td>
      <td>-0.020395</td>
      <td>0.429146</td>
      <td>-0.167705</td>
      <td>0.441505</td>
      <td>...</td>
      <td>-0.076828</td>
      <td>-0.227890</td>
      <td>0.061292</td>
      <td>-0.208294</td>
      <td>0.138842</td>
      <td>-0.208855</td>
      <td>0.010276</td>
      <td>-0.181166</td>
      <td>0.066584</td>
      <td>0.159940</td>
    </tr>
    <tr>
      <th>V13</th>
      <td>0.102558</td>
      <td>NaN</td>
      <td>0.217597</td>
      <td>-0.149216</td>
      <td>0.482118</td>
      <td>-0.307197</td>
      <td>0.631060</td>
      <td>-0.355875</td>
      <td>0.562072</td>
      <td>-0.406358</td>
      <td>...</td>
      <td>0.197266</td>
      <td>0.290095</td>
      <td>0.146800</td>
      <td>0.277995</td>
      <td>0.094400</td>
      <td>0.355473</td>
      <td>-0.041629</td>
      <td>0.473780</td>
      <td>-0.065131</td>
      <td>0.181682</td>
    </tr>
    <tr>
      <th>V14</th>
      <td>0.199230</td>
      <td>NaN</td>
      <td>0.164550</td>
      <td>0.236604</td>
      <td>0.127217</td>
      <td>0.135206</td>
      <td>0.083657</td>
      <td>0.253740</td>
      <td>-0.088988</td>
      <td>0.323813</td>
      <td>...</td>
      <td>0.070563</td>
      <td>-0.313597</td>
      <td>0.066965</td>
      <td>-0.215050</td>
      <td>0.095677</td>
      <td>-0.147949</td>
      <td>-0.067790</td>
      <td>-0.096091</td>
      <td>0.096444</td>
      <td>0.197041</td>
    </tr>
    <tr>
      <th>V15</th>
      <td>0.113622</td>
      <td>NaN</td>
      <td>0.198306</td>
      <td>-0.253150</td>
      <td>0.398878</td>
      <td>-0.359342</td>
      <td>0.615407</td>
      <td>-0.352216</td>
      <td>0.618461</td>
      <td>-0.374908</td>
      <td>...</td>
      <td>0.228787</td>
      <td>0.339516</td>
      <td>0.164177</td>
      <td>0.355917</td>
      <td>0.090009</td>
      <td>0.445175</td>
      <td>-0.013132</td>
      <td>0.490879</td>
      <td>-0.119929</td>
      <td>0.207201</td>
    </tr>
    <tr>
      <th>V16</th>
      <td>0.100474</td>
      <td>NaN</td>
      <td>0.094301</td>
      <td>0.185872</td>
      <td>0.087992</td>
      <td>0.157740</td>
      <td>-0.021493</td>
      <td>0.419673</td>
      <td>-0.032689</td>
      <td>0.334135</td>
      <td>...</td>
      <td>0.179892</td>
      <td>-0.313300</td>
      <td>0.219421</td>
      <td>-0.159639</td>
      <td>0.146052</td>
      <td>-0.183578</td>
      <td>0.101401</td>
      <td>-0.179906</td>
      <td>0.241702</td>
      <td>0.148775</td>
    </tr>
    <tr>
      <th>V17</th>
      <td>0.057783</td>
      <td>NaN</td>
      <td>0.221446</td>
      <td>-0.251143</td>
      <td>0.277932</td>
      <td>-0.316705</td>
      <td>0.379737</td>
      <td>-0.491863</td>
      <td>0.633574</td>
      <td>-0.392047</td>
      <td>...</td>
      <td>0.129545</td>
      <td>0.427263</td>
      <td>0.168507</td>
      <td>0.404276</td>
      <td>0.074604</td>
      <td>0.358692</td>
      <td>0.015888</td>
      <td>0.360039</td>
      <td>-0.063429</td>
      <td>0.087060</td>
    </tr>
    <tr>
      <th>V18</th>
      <td>0.076019</td>
      <td>NaN</td>
      <td>0.172002</td>
      <td>-0.147451</td>
      <td>0.027588</td>
      <td>0.188073</td>
      <td>0.115927</td>
      <td>0.068717</td>
      <td>0.200786</td>
      <td>0.130752</td>
      <td>...</td>
      <td>0.415094</td>
      <td>-0.199799</td>
      <td>0.289871</td>
      <td>-0.118047</td>
      <td>0.067712</td>
      <td>-0.158096</td>
      <td>0.256970</td>
      <td>-0.059076</td>
      <td>0.054113</td>
      <td>0.119346</td>
    </tr>
    <tr>
      <th>V19</th>
      <td>0.200237</td>
      <td>NaN</td>
      <td>0.285280</td>
      <td>-0.332213</td>
      <td>0.221532</td>
      <td>-0.208571</td>
      <td>0.372572</td>
      <td>-0.400523</td>
      <td>0.673490</td>
      <td>-0.471665</td>
      <td>...</td>
      <td>0.173517</td>
      <td>0.424692</td>
      <td>0.205673</td>
      <td>0.473187</td>
      <td>0.085579</td>
      <td>0.418580</td>
      <td>0.124713</td>
      <td>0.446093</td>
      <td>0.001458</td>
      <td>0.117435</td>
    </tr>
    <tr>
      <th>V20</th>
      <td>0.019230</td>
      <td>NaN</td>
      <td>0.150800</td>
      <td>0.167244</td>
      <td>0.041959</td>
      <td>-0.061261</td>
      <td>0.158917</td>
      <td>0.077624</td>
      <td>0.067314</td>
      <td>-0.001418</td>
      <td>...</td>
      <td>0.397429</td>
      <td>-0.165945</td>
      <td>0.272236</td>
      <td>-0.213057</td>
      <td>0.396781</td>
      <td>-0.305049</td>
      <td>0.125374</td>
      <td>-0.107816</td>
      <td>0.188940</td>
      <td>0.035620</td>
    </tr>
    <tr>
      <th>V21</th>
      <td>0.173828</td>
      <td>NaN</td>
      <td>0.149374</td>
      <td>-0.281084</td>
      <td>0.326223</td>
      <td>-0.114966</td>
      <td>0.586627</td>
      <td>-0.370473</td>
      <td>0.492411</td>
      <td>-0.404818</td>
      <td>...</td>
      <td>0.160227</td>
      <td>0.398951</td>
      <td>0.073413</td>
      <td>0.479619</td>
      <td>0.071592</td>
      <td>0.511278</td>
      <td>0.121923</td>
      <td>0.616620</td>
      <td>-0.002596</td>
      <td>0.219583</td>
    </tr>
    <tr>
      <th>V22</th>
      <td>-0.153902</td>
      <td>NaN</td>
      <td>0.138065</td>
      <td>-0.035401</td>
      <td>0.163663</td>
      <td>-0.132422</td>
      <td>0.190805</td>
      <td>-0.212007</td>
      <td>0.237322</td>
      <td>-0.040414</td>
      <td>...</td>
      <td>0.396342</td>
      <td>-0.031281</td>
      <td>0.407093</td>
      <td>-0.086614</td>
      <td>0.323712</td>
      <td>-0.075714</td>
      <td>0.207718</td>
      <td>-0.090417</td>
      <td>0.132848</td>
      <td>-0.116385</td>
    </tr>
    <tr>
      <th>V23</th>
      <td>0.011772</td>
      <td>NaN</td>
      <td>0.250832</td>
      <td>-0.143719</td>
      <td>0.502878</td>
      <td>-0.215778</td>
      <td>0.373186</td>
      <td>-0.270624</td>
      <td>0.352218</td>
      <td>-0.318463</td>
      <td>...</td>
      <td>0.051802</td>
      <td>0.569176</td>
      <td>0.207700</td>
      <td>0.540443</td>
      <td>0.165890</td>
      <td>0.538973</td>
      <td>0.160531</td>
      <td>0.503426</td>
      <td>0.049284</td>
      <td>0.204361</td>
    </tr>
    <tr>
      <th>V24</th>
      <td>-0.082586</td>
      <td>NaN</td>
      <td>-0.012570</td>
      <td>0.164196</td>
      <td>0.098274</td>
      <td>-0.286541</td>
      <td>0.112717</td>
      <td>0.007045</td>
      <td>0.161258</td>
      <td>0.101850</td>
      <td>...</td>
      <td>0.352872</td>
      <td>-0.036682</td>
      <td>0.376951</td>
      <td>-0.167937</td>
      <td>0.364626</td>
      <td>-0.127822</td>
      <td>0.084291</td>
      <td>-0.211597</td>
      <td>0.094685</td>
      <td>0.006193</td>
    </tr>
    <tr>
      <th>V25</th>
      <td>0.016717</td>
      <td>NaN</td>
      <td>0.304898</td>
      <td>-0.104632</td>
      <td>0.243063</td>
      <td>-0.177576</td>
      <td>0.286749</td>
      <td>-0.179928</td>
      <td>0.356564</td>
      <td>-0.254785</td>
      <td>...</td>
      <td>-0.077006</td>
      <td>0.503526</td>
      <td>0.176257</td>
      <td>0.650908</td>
      <td>0.013639</td>
      <td>0.516121</td>
      <td>0.134478</td>
      <td>0.460692</td>
      <td>0.111086</td>
      <td>0.188185</td>
    </tr>
    <tr>
      <th>V26</th>
      <td>0.149789</td>
      <td>NaN</td>
      <td>-0.073202</td>
      <td>-0.236987</td>
      <td>-0.032254</td>
      <td>0.041787</td>
      <td>0.087734</td>
      <td>-0.133023</td>
      <td>0.107478</td>
      <td>-0.043680</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.011314</td>
      <td>0.432426</td>
      <td>-0.113499</td>
      <td>0.281075</td>
      <td>-0.162707</td>
      <td>0.340692</td>
      <td>-0.085966</td>
      <td>0.221630</td>
      <td>0.001541</td>
    </tr>
    <tr>
      <th>V27</th>
      <td>-0.203100</td>
      <td>NaN</td>
      <td>0.077911</td>
      <td>-0.046910</td>
      <td>0.140899</td>
      <td>-0.175433</td>
      <td>0.097566</td>
      <td>-0.254130</td>
      <td>0.172210</td>
      <td>-0.250947</td>
      <td>...</td>
      <td>-0.011314</td>
      <td>1.000000</td>
      <td>0.058253</td>
      <td>0.509070</td>
      <td>0.163412</td>
      <td>0.376400</td>
      <td>0.160318</td>
      <td>0.469326</td>
      <td>0.082969</td>
      <td>-0.111107</td>
    </tr>
    <tr>
      <th>V28</th>
      <td>-0.010725</td>
      <td>NaN</td>
      <td>0.123345</td>
      <td>0.000743</td>
      <td>0.184517</td>
      <td>-0.068775</td>
      <td>0.109391</td>
      <td>0.072373</td>
      <td>0.146817</td>
      <td>0.072018</td>
      <td>...</td>
      <td>0.432426</td>
      <td>0.058253</td>
      <td>1.000000</td>
      <td>0.042002</td>
      <td>0.385475</td>
      <td>-0.008136</td>
      <td>0.500801</td>
      <td>-0.124826</td>
      <td>0.376772</td>
      <td>0.042756</td>
    </tr>
    <tr>
      <th>V29</th>
      <td>0.133632</td>
      <td>NaN</td>
      <td>0.344459</td>
      <td>-0.041090</td>
      <td>0.257646</td>
      <td>-0.029392</td>
      <td>0.300632</td>
      <td>-0.139725</td>
      <td>0.329813</td>
      <td>-0.123296</td>
      <td>...</td>
      <td>-0.113499</td>
      <td>0.509070</td>
      <td>0.042002</td>
      <td>1.000000</td>
      <td>-0.011000</td>
      <td>0.553855</td>
      <td>0.055997</td>
      <td>0.579684</td>
      <td>0.082504</td>
      <td>0.250036</td>
    </tr>
    <tr>
      <th>V30</th>
      <td>-0.121415</td>
      <td>NaN</td>
      <td>0.057890</td>
      <td>0.342301</td>
      <td>0.051068</td>
      <td>-0.158090</td>
      <td>-0.015158</td>
      <td>0.078585</td>
      <td>-0.031983</td>
      <td>-0.008578</td>
      <td>...</td>
      <td>0.281075</td>
      <td>0.163412</td>
      <td>0.385475</td>
      <td>-0.011000</td>
      <td>1.000000</td>
      <td>-0.163586</td>
      <td>0.297085</td>
      <td>-0.180566</td>
      <td>0.391271</td>
      <td>-0.003942</td>
    </tr>
    <tr>
      <th>V31</th>
      <td>0.167031</td>
      <td>NaN</td>
      <td>0.246653</td>
      <td>-0.172276</td>
      <td>0.399840</td>
      <td>-0.100240</td>
      <td>0.415216</td>
      <td>-0.166682</td>
      <td>0.316021</td>
      <td>-0.155661</td>
      <td>...</td>
      <td>-0.162707</td>
      <td>0.376400</td>
      <td>-0.008136</td>
      <td>0.553855</td>
      <td>-0.163586</td>
      <td>1.000000</td>
      <td>-0.028877</td>
      <td>0.692408</td>
      <td>-0.037579</td>
      <td>0.294417</td>
    </tr>
    <tr>
      <th>V32</th>
      <td>-0.100914</td>
      <td>NaN</td>
      <td>-0.009332</td>
      <td>-0.122788</td>
      <td>0.025681</td>
      <td>0.316802</td>
      <td>-0.008323</td>
      <td>0.152381</td>
      <td>-0.067499</td>
      <td>-0.015640</td>
      <td>...</td>
      <td>0.340692</td>
      <td>0.160318</td>
      <td>0.500801</td>
      <td>0.055997</td>
      <td>0.297085</td>
      <td>-0.028877</td>
      <td>1.000000</td>
      <td>-0.012998</td>
      <td>0.514992</td>
      <td>-0.036004</td>
    </tr>
    <tr>
      <th>V33</th>
      <td>0.162962</td>
      <td>NaN</td>
      <td>0.263343</td>
      <td>-0.153964</td>
      <td>0.383467</td>
      <td>0.016899</td>
      <td>0.545881</td>
      <td>-0.200860</td>
      <td>0.344814</td>
      <td>-0.203629</td>
      <td>...</td>
      <td>-0.085966</td>
      <td>0.469326</td>
      <td>-0.124826</td>
      <td>0.579684</td>
      <td>-0.180566</td>
      <td>0.692408</td>
      <td>-0.012998</td>
      <td>1.000000</td>
      <td>-0.131840</td>
      <td>0.261157</td>
    </tr>
    <tr>
      <th>V34</th>
      <td>0.010788</td>
      <td>NaN</td>
      <td>0.000584</td>
      <td>0.034608</td>
      <td>-0.099478</td>
      <td>0.185215</td>
      <td>-0.076460</td>
      <td>0.360610</td>
      <td>-0.095597</td>
      <td>0.098104</td>
      <td>...</td>
      <td>0.221630</td>
      <td>0.082969</td>
      <td>0.376772</td>
      <td>0.082504</td>
      <td>0.391271</td>
      <td>-0.037579</td>
      <td>0.514992</td>
      <td>-0.131840</td>
      <td>1.000000</td>
      <td>-0.064168</td>
    </tr>
    <tr>
      <th>Class</th>
      <td>0.465614</td>
      <td>NaN</td>
      <td>0.519145</td>
      <td>0.125884</td>
      <td>0.516477</td>
      <td>0.149099</td>
      <td>0.450429</td>
      <td>0.207544</td>
      <td>0.294852</td>
      <td>0.120634</td>
      <td>...</td>
      <td>0.001541</td>
      <td>-0.111107</td>
      <td>0.042756</td>
      <td>0.250036</td>
      <td>-0.003942</td>
      <td>0.294417</td>
      <td>-0.036004</td>
      <td>0.261157</td>
      <td>-0.064168</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>35 rows × 35 columns</p>
</div>



variable V2 have no significance in data as it is not corelated with any other variable.


```python
df = df.drop('V2',axis=1)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>...</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>V29</th>
      <th>V30</th>
      <th>V31</th>
      <th>V32</th>
      <th>V33</th>
      <th>V34</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.99539</td>
      <td>-0.05889</td>
      <td>0.85243</td>
      <td>0.02306</td>
      <td>0.83398</td>
      <td>-0.37708</td>
      <td>1.00000</td>
      <td>0.03760</td>
      <td>0.85243</td>
      <td>...</td>
      <td>-0.51171</td>
      <td>0.41078</td>
      <td>-0.46168</td>
      <td>0.21266</td>
      <td>-0.34090</td>
      <td>0.42267</td>
      <td>-0.54487</td>
      <td>0.18641</td>
      <td>-0.45300</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.00000</td>
      <td>-0.18829</td>
      <td>0.93035</td>
      <td>-0.36156</td>
      <td>-0.10868</td>
      <td>-0.93597</td>
      <td>1.00000</td>
      <td>-0.04549</td>
      <td>0.50874</td>
      <td>...</td>
      <td>-0.26569</td>
      <td>-0.20468</td>
      <td>-0.18401</td>
      <td>-0.19040</td>
      <td>-0.11593</td>
      <td>-0.16626</td>
      <td>-0.06288</td>
      <td>-0.13738</td>
      <td>-0.02447</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1.00000</td>
      <td>-0.03365</td>
      <td>1.00000</td>
      <td>0.00485</td>
      <td>1.00000</td>
      <td>-0.12062</td>
      <td>0.88965</td>
      <td>0.01198</td>
      <td>0.73082</td>
      <td>...</td>
      <td>-0.40220</td>
      <td>0.58984</td>
      <td>-0.22145</td>
      <td>0.43100</td>
      <td>-0.17365</td>
      <td>0.60436</td>
      <td>-0.24180</td>
      <td>0.56045</td>
      <td>-0.38238</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1.00000</td>
      <td>-0.45161</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.71216</td>
      <td>-1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>...</td>
      <td>0.90695</td>
      <td>0.51613</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.20099</td>
      <td>0.25682</td>
      <td>1.00000</td>
      <td>-0.32382</td>
      <td>1.00000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1.00000</td>
      <td>-0.02401</td>
      <td>0.94140</td>
      <td>0.06531</td>
      <td>0.92106</td>
      <td>-0.23255</td>
      <td>0.77152</td>
      <td>-0.16399</td>
      <td>0.52798</td>
      <td>...</td>
      <td>-0.65158</td>
      <td>0.13290</td>
      <td>-0.53206</td>
      <td>0.02431</td>
      <td>-0.62197</td>
      <td>-0.05707</td>
      <td>-0.59573</td>
      <td>-0.04608</td>
      <td>-0.65697</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
y = df.Class
```


```python
X = df.drop("Class",axis=1)
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100,stratify=y)
```


```python
col = X_train.columns
```

Converting all values between 0 and 1 to make it easy for the machine to learn


```python
from sklearn.linear_model import LogisticRegression
```


```python
logreg=LogisticRegression()
```

# Feature Engineering and Model Building


```python
from sklearn.feature_selection import RFE
```


```python
rfe=RFE(logreg,10)
```


```python
rfe=rfe.fit(X_train,y_train)
```

Getting the Rankings of columns of trained data in rfe


```python
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
```




    [('V1', True, 1),
     ('V3', True, 1),
     ('V4', False, 21),
     ('V5', True, 1),
     ('V6', False, 11),
     ('V7', True, 1),
     ('V8', True, 1),
     ('V9', True, 1),
     ('V10', False, 14),
     ('V11', True, 1),
     ('V12', False, 24),
     ('V13', False, 15),
     ('V14', False, 6),
     ('V15', False, 16),
     ('V16', False, 22),
     ('V17', False, 23),
     ('V18', False, 13),
     ('V19', False, 9),
     ('V20', False, 20),
     ('V21', False, 19),
     ('V22', True, 1),
     ('V23', False, 8),
     ('V24', False, 17),
     ('V25', False, 10),
     ('V26', False, 2),
     ('V27', True, 1),
     ('V28', False, 18),
     ('V29', False, 3),
     ('V30', False, 5),
     ('V31', True, 1),
     ('V32', False, 12),
     ('V33', False, 7),
     ('V34', False, 4)]




```python
col = X_train.columns[rfe.support_]
```


```python
col
```




    Index(['V1', 'V3', 'V5', 'V7', 'V8', 'V9', 'V11', 'V22', 'V27', 'V31'], dtype='object')




```python
X_train=X_train[col]
```


```python
import statsmodels.api as sm
```


```python
X_train_sm=sm.add_constant(X_train)
```

    C:\Users\kosha\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    


```python
logm2=sm.GLM(y_train,X_train_sm,family=sm.families.Binomial())
```


```python
res=logm2.fit()
```


```python
res.summary()
```




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Class</td>      <th>  No. Observations:  </th>  <td>   245</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   234</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>    10</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -63.255</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sun, 14 Feb 2021</td> <th>  Deviance:          </th> <td>  126.51</td>
</tr>
<tr>
  <th>Time:</th>                <td>11:39:32</td>     <th>  Pearson chi2:      </th>  <td>  248.</td> 
</tr>
<tr>
  <th>No. Iterations:</th>         <td>23</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>  -28.8370</td> <td> 1.69e+04</td> <td>   -0.002</td> <td> 0.999</td> <td>-3.32e+04</td> <td> 3.31e+04</td>
</tr>
<tr>
  <th>V1</th>    <td>   27.1089</td> <td> 1.69e+04</td> <td>    0.002</td> <td> 0.999</td> <td>-3.31e+04</td> <td> 3.32e+04</td>
</tr>
<tr>
  <th>V3</th>    <td>    3.1352</td> <td>    0.942</td> <td>    3.328</td> <td> 0.001</td> <td>    1.289</td> <td>    4.982</td>
</tr>
<tr>
  <th>V5</th>    <td>    0.4975</td> <td>    0.924</td> <td>    0.538</td> <td> 0.591</td> <td>   -1.314</td> <td>    2.309</td>
</tr>
<tr>
  <th>V7</th>    <td>    1.7954</td> <td>    1.091</td> <td>    1.645</td> <td> 0.100</td> <td>   -0.344</td> <td>    3.934</td>
</tr>
<tr>
  <th>V8</th>    <td>    1.7139</td> <td>    0.697</td> <td>    2.459</td> <td> 0.014</td> <td>    0.348</td> <td>    3.080</td>
</tr>
<tr>
  <th>V9</th>    <td>    2.4189</td> <td>    1.265</td> <td>    1.912</td> <td> 0.056</td> <td>   -0.060</td> <td>    4.898</td>
</tr>
<tr>
  <th>V11</th>   <td>   -2.9364</td> <td>    1.017</td> <td>   -2.888</td> <td> 0.004</td> <td>   -4.929</td> <td>   -0.943</td>
</tr>
<tr>
  <th>V22</th>   <td>   -0.8764</td> <td>    0.718</td> <td>   -1.220</td> <td> 0.222</td> <td>   -2.284</td> <td>    0.531</td>
</tr>
<tr>
  <th>V27</th>   <td>   -3.0012</td> <td>    0.918</td> <td>   -3.271</td> <td> 0.001</td> <td>   -4.800</td> <td>   -1.203</td>
</tr>
<tr>
  <th>V31</th>   <td>    2.8839</td> <td>    0.953</td> <td>    3.027</td> <td> 0.002</td> <td>    1.017</td> <td>    4.751</td>
</tr>
</table>




```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```


```python
vif=pd.DataFrame()
vif["Features"]=X_train.columns
vif["VIF"]=[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
vif["VIF"]=round(vif["VIF"])
vif=vif.sort_values(by="VIF",ascending=False)
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>V9</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V3</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V5</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>V1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V7</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V11</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>V27</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>V31</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V8</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>V22</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Now we will drop the columns whose p-value is more than 0.05


```python
X_train.drop("V1",axis=1,inplace=True)
```


```python
logm1=sm.GLM(y_train,(sm.add_constant(X_train)),family=sm.families.Binomial())
logm1.fit().summary()
```

    C:\Users\kosha\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Class</td>      <th>  No. Observations:  </th>  <td>   245</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   235</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     9</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -84.822</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sun, 14 Feb 2021</td> <th>  Deviance:          </th> <td>  169.64</td>
</tr>
<tr>
  <th>Time:</th>                <td>11:40:19</td>     <th>  Pearson chi2:      </th>  <td>  392.</td> 
</tr>
<tr>
  <th>No. Iterations:</th>          <td>6</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>   -2.0174</td> <td>    0.514</td> <td>   -3.924</td> <td> 0.000</td> <td>   -3.025</td> <td>   -1.010</td>
</tr>
<tr>
  <th>V3</th>    <td>    2.4741</td> <td>    0.713</td> <td>    3.470</td> <td> 0.001</td> <td>    1.077</td> <td>    3.872</td>
</tr>
<tr>
  <th>V5</th>    <td>    1.9804</td> <td>    0.731</td> <td>    2.709</td> <td> 0.007</td> <td>    0.547</td> <td>    3.413</td>
</tr>
<tr>
  <th>V7</th>    <td>    0.9820</td> <td>    0.657</td> <td>    1.495</td> <td> 0.135</td> <td>   -0.305</td> <td>    2.269</td>
</tr>
<tr>
  <th>V8</th>    <td>    0.8630</td> <td>    0.473</td> <td>    1.823</td> <td> 0.068</td> <td>   -0.065</td> <td>    1.791</td>
</tr>
<tr>
  <th>V9</th>    <td>    2.0430</td> <td>    0.817</td> <td>    2.502</td> <td> 0.012</td> <td>    0.443</td> <td>    3.643</td>
</tr>
<tr>
  <th>V11</th>   <td>   -1.8751</td> <td>    0.670</td> <td>   -2.797</td> <td> 0.005</td> <td>   -3.189</td> <td>   -0.561</td>
</tr>
<tr>
  <th>V22</th>   <td>   -2.0265</td> <td>    0.512</td> <td>   -3.959</td> <td> 0.000</td> <td>   -3.030</td> <td>   -1.023</td>
</tr>
<tr>
  <th>V27</th>   <td>   -2.3860</td> <td>    0.590</td> <td>   -4.041</td> <td> 0.000</td> <td>   -3.543</td> <td>   -1.229</td>
</tr>
<tr>
  <th>V31</th>   <td>    1.0872</td> <td>    0.487</td> <td>    2.234</td> <td> 0.025</td> <td>    0.133</td> <td>    2.041</td>
</tr>
</table>




```python
vif=pd.DataFrame()
vif["Features"]=X_train.columns
vif["VIF"]=[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
vif["VIF"]=round(vif["VIF"])
vif=vif.sort_values(by="VIF",ascending=False)
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>V9</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V5</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>V3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V7</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V11</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>V27</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>V31</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V8</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V22</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train.drop("V7",axis=1,inplace=True)
```


```python
logm1=sm.GLM(y_train,(sm.add_constant(X_train)),family=sm.families.Binomial())
logm1.fit().summary()
```

    C:\Users\kosha\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Class</td>      <th>  No. Observations:  </th>  <td>   245</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   236</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     8</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -86.022</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sun, 14 Feb 2021</td> <th>  Deviance:          </th> <td>  172.04</td>
</tr>
<tr>
  <th>Time:</th>                <td>11:40:50</td>     <th>  Pearson chi2:      </th>  <td>  434.</td> 
</tr>
<tr>
  <th>No. Iterations:</th>          <td>6</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>   -1.8363</td> <td>    0.499</td> <td>   -3.683</td> <td> 0.000</td> <td>   -2.814</td> <td>   -0.859</td>
</tr>
<tr>
  <th>V3</th>    <td>    2.4561</td> <td>    0.684</td> <td>    3.589</td> <td> 0.000</td> <td>    1.115</td> <td>    3.798</td>
</tr>
<tr>
  <th>V5</th>    <td>    2.3033</td> <td>    0.664</td> <td>    3.468</td> <td> 0.001</td> <td>    1.002</td> <td>    3.605</td>
</tr>
<tr>
  <th>V8</th>    <td>    0.8816</td> <td>    0.473</td> <td>    1.864</td> <td> 0.062</td> <td>   -0.045</td> <td>    1.808</td>
</tr>
<tr>
  <th>V9</th>    <td>    2.2908</td> <td>    0.806</td> <td>    2.843</td> <td> 0.004</td> <td>    0.711</td> <td>    3.870</td>
</tr>
<tr>
  <th>V11</th>   <td>   -1.7774</td> <td>    0.663</td> <td>   -2.682</td> <td> 0.007</td> <td>   -3.076</td> <td>   -0.478</td>
</tr>
<tr>
  <th>V22</th>   <td>   -2.0083</td> <td>    0.513</td> <td>   -3.912</td> <td> 0.000</td> <td>   -3.015</td> <td>   -1.002</td>
</tr>
<tr>
  <th>V27</th>   <td>   -2.5563</td> <td>    0.595</td> <td>   -4.294</td> <td> 0.000</td> <td>   -3.723</td> <td>   -1.390</td>
</tr>
<tr>
  <th>V31</th>   <td>    1.3486</td> <td>    0.470</td> <td>    2.870</td> <td> 0.004</td> <td>    0.428</td> <td>    2.270</td>
</tr>
</table>




```python
vif=pd.DataFrame()
vif["Features"]=X_train.columns
vif["VIF"]=[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
vif["VIF"]=round(vif["VIF"])
vif=vif.sort_values(by="VIF",ascending=False)
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>V9</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>V3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V11</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V27</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>V31</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V8</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V22</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train.drop("V8",axis=1,inplace=True)
```


```python
logm1=sm.GLM(y_train,(sm.add_constant(X_train)),family=sm.families.Binomial())
logm1.fit().summary()
```

    C:\Users\kosha\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Class</td>      <th>  No. Observations:  </th>  <td>   245</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   237</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     7</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -87.816</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sun, 14 Feb 2021</td> <th>  Deviance:          </th> <td>  175.63</td>
</tr>
<tr>
  <th>Time:</th>                <td>11:42:52</td>     <th>  Pearson chi2:      </th>  <td>  549.</td> 
</tr>
<tr>
  <th>No. Iterations:</th>          <td>6</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>   -1.6593</td> <td>    0.483</td> <td>   -3.437</td> <td> 0.001</td> <td>   -2.606</td> <td>   -0.713</td>
</tr>
<tr>
  <th>V3</th>    <td>    2.6726</td> <td>    0.675</td> <td>    3.958</td> <td> 0.000</td> <td>    1.349</td> <td>    3.996</td>
</tr>
<tr>
  <th>V5</th>    <td>    2.5148</td> <td>    0.673</td> <td>    3.737</td> <td> 0.000</td> <td>    1.196</td> <td>    3.834</td>
</tr>
<tr>
  <th>V9</th>    <td>    1.8493</td> <td>    0.768</td> <td>    2.408</td> <td> 0.016</td> <td>    0.344</td> <td>    3.355</td>
</tr>
<tr>
  <th>V11</th>   <td>   -1.7062</td> <td>    0.645</td> <td>   -2.645</td> <td> 0.008</td> <td>   -2.971</td> <td>   -0.442</td>
</tr>
<tr>
  <th>V22</th>   <td>   -2.1630</td> <td>    0.500</td> <td>   -4.330</td> <td> 0.000</td> <td>   -3.142</td> <td>   -1.184</td>
</tr>
<tr>
  <th>V27</th>   <td>   -2.8096</td> <td>    0.606</td> <td>   -4.639</td> <td> 0.000</td> <td>   -3.997</td> <td>   -1.623</td>
</tr>
<tr>
  <th>V31</th>   <td>    1.2791</td> <td>    0.481</td> <td>    2.661</td> <td> 0.008</td> <td>    0.337</td> <td>    2.221</td>
</tr>
</table>




```python
vif=pd.DataFrame()
vif["Features"]=X_train.columns
vif["VIF"]=[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
vif["VIF"]=round(vif["VIF"])
vif=vif.sort_values(by="VIF",ascending=False)
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>V9</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>V3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V11</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V27</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V31</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V22</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train.drop("V9",axis=1,inplace=True)
```


```python
logm1=sm.GLM(y_train,(sm.add_constant(X_train)),family=sm.families.Binomial())
logm1.fit().summary()
```

    C:\Users\kosha\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Class</td>      <th>  No. Observations:  </th>  <td>   245</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   238</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     6</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -91.173</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sun, 14 Feb 2021</td> <th>  Deviance:          </th> <td>  182.35</td>
</tr>
<tr>
  <th>Time:</th>                <td>11:43:54</td>     <th>  Pearson chi2:      </th>  <td>  354.</td> 
</tr>
<tr>
  <th>No. Iterations:</th>          <td>6</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>   -1.6375</td> <td>    0.478</td> <td>   -3.427</td> <td> 0.001</td> <td>   -2.574</td> <td>   -0.701</td>
</tr>
<tr>
  <th>V3</th>    <td>    2.8463</td> <td>    0.653</td> <td>    4.356</td> <td> 0.000</td> <td>    1.565</td> <td>    4.127</td>
</tr>
<tr>
  <th>V5</th>    <td>    2.8874</td> <td>    0.673</td> <td>    4.291</td> <td> 0.000</td> <td>    1.569</td> <td>    4.206</td>
</tr>
<tr>
  <th>V11</th>   <td>   -0.7431</td> <td>    0.510</td> <td>   -1.456</td> <td> 0.145</td> <td>   -1.743</td> <td>    0.257</td>
</tr>
<tr>
  <th>V22</th>   <td>   -2.0274</td> <td>    0.485</td> <td>   -4.184</td> <td> 0.000</td> <td>   -2.977</td> <td>   -1.078</td>
</tr>
<tr>
  <th>V27</th>   <td>   -2.4665</td> <td>    0.551</td> <td>   -4.477</td> <td> 0.000</td> <td>   -3.546</td> <td>   -1.387</td>
</tr>
<tr>
  <th>V31</th>   <td>    1.0500</td> <td>    0.464</td> <td>    2.262</td> <td> 0.024</td> <td>    0.140</td> <td>    1.960</td>
</tr>
</table>




```python
vif=pd.DataFrame()
vif["Features"]=X_train.columns
vif["VIF"]=[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
vif["VIF"]=round(vif["VIF"])
vif=vif.sort_values(by="VIF",ascending=False)
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>V5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>V3</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V11</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V27</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V31</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V22</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train_pred=res.predict(X_train_sm)
```


```python
round(y_train_pred[:10],4)
```




    139    0.8651
    197    0.9511
    315    0.9974
    205    0.9251
    2      0.9762
    201    0.9390
    141    0.9440
    310    0.9991
    218    0.0000
    320    0.9548
    dtype: float64




```python
y_train_pred = round(y_train_pred,4)
```


```python
y_train_pred[:10]
```




    139    0.8651
    197    0.9511
    315    0.9974
    205    0.9251
    2      0.9762
    201    0.9390
    141    0.9440
    310    0.9991
    218    0.0000
    320    0.9548
    dtype: float64




```python
y_train_pred_final=pd.DataFrame({"Converted":y_train.values,"Conversion_prob":y_train_pred})
```


```python
y_train_pred_final.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Converted</th>
      <th>Conversion_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>139</th>
      <td>1</td>
      <td>0.8651</td>
    </tr>
    <tr>
      <th>197</th>
      <td>1</td>
      <td>0.9511</td>
    </tr>
    <tr>
      <th>315</th>
      <td>1</td>
      <td>0.9974</td>
    </tr>
    <tr>
      <th>205</th>
      <td>1</td>
      <td>0.9251</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.9762</td>
    </tr>
    <tr>
      <th>201</th>
      <td>1</td>
      <td>0.9390</td>
    </tr>
    <tr>
      <th>141</th>
      <td>1</td>
      <td>0.9440</td>
    </tr>
    <tr>
      <th>310</th>
      <td>1</td>
      <td>0.9991</td>
    </tr>
    <tr>
      <th>218</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>320</th>
      <td>1</td>
      <td>0.9548</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train_pred_final["predicted"]=y_train_pred_final.Conversion_prob.map(lambda x: 1 if x>0.5 else 0 )
```


```python
y_train_pred_final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Converted</th>
      <th>Conversion_prob</th>
      <th>predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>139</th>
      <td>1</td>
      <td>0.8651</td>
      <td>1</td>
    </tr>
    <tr>
      <th>197</th>
      <td>1</td>
      <td>0.9511</td>
      <td>1</td>
    </tr>
    <tr>
      <th>315</th>
      <td>1</td>
      <td>0.9974</td>
      <td>1</td>
    </tr>
    <tr>
      <th>205</th>
      <td>1</td>
      <td>0.9251</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.9762</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn import metrics
```


```python
confusion=metrics.confusion_matrix(y_train_pred_final.Converted,y_train_pred_final.predicted)
```


```python
confusion
```




    array([[ 70,  18],
           [  9, 148]], dtype=int64)




```python
confusion.sum()
```




    245




```python
metrics.accuracy_score(y_train_pred_final.Converted,y_train_pred_final.predicted)
```




    0.889795918367347




```python
TP= confusion[1,1]
TN = confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]
```


```python
#sensitivity
TP/(TP+FN)
```




    0.9426751592356688




```python
#specificity
TN/(TN+FP)
```




    0.7954545454545454




```python
def draw_roc(actual,probs):
    fpr,tpr,thresholds=metrics.roc_curve(actual,probs,drop_intermediate=False)
    auc_score=metrics.roc_auc_score(actual,probs)
    plt.figure(figsize=(5,5))
    plt.plot(fpr,tpr,label="ROC curve (area=%0.2f)" % auc_score)
    plt.plot([0,1],[0,1],"k--")
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.05)
    plt.legend(loc="lower right")
    plt.show()
    
    return None
```


```python
fpr,tpr,thresholds=metrics.roc_curve(y_train_pred_final.Converted,y_train_pred_final.Conversion_prob)
```

# ROC curve


```python
draw_roc(y_train_pred_final.Converted,y_train_pred_final.Conversion_prob)
```


![png](Case_study_files/Case_study_71_0.png)


### Feature Engineering is done using Stats Model (RFE). 
### Logistic Regression is used for model building
### Accuracy is not the only parameter used to test model. Along with accuracy, specificity and sensivity are also the model testing parameters.

### Accuracy of model on training data is 89 percent

# Model Testing


```python
X_test=X_test[col]
```


```python
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V3</th>
      <th>V5</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V11</th>
      <th>V22</th>
      <th>V27</th>
      <th>V31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>-0.01864</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.11470</td>
      <td>-0.45663</td>
      <td>0.00000</td>
      <td>0.20645</td>
      <td>0.16595</td>
    </tr>
    <tr>
      <th>284</th>
      <td>1</td>
      <td>0.29073</td>
      <td>0.23308</td>
      <td>0.03759</td>
      <td>0.34336</td>
      <td>0.12030</td>
      <td>0.06266</td>
      <td>-0.00280</td>
      <td>-0.00886</td>
      <td>0.00096</td>
    </tr>
    <tr>
      <th>308</th>
      <td>1</td>
      <td>0.75564</td>
      <td>0.83550</td>
      <td>0.54916</td>
      <td>0.72063</td>
      <td>0.35225</td>
      <td>0.13469</td>
      <td>0.24956</td>
      <td>-0.70352</td>
      <td>-0.34131</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.06682</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.18949</td>
      <td>0.87280</td>
      <td>0.78479</td>
    </tr>
    <tr>
      <th>335</th>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.00224</td>
      <td>1.00000</td>
      <td>0.97763</td>
      <td>0.09396</td>
      <td>0.99989</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test_sm=sm.add_constant(X_test[col])
```

    C:\Users\kosha\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    


```python
X_test_sm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>V1</th>
      <th>V3</th>
      <th>V5</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V11</th>
      <th>V22</th>
      <th>V27</th>
      <th>V31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>1</td>
      <td>-0.01864</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.11470</td>
      <td>-0.45663</td>
      <td>0.00000</td>
      <td>0.20645</td>
      <td>0.16595</td>
    </tr>
    <tr>
      <th>284</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.29073</td>
      <td>0.23308</td>
      <td>0.03759</td>
      <td>0.34336</td>
      <td>0.12030</td>
      <td>0.06266</td>
      <td>-0.00280</td>
      <td>-0.00886</td>
      <td>0.00096</td>
    </tr>
    <tr>
      <th>308</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.75564</td>
      <td>0.83550</td>
      <td>0.54916</td>
      <td>0.72063</td>
      <td>0.35225</td>
      <td>0.13469</td>
      <td>0.24956</td>
      <td>-0.70352</td>
      <td>-0.34131</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.06682</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.18949</td>
      <td>0.87280</td>
      <td>0.78479</td>
    </tr>
    <tr>
      <th>335</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.00224</td>
      <td>1.00000</td>
      <td>0.97763</td>
      <td>0.09396</td>
      <td>0.99989</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>122</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.73810</td>
      <td>-0.76190</td>
      <td>0.33333</td>
      <td>-0.14286</td>
      <td>0.45238</td>
      <td>-0.67285</td>
      <td>0.04762</td>
      <td>0.24889</td>
      <td>-0.66667</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.0</td>
      <td>0</td>
      <td>1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>-1.00000</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-1.00000</td>
    </tr>
    <tr>
      <th>266</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.76627</td>
      <td>0.63935</td>
      <td>0.48409</td>
      <td>0.52500</td>
      <td>0.15000</td>
      <td>0.13753</td>
      <td>0.18864</td>
      <td>-0.33942</td>
      <td>-0.19962</td>
    </tr>
    <tr>
      <th>190</th>
      <td>1.0</td>
      <td>0</td>
      <td>-1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.99539</td>
      <td>0.85243</td>
      <td>0.83398</td>
      <td>-0.37708</td>
      <td>1.00000</td>
      <td>0.85243</td>
      <td>-0.29674</td>
      <td>0.41078</td>
      <td>0.42267</td>
    </tr>
    <tr>
      <th>108</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.10135</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.54730</td>
      <td>0.31081</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>272</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.63510</td>
      <td>0.76530</td>
      <td>0.61432</td>
      <td>0.36028</td>
      <td>0.65358</td>
      <td>0.64203</td>
      <td>-0.25404</td>
      <td>0.68775</td>
      <td>0.66316</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.84134</td>
      <td>0.43644</td>
      <td>0.93421</td>
      <td>-0.00267</td>
      <td>0.87947</td>
      <td>0.81121</td>
      <td>1.00000</td>
      <td>0.93358</td>
      <td>0.84463</td>
    </tr>
    <tr>
      <th>177</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.91667</td>
      <td>0.83333</td>
      <td>0.70833</td>
      <td>0.25000</td>
      <td>0.87500</td>
      <td>0.91667</td>
      <td>-0.08333</td>
      <td>0.83796</td>
      <td>0.70833</td>
    </tr>
    <tr>
      <th>294</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.74704</td>
      <td>0.53755</td>
      <td>0.72727</td>
      <td>0.09486</td>
      <td>0.69565</td>
      <td>0.66798</td>
      <td>-0.25296</td>
      <td>0.61847</td>
      <td>0.37549</td>
    </tr>
    <tr>
      <th>208</th>
      <td>1.0</td>
      <td>1</td>
      <td>-0.00641</td>
      <td>0.00000</td>
      <td>-0.01923</td>
      <td>1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.74359</td>
      <td>-0.61538</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>147</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.06486</td>
      <td>0.95135</td>
      <td>0.98919</td>
      <td>0.06486</td>
      <td>0.98556</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>133</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.34694</td>
      <td>0.46939</td>
      <td>0.40816</td>
      <td>0.20408</td>
      <td>0.46939</td>
      <td>0.30612</td>
      <td>0.44898</td>
      <td>-0.46579</td>
      <td>-0.34694</td>
    </tr>
    <tr>
      <th>163</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.85209</td>
      <td>0.38887</td>
      <td>0.08858</td>
      <td>0.98903</td>
      <td>-0.42625</td>
      <td>-0.76229</td>
      <td>-0.80496</td>
      <td>0.71698</td>
      <td>0.19667</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.05131</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.31558</td>
      <td>0.72529</td>
      <td>0.54356</td>
    </tr>
    <tr>
      <th>225</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.92308</td>
      <td>0.86399</td>
      <td>0.72582</td>
      <td>0.36790</td>
      <td>0.70588</td>
      <td>0.57449</td>
      <td>0.76969</td>
      <td>-0.38798</td>
      <td>-0.53146</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1.0</td>
      <td>0</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-1.00000</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>-1.00000</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>0.81309</td>
      <td>0.43019</td>
      <td>1.00000</td>
      <td>0.20619</td>
      <td>-0.43872</td>
      <td>-0.76339</td>
      <td>0.41778</td>
      <td>0.93570</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.0</td>
      <td>0</td>
      <td>1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>348</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.94701</td>
      <td>0.93207</td>
      <td>0.95177</td>
      <td>-0.03431</td>
      <td>0.95584</td>
      <td>0.94124</td>
      <td>0.07677</td>
      <td>0.92489</td>
      <td>0.92459</td>
    </tr>
    <tr>
      <th>278</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.98822</td>
      <td>0.93102</td>
      <td>0.83904</td>
      <td>0.35222</td>
      <td>0.74706</td>
      <td>0.73584</td>
      <td>0.73696</td>
      <td>-0.16556</td>
      <td>-0.34717</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.22574</td>
      <td>0.98602</td>
      <td>0.94930</td>
      <td>0.31698</td>
      <td>0.68452</td>
      <td>0.56791</td>
    </tr>
    <tr>
      <th>179</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.99246</td>
      <td>-0.29802</td>
      <td>1.00000</td>
      <td>0.96662</td>
      <td>-0.78316</td>
      <td>0.48757</td>
      <td>0.24117</td>
    </tr>
    <tr>
      <th>131</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.64122</td>
      <td>0.34146</td>
      <td>0.52751</td>
      <td>0.03466</td>
      <td>0.19512</td>
      <td>0.43313</td>
      <td>0.12195</td>
      <td>0.19304</td>
      <td>0.19512</td>
    </tr>
    <tr>
      <th>161</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.85736</td>
      <td>0.81927</td>
      <td>0.77521</td>
      <td>-0.04182</td>
      <td>0.84317</td>
      <td>0.86258</td>
      <td>0.01195</td>
      <td>0.82383</td>
      <td>0.73936</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.10991</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>259</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.68317</td>
      <td>0.84803</td>
      <td>0.84341</td>
      <td>0.00301</td>
      <td>0.84300</td>
      <td>0.75813</td>
      <td>0.00909</td>
      <td>0.70824</td>
      <td>0.66624</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.08333</td>
      <td>-1.00000</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>0.71875</td>
      <td>-0.82143</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>-1.00000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.88305</td>
      <td>1.00000</td>
      <td>0.82403</td>
      <td>0.19206</td>
      <td>0.85086</td>
      <td>0.90558</td>
      <td>0.27575</td>
      <td>0.69811</td>
      <td>0.43026</td>
    </tr>
    <tr>
      <th>233</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.50466</td>
      <td>0.71442</td>
      <td>0.71063</td>
      <td>0.02258</td>
      <td>0.68065</td>
      <td>0.34615</td>
      <td>0.06856</td>
      <td>0.59935</td>
      <td>0.56466</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-1.00000</td>
      <td>-1.00000</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>-1.00000</td>
    </tr>
    <tr>
      <th>264</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.94631</td>
      <td>0.90946</td>
      <td>0.85096</td>
      <td>0.49960</td>
      <td>0.73678</td>
      <td>0.59215</td>
      <td>0.89383</td>
      <td>-0.58563</td>
      <td>-0.75321</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.02337</td>
      <td>-0.09924</td>
      <td>-0.00763</td>
      <td>-0.11824</td>
      <td>0.14706</td>
      <td>0.03786</td>
      <td>0.03669</td>
      <td>-0.03240</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.71216</td>
      <td>-1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>0.51613</td>
      <td>0.25682</td>
    </tr>
    <tr>
      <th>187</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.77123</td>
      <td>1.00000</td>
      <td>-0.33333</td>
      <td>-1.00000</td>
      <td>-0.92536</td>
      <td>0.19235</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>309</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.83789</td>
      <td>0.72113</td>
      <td>0.45625</td>
      <td>0.78115</td>
      <td>0.16470</td>
      <td>-0.13012</td>
      <td>-0.31351</td>
      <td>-0.05054</td>
      <td>0.37093</td>
    </tr>
    <tr>
      <th>255</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.83427</td>
      <td>0.54040</td>
      <td>0.12326</td>
      <td>0.89402</td>
      <td>-0.33221</td>
      <td>-0.70086</td>
      <td>-0.89064</td>
      <td>0.83008</td>
      <td>0.37542</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.26968</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.61267</td>
      <td>0.89502</td>
      <td>0.74389</td>
    </tr>
    <tr>
      <th>319</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.85013</td>
      <td>0.92211</td>
      <td>0.92046</td>
      <td>0.02180</td>
      <td>0.92765</td>
      <td>0.87597</td>
      <td>0.07047</td>
      <td>0.86923</td>
      <td>0.85198</td>
    </tr>
    <tr>
      <th>153</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.94653</td>
      <td>0.72554</td>
      <td>0.47564</td>
      <td>0.82455</td>
      <td>0.01267</td>
      <td>-0.24871</td>
      <td>-0.44000</td>
      <td>0.16085</td>
      <td>0.46059</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>0.00000</td>
      <td>0.77941</td>
      <td>-0.99265</td>
      <td>0.80882</td>
      <td>-0.41912</td>
      <td>0.00000</td>
      <td>-1.00000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>274</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.71521</td>
      <td>0.66667</td>
      <td>0.63107</td>
      <td>-0.05178</td>
      <td>0.77994</td>
      <td>0.67314</td>
      <td>0.14887</td>
      <td>0.45105</td>
      <td>0.49191</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.0</td>
      <td>0</td>
      <td>-1.00000</td>
      <td>0.00000</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.00000</td>
      <td>-1.00000</td>
      <td>-1.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.98166</td>
      <td>0.98103</td>
      <td>0.97565</td>
      <td>-0.05699</td>
      <td>0.95947</td>
      <td>0.99004</td>
      <td>-0.17387</td>
      <td>0.81813</td>
      <td>0.76911</td>
    </tr>
    <tr>
      <th>291</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.59887</td>
      <td>0.69868</td>
      <td>0.85122</td>
      <td>-0.13936</td>
      <td>0.80979</td>
      <td>0.50471</td>
      <td>-0.00377</td>
      <td>0.61444</td>
      <td>0.53861</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>248</th>
      <td>1.0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>0.94140</td>
      <td>0.92106</td>
      <td>-0.23255</td>
      <td>0.77152</td>
      <td>0.52798</td>
      <td>-0.35575</td>
      <td>0.13290</td>
      <td>-0.05707</td>
    </tr>
    <tr>
      <th>198</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.36876</td>
      <td>-1.00000</td>
      <td>-0.07661</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.74597</td>
      <td>-0.20188</td>
      <td>0.03443</td>
      <td>-0.00761</td>
    </tr>
    <tr>
      <th>69</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>113</th>
      <td>1.0</td>
      <td>1</td>
      <td>1.00000</td>
      <td>0.91404</td>
      <td>0.78020</td>
      <td>0.72144</td>
      <td>0.47660</td>
      <td>0.27639</td>
      <td>0.44614</td>
      <td>-0.71731</td>
      <td>-0.51251</td>
    </tr>
    <tr>
      <th>63</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.63816</td>
      <td>0.20833</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.87719</td>
      <td>-0.66886</td>
      <td>-0.43860</td>
      <td>1.00000</td>
      <td>0.20614</td>
    </tr>
    <tr>
      <th>188</th>
      <td>1.0</td>
      <td>0</td>
      <td>-1.00000</td>
      <td>-1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>-1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.88208</td>
      <td>0.93408</td>
      <td>0.92100</td>
      <td>-0.16450</td>
      <td>0.88307</td>
      <td>0.88462</td>
      <td>-0.47153</td>
      <td>0.56804</td>
      <td>0.47075</td>
    </tr>
    <tr>
      <th>305</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.52542</td>
      <td>0.94915</td>
      <td>0.52542</td>
      <td>-0.16949</td>
      <td>0.30508</td>
      <td>0.50847</td>
      <td>0.30508</td>
      <td>0.67287</td>
      <td>0.83051</td>
    </tr>
  </tbody>
</table>
<p>106 rows × 11 columns</p>
</div>




```python
y_test_pred=res.predict(X_test_sm)
```


```python
y_test_pred
```




    9      4.232862e-01
    284    5.235120e-01
    308    9.904594e-01
    56     9.467413e-01
    335    9.545678e-01
    122    7.148871e-01
    21     7.093148e-16
    266    9.524813e-01
    190    6.474929e-16
    0      9.442649e-01
    108    1.797969e-02
    272    8.938767e-01
    50     7.862365e-01
    177    9.077416e-01
    294    8.204557e-01
    208    7.551649e-01
    147    9.455560e-01
    133    7.142556e-01
    163    9.650770e-01
    24     9.405862e-01
    225    9.246413e-01
    29     7.329157e-19
    68     9.997250e-01
    19     1.519091e-12
    348    9.385591e-01
    278    9.197959e-01
    60     9.521235e-01
    179    9.355229e-01
    131    6.307008e-01
    161    8.633711e-01
               ...     
    49     1.859656e-02
    259    8.760861e-01
    55     5.195827e-02
    90     8.388319e-01
    233    8.725435e-01
    17     6.818894e-14
    264    9.488635e-01
    5      1.664595e-01
    3      4.463349e-01
    187    9.993686e-01
    309    9.966391e-01
    255    9.717018e-01
    70     9.287028e-01
    319    9.271804e-01
    153    9.971559e-01
    100    9.999620e-01
    274    8.500030e-01
    27     6.513582e-12
    48     9.437700e-01
    291    8.782336e-01
    31     6.165474e-01
    248    1.488825e-14
    4      9.606364e-01
    198    6.871908e-01
    69     6.165474e-01
    113    9.940836e-01
    63     9.973986e-01
    188    1.384392e-17
    32     9.226022e-01
    305    5.980709e-01
    Length: 106, dtype: float64




```python
y_pred_1=pd.DataFrame(y_test_pred)
```


```python
y_test_df=pd.DataFrame(y_test)
y_pred_1.reset_index(drop=True,inplace=True)
y_test_df.reset_index(drop=True,inplace=True)
```


```python
y_pred_1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.232862e-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.235120e-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.904594e-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.467413e-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.545678e-01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7.148871e-01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.093148e-16</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9.524813e-01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6.474929e-16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.442649e-01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.797969e-02</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8.938767e-01</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7.862365e-01</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9.077416e-01</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8.204557e-01</td>
    </tr>
    <tr>
      <th>15</th>
      <td>7.551649e-01</td>
    </tr>
    <tr>
      <th>16</th>
      <td>9.455560e-01</td>
    </tr>
    <tr>
      <th>17</th>
      <td>7.142556e-01</td>
    </tr>
    <tr>
      <th>18</th>
      <td>9.650770e-01</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9.405862e-01</td>
    </tr>
    <tr>
      <th>20</th>
      <td>9.246413e-01</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7.329157e-19</td>
    </tr>
    <tr>
      <th>22</th>
      <td>9.997250e-01</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.519091e-12</td>
    </tr>
    <tr>
      <th>24</th>
      <td>9.385591e-01</td>
    </tr>
    <tr>
      <th>25</th>
      <td>9.197959e-01</td>
    </tr>
    <tr>
      <th>26</th>
      <td>9.521235e-01</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9.355229e-01</td>
    </tr>
    <tr>
      <th>28</th>
      <td>6.307008e-01</td>
    </tr>
    <tr>
      <th>29</th>
      <td>8.633711e-01</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1.859656e-02</td>
    </tr>
    <tr>
      <th>77</th>
      <td>8.760861e-01</td>
    </tr>
    <tr>
      <th>78</th>
      <td>5.195827e-02</td>
    </tr>
    <tr>
      <th>79</th>
      <td>8.388319e-01</td>
    </tr>
    <tr>
      <th>80</th>
      <td>8.725435e-01</td>
    </tr>
    <tr>
      <th>81</th>
      <td>6.818894e-14</td>
    </tr>
    <tr>
      <th>82</th>
      <td>9.488635e-01</td>
    </tr>
    <tr>
      <th>83</th>
      <td>1.664595e-01</td>
    </tr>
    <tr>
      <th>84</th>
      <td>4.463349e-01</td>
    </tr>
    <tr>
      <th>85</th>
      <td>9.993686e-01</td>
    </tr>
    <tr>
      <th>86</th>
      <td>9.966391e-01</td>
    </tr>
    <tr>
      <th>87</th>
      <td>9.717018e-01</td>
    </tr>
    <tr>
      <th>88</th>
      <td>9.287028e-01</td>
    </tr>
    <tr>
      <th>89</th>
      <td>9.271804e-01</td>
    </tr>
    <tr>
      <th>90</th>
      <td>9.971559e-01</td>
    </tr>
    <tr>
      <th>91</th>
      <td>9.999620e-01</td>
    </tr>
    <tr>
      <th>92</th>
      <td>8.500030e-01</td>
    </tr>
    <tr>
      <th>93</th>
      <td>6.513582e-12</td>
    </tr>
    <tr>
      <th>94</th>
      <td>9.437700e-01</td>
    </tr>
    <tr>
      <th>95</th>
      <td>8.782336e-01</td>
    </tr>
    <tr>
      <th>96</th>
      <td>6.165474e-01</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1.488825e-14</td>
    </tr>
    <tr>
      <th>98</th>
      <td>9.606364e-01</td>
    </tr>
    <tr>
      <th>99</th>
      <td>6.871908e-01</td>
    </tr>
    <tr>
      <th>100</th>
      <td>6.165474e-01</td>
    </tr>
    <tr>
      <th>101</th>
      <td>9.940836e-01</td>
    </tr>
    <tr>
      <th>102</th>
      <td>9.973986e-01</td>
    </tr>
    <tr>
      <th>103</th>
      <td>1.384392e-17</td>
    </tr>
    <tr>
      <th>104</th>
      <td>9.226022e-01</td>
    </tr>
    <tr>
      <th>105</th>
      <td>5.980709e-01</td>
    </tr>
  </tbody>
</table>
<p>106 rows × 1 columns</p>
</div>




```python
y_pred_final=pd.concat([y_pred_1,y_test_df],axis=1)
```


```python
y_pred_final=y_pred_final.rename(columns={0:"Conversion_Prob"})
```


```python
y_pred_final.head(7)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Conversion_Prob</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.232862e-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.235120e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.904594e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.467413e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.545678e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7.148871e-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.093148e-16</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_pred_final["final_predicted"]=y_pred_final["Conversion_Prob"].map(lambda x: 1 if x>0.42 else 0)
```


```python
metrics.accuracy_score(y_pred_final["Class"],y_pred_final["final_predicted"])
```




    0.8584905660377359



### So, the model is 86 percent accurate on unseen data

### It can be concluded that model is learning good and it is neither underfitting nor overfitting as accuracy for testing and training dataset are ideal.


```python

```
