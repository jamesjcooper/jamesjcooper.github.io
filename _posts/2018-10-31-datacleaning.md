---
title: "Exploratory Analysis and Some Simple Data Cleaning Techniques"
date: 2018-10-31
#header:
#  image: "/images/datacleaning/cleaning.jpg"
excerpt: "Data Cleaning, Data Science"
mathjax: "true"
---

We're all familiar with the fact that data cleaning is a huge part of data science. 
The truth is, data is messy, and if it wasn't it wouldn't be nearly as hard 
to analyze it. When thought of like most anything else, a lot of groundwork goes 
into constructing something impressive. An architect relies on surverying, planning, 
and blueprints. Athletes must build up fitness to achieve peak performance, and data 
scientists must clean data in order to model it. It's not glamorous, but it is *extremely* 
important in this realm. This is the reasoning behind this post about a few simple
beginning techniques that can be used to explore your data, take care of missing data, 
transform your data, along with identifying and taking care of outliers. These techniques 
represent just a few of a whole multitude of techniques, but provide a good start. 

# Clean Data, the Final Frontier

One of the most important things you will quickly learn in data science is that you must,
must know what your data is all about before diving into anything. Where are the issues?
Are there outliers? Missing Data? Weird values? Are there some easily detectable patterns 
when first observing? This is the ground level exploration that must be completed. To know
your data is to love your data. It should also be noted that 
this stretches beyond just coding - you must talk to the domain experts around you! Before diving 
into any project, I like the idea of meeting with anyone you can who is involved with the data,
be it the sales team that might use the data, or the business development team who will present results, 
in order to fully understand the objectives of what you or they want out of the data. This 
can guide your strategy when it comes time to dive in. 

In this example I am using a sample bank dataset with features such as age, income, region, gender, 
children, and categorical data consisting of 'yes' and 'no' answers regarding if a particular 
subject is married, has a savings account, mortgage, etc. Each row is a different client with 
a unique id, and each column consists of the variables mentioned before. 

The first thing I like to do would be to get the layout of the dataset, so looking at the columns and the
descriptive statistics to get an idea of the data types, the amount of missing values, the mean, count, 
standard deviation, etc. After reading in the file, here is an example of breaking it down:

Read the file in and look at the first five rows:
![alt]({{ site.url }}{{ site.baseurl }}/images/datacleaning/head.JPG)

I like to get some basic info, and then reset the index and calculate the percentage of complete data you have,
data with no missing values in each column. Here we can see the 26% of the age category is missing, nearly 19% 
of the income category, etc. Also taking the sum using the isna() function in Python will give the count of missing
values per column. To get the row count we would just specify axis = 1. Below is just column count:
```python
bank_data.columns
```




    Index(['id', 'age', 'income', 'children', 'gender', 'region', 'married', 'car',
           'savings_acct', 'current_acct', 'mortgage', 'pep'],
          dtype='object')




```python
## Count of instances and features
rows, columns = bank_data.shape
print (bank_data.shape)
```

    (600, 12)
    


```python
breakdown = bank_data.describe(include='all').T.reset_index()
breakdown.rename(columns={'index':'feature'},inplace=True)
breakdown.insert(1,'missing',(rows - breakdown['count'])/float(rows))
breakdown
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
      <th>feature</th>
      <th>missing</th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id</td>
      <td>0</td>
      <td>600</td>
      <td>600</td>
      <td>ID12429</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>age</td>
      <td>0.268333</td>
      <td>439</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42.3212</td>
      <td>14.3532</td>
      <td>18</td>
      <td>30</td>
      <td>42</td>
      <td>54</td>
      <td>67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>income</td>
      <td>0.188333</td>
      <td>487</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27190.1</td>
      <td>12882</td>
      <td>5014.21</td>
      <td>16780.3</td>
      <td>24763.3</td>
      <td>35078.2</td>
      <td>63130.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>children</td>
      <td>0</td>
      <td>600</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.01167</td>
      <td>1.05675</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gender</td>
      <td>0.178333</td>
      <td>493</td>
      <td>2</td>
      <td>MALE</td>
      <td>247</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>region</td>
      <td>0.14</td>
      <td>516</td>
      <td>4</td>
      <td>INNER_CITY</td>
      <td>234</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>married</td>
      <td>0</td>
      <td>600</td>
      <td>2</td>
      <td>YES</td>
      <td>396</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>car</td>
      <td>0</td>
      <td>600</td>
      <td>2</td>
      <td>NO</td>
      <td>304</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>savings_acct</td>
      <td>0</td>
      <td>600</td>
      <td>2</td>
      <td>YES</td>
      <td>414</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>current_acct</td>
      <td>0</td>
      <td>600</td>
      <td>2</td>
      <td>YES</td>
      <td>455</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>mortgage</td>
      <td>0</td>
      <td>600</td>
      <td>2</td>
      <td>NO</td>
      <td>391</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>pep</td>
      <td>0</td>
      <td>600</td>
      <td>2</td>
      <td>NO</td>
      <td>326</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




 


And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/jamesjcooper)

Here's a bulleted list:
* First Item
- Second Item

Here's a numbered list:
1. First
2. Second
3. Third

Python code block:
```python

    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```
R code block:
```r

library(tidyverse)
df <- read_csv("some_file.csv")
head(df)

```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/datacleaning/cleaning.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/datacleaning/cleaning.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$