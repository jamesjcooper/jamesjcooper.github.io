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

```python
 bank_data.head()
```
![alt]({{ site.url }}{{ site.baseurl }}/images/datacleaning/head.JPG)

 


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