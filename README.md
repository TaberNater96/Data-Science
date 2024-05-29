<div align="center">
  <h2><b>Data Science Projects & Challenges<b></h2>
</div>

&nbsp;

<details>
  <summary><b>Click Here To Navigate To Each Repository<b></summary>

  - [Hybrid Text Summarization Engine](https://github.com/TaberNater96/Data-Science/tree/main/Hybrid%20Text%20Summarization%20Engine)
  - [US Wind Power Production with ARIMA](https://github.com/TaberNater96/Data-Science/blob/main/US%20Wind%20Power%20Production%20with%20ARIMA/US%20Wind%20Power%20Production%20with%20ARIMA.ipynb)
  - [NASA Meteorite Analysis](https://github.com/TaberNater96/Data-Science/tree/main/NASA%20Meteorites)
  - [TensorFlow Regression Challenge](https://github.com/TaberNater96/Data-Science/blob/main/TensorFlow%20Regression%20-%20Admission%20Scores/TensorFlow%20Regression%20Challenge.ipynb)
  - [PySpark - Web Crawler Analysis](https://github.com/TaberNater96/Data-Science/blob/main/Big%20Data%20with%20PySpark/PySpark%20-%20Web%20Crawler%20Analysis.ipynb)
  - [PySpark - Wikipedia Clickstream Data](https://github.com/TaberNater96/Data-Science/blob/main/Big%20Data%20with%20PySpark/PySpark%20-%20Wikipedia%20Clickstream%20Data.ipynb)
</details>

&nbsp;

This data science repository serves as a pivotal resource for navigating the complexities of data analysis and interpretation, symbolizing the future of technology where data science becomes the cornerstone. It highlights how data scientists play an essential role in bridging the gap between the intricate world of artificial intelligence (AI) and those without a technical background, making complex data accessible to all and enabling organizations to harness the full potential of AI.

Users navigating through this repository are introduced to the transformative truths hidden within datasets. The knowledge gained here covers a broad spectrum of data science methodologies, from the basics to advanced applications in machine learning and deep learning. This repository stands at the forefront of an era where emerging minds leverage data science, guiding novices through the data-driven landscape with dynamic processes that delve into the essence of data for analysis and interpretation.

The projects featured within this repository uncover patterns through both data visualization and statistical analysis, establishing a foundation for algorithms that predict outcomes and categorize information in real-life scenarios. Whether users are captivated by the visual beauty of data or the precision of predictive algorithms, this repository offers a first step towards a comprehensive understanding of data science. It combines creative insights with practical, example-driven knowledge, fostering independent exploration in the data science journey and demonstrating the transformative role of data science in connecting advanced technology with practical applications.

<div align="center">
  <h2>Quick Summaries (TLDR)</h2>
</div>

## Table of Contents
- [Hybrid Text Summarization Engine](#hybrid-text-summarization-engine)
- [US Wind Power Production with ARIMA](#us-wind-power-production-with-arima)
- [NASA Meteorite Analysis](#nasa-meteorite-analysis)
- [TensorFlow Regression Challenge](#tensorflow-regression-challenge)
- [PySpark - Web Crawler Analysis](#pyspark-web-crawler-analysis)
- [PySpark - Wikipedia Clickstream Data](#pyspark-wikipedia-clickstream-data)

<div id="hybrid-text-summarization-engine" align="center">
  <h2>Hybrid Text Summarization Engine</h2>
</div>

https://github.com/TaberNater96/Data-Science/assets/127979108/c95b2d02-4948-45a4-8678-1ac320443429

For my term project in my Data Mining: Text Analytics class, I created a hybrid text classification engine that utilizes the power of both an extractive and abstractive model, where the extractive model selects the most important information, while the abstractive model conveys the most critical information. Raw text is extracted from a PDF file and fed through a comprehensive pipeline to normalize, tokenize, and summarize. Text summarization is one of the core functionalities of Natural Language Processing (NLP) and by far one of the most useful, as it can be used to extract the main concepts from documents that are hundreds or even thousands of pages long. 

This repository is a full pipeline project that uses advanced pre-trained frameworks such as Hugging Face’s AutoTokenizer and Facebook’s BART to extract and summarize the most important information. The core interface is a custom-built streamlit app where the user can easily upload a PDF and the NLP pipeline will generate and output a summary, no matter how long or complicated the document. This project serves as a foundation for a larger and more in-depth NLP pipeline and is set up to be highly customizable and adjustable.

## Usage
**Clone the repository and run the following in the terminal:**

```sh
streamlit run main.py
```

<div id="us-wind-power-production-with-arima" align="center">
  <h2>US Wind Power Production with ARIMA</h2>
</div>

The "US Wind Power Production with ARIMA" notebook offers a detailed analysis aimed at understanding and forecasting US wind power production. After the data is prepared and cleaned, it addresses missing values and refines the dataset for analysis. A significant portion is dedicated to exploratory data analysis (EDA), identifying Texas as the leading state in wind power production and examining the correlation between production outputs across states. This suggests a shared influence of climatic conditions or energy policies. The notebook's core objective is to forecast future wind power production values using the ARIMA (AutoRegressive Integrated Moving Average) model, indicating a focus on predictive analytics to discern temporal patterns and guide future renewable energy strategies. The focus here is to showcase an ability to perform a deep and detailed analysis given a relatively small and limited dataset.

![Correlation Matrix](https://github.com/TaberNater96/Data-Science/blob/main/US%20Wind%20Power%20Production%20with%20ARIMA/Images/Correlation%20Matrix.png?raw=true)

Values between 0.90 and 0.99 indicate a very high positive correlation in wind power output across states, often due to similar climatic conditions and regional energy policies, leading to synchronous growth in wind energy. Conversely, dark blue values indicate a moderately inverse relationship, where an increase in wind energy in one state corresponds to a decrease in another, likely due to geographical and weather system diversity. Understanding these correlations is vital for energy planning, as states with high correlations might not ensure energy security during dips, whereas those with inverse relationships could help stabilize the national energy grid. 

#### ARIMA Model Results For National Wind Power Generation

![ARIMA Output](https://github.com/TaberNater96/Data-Science/blob/main/US%20Wind%20Power%20Production%20with%20ARIMA/Images/ARIMA%20Output.png?raw=true)

<div id="nasa-meteorite-analysis" align="center">
  <h2>NASA Meteorite Analysis</h2>
</div>

![Global Mass Distribution](https://github.com/TaberNater96/Data-Science/blob/main/NASA%20Meteorites/images/Global%20Mass%20Distribution.png?raw=true)

This project delved into NASA's meteorite dataset, uncovering unique insights into global meteorite characteristics, especially ancient, larger, relict meteorites found mainly near Earth's poles. The above screenshot is just a single photo of the interactive map that allows users to see the exact coordinates, mass, and name of all known meteorites on Earth when they hover over. This program is computationally expensive, hence a screenshot. The results highlighted how certain environmental conditions foster preservation, minimizing erosion. An advanced XGBoost Regression model was developed to predict future meteorite impacts with very high accuracy, showcasing the potential of machine learning in understanding celestial phenomena. 

![ML Output](https://github.com/TaberNater96/Data-Science/blob/main/NASA%20Meteorites/images/ML%20Output.png?raw=true)

<div id="tensorflow-regression-challenge" align="center">
  <h2>TensorFlow Regression Challenge</h2>
</div>

The purpose of this project is to create a deep learning regression model that predicts the likelihood that a student applying to graduate school will be accepted based on various application factors (such as test scores). The hope here is to give further insight into the graduate admissions process to improve test prep strategy. This model resulted in an MSE score of 0.0044 and an MAE score of 0.0505.

![MAE Score](https://github.com/TaberNater96/Data-Science/blob/main/TensorFlow%20Regression%20-%20Admission%20Scores/MAE%20Score.png?raw=true)

<div id="pyspark-web-crawler-analysis" align="center">
  <h2>PySpark - Web Crawler Analysis</h2>
</div>

This project leverages PySpark to analyze web domain data from the Common Crawl dataset, focusing on understanding the distribution of internet domains and their subdomains. By initializing Spark Sessions and Contexts, it reads domain information into resilient distributed datasets (RDDs) and DataFrames, demonstrating PySpark's ability to handle big data processing efficiently. The analysis involves formatting the data, extracting key domain metrics, and aggregating subdomain counts to unveil insights into the web's structure. Through transformations, aggregations, and SQL queries, the project explores domain counts across different top-level domains, calculates total subdomains, and filters specific entries, like government domains, to assess their internet presence. This use of PySpark illustrates its powerful capabilities in data manipulation, storage, and complex querying, showcasing how big data technologies can provide deep insights into vast datasets like Common Crawl.

<div id="pyspark-wikipedia-clickstream-data" align="center">
  <h2>PySpark - Wikipedia Clickstream Data</h2>
</div>

This project aims to demonstrate the application of PySpark for analyzing Wikipedia clickstream data. At the outset, it guides users through the initiation of a new PySpark session, setting the stage for data exploration and analysis. The core of this project involves the creation of a Resilient Distributed Dataset (RDD) derived from sample clickstream counts, illustrating the process of transforming raw data into a structured format suitable for analysis.

Throughout this endeavor, we focus on the manipulation of data within the PySpark RDD framework, specifically targeting clickstream data to uncover traffic patterns across Wikipedia pages. This involves a detailed examination of how users navigate through Wikipedia, identifying trends and insights that can inform our understanding of digital behavior and information flow on the web.

By leveraging PySpark's powerful data processing capabilities, this project offers a hands-on experience in handling large-scale data, providing a practical understanding of RDDs and their significance in distributed data processing. Through this analysis, we aim to shed light on the complexities of web traffic and user interactions on one of the world's largest repositories of knowledge, Wikipedia, thereby contributing to the broader field of data science and analytics.









