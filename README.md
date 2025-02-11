<div align="center">
  <h1><b>Data Science Projects & Challenges<b></h1>
</div>

&nbsp;

<details>
  <summary><b>Click Here To Navigate To Each Repository<b></summary>

  - [Tackle Opportunity Window](https://github.com/TaberNater96/Data-Science/tree/main/NFL%20Big%20Data%20Bowl%202024)
  - [Dashboards Built Using Tableau & Power BI](https://github.com/TaberNater96/Data-Science/tree/main/Dashboards)
  - [Hybrid Text Summarization Engine](https://github.com/TaberNater96/Data-Science/tree/main/Hybrid%20Text%20Summarization%20Engine)
  - [US Wind Power Production with ARIMA](https://github.com/TaberNater96/Data-Science/blob/main/US%20Wind%20Power%20Production%20with%20ARIMA/US%20Wind%20Power%20Production%20with%20ARIMA.ipynb)
  - [NASA Meteorite Analysis](https://github.com/TaberNater96/Data-Science/tree/main/NASA%20Meteorites)
  - [TensorFlow Regression Challenge](https://github.com/TaberNater96/Data-Science/blob/main/TensorFlow%20Regression%20-%20Admission%20Scores/TensorFlow%20Regression%20Challenge.ipynb)
  - [PySpark - Web Crawler Analysis](https://github.com/TaberNater96/Data-Science/blob/main/Big%20Data%20with%20PySpark/PySpark%20-%20Web%20Crawler%20Analysis.ipynb)
  - [PySpark - Wikipedia Clickstream Data](https://github.com/TaberNater96/Data-Science/blob/main/Big%20Data%20with%20PySpark/PySpark%20-%20Wikipedia%20Clickstream%20Data.ipynb)

</details>

&nbsp;

This repository showcases a collection of data science projects that show the power of transforming raw information into actionable insights. Through data visualization and statistical analysis, these projects demonstrate how patterns emerge from complexity, laying the groundwork for predictive algorithms with real-world applications. The work presented here serves as an entry point for understanding modern data science, appealing to both those who appreciate the aesthetic dimensions of data visualization and those interested in the mathematical precision of predictive modeling. At its core, this collection shows how data science serves as a crucial bridge between advanced artificial intelligence technologies and their practical implementation. The projects demonstrate the essential role data scientists play in translating complex analytical concepts into accessible insights, allowing organizations to leverage AI capabilities effectively and make data-driven decisions with confidence.

<div align="center">
  <h2>Quick Summaries (TLDR)</h2>
</div>

<a name="top"></a>
## Table of Contents

- [Tackle Opportunity Window](#tackle-opportunity-window)
- [Dashboards](#dashboards)
- [Hybrid Text Summarization Engine](#hybrid-text-summarization-engine)
- [US Wind Power Production with ARIMA](#us-wind-power-production-with-arima)
- [NASA Meteorite Analysis](#nasa-meteorite-analysis)
- [TensorFlow Regression Challenge](#tensorflow-regression-challenge)
- [PySpark - Web Crawler Analysis](#pyspark-web-crawler-analysis)
- [PySpark - Wikipedia Clickstream Data](#pyspark-wikipedia-clickstream-data)

<div id="tackle-opportunity-window" align="center">
  <h2>Tackle Opportunity Window</h2>
</div>

<a href="https://www.kaggle.com/code/godragons6/tackle-opportunity-window" target="_blank"><img align="left" alt="Kaggle" title="View Competition Submission" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

&nbsp;

Over the holidays of winter 2024, I entered a data science competition where hundreds of professional data scientists around the world competed to win a grand prize. My submission, which I named "Tackle Opportunity Window (TOW)," was presented at the NFL Big Data Bowl 2024 where I recieved an honorable mention. The goal of the competition was to design a new metric that can quantify the probability of a tackle occurring on a frame-by-frame basis. The TOW metric I designed quantifies the crucial timeframe a defender has to execute a tackle, showcasing my capability to handle and extract meaningful insights from extensive and complex datasets containing over 12 million rows of high dimensional data. What makes me proud of this project is that this competition was one of my first ever data science projects where python and machine learning were very new concepts to me. I actually joined this competition when it was over halfway done, but I decided to fully dive in, working through the holidays, and did not stop until the final model, simulation, and report was complete (refer to the NFL directory README for full a report). This competition repo differs from standard data science projects since the entire codebase is in a single notebook. This is so the notebook can be uploaded to Kaggle and viewed by other Data Scientists.

The neural network I developed for this project processes real-time player tracking data to predict tackle probabilities, a key output visualized using Plotly Express that I built from scratch. As the game progresses, each player's position relative to the ball carrier is fed into the model, which calculates the likelihood of a tackle. This probability is graphically represented by a dynamic 'probability bubble' around each player on the field. These bubbles grow in size as players enter closer proximity to the ball carrier, aligning with the increasing Tackle Opportunity Window (TOW). This visualization technique not only illustrates the model’s predictive capabilities but also provides an intuitive display of shifting tackle probabilities during live gameplay, highlighting the application of advanced neural network analysis in real-time sports scenarios.

<div align="center">
<img src="https://github.com/TaberNater96/Data-Science/blob/main/NFL%20Big%20Data%20Bowl%202024/images/Players/Marshon%20and%20Tyrann.png" width="800" height="90">
</div>

<div align="center">
<img src="https://github.com/TaberNater96/Data-Science/blob/main/NFL%20Big%20Data%20Bowl%202024/images/TOW%20Animation.gif" width="800" height="400">
</div>

In developing TOW, I leveraged a sophisticated approach to manage the sheer volume of data efficiently. The methodology involved calculating the dynamic Euclidean distance between a defender and the ball carrier across multiple frames, utilizing a vectorized computation method. This innovation not only optimized the processing of vast datasets but also illuminated hidden patterns within the chaotic and fast-paced movements of NFL games. The result is a unique, finely tuned metric that significantly enhances our understanding of defensive tactics and player effectiveness.

<div align="center">
<img src="https://github.com/TaberNater96/Data-Science/blob/main/NFL%20Big%20Data%20Bowl%202024/images/Players/Marshon%20and%20Tyrann.png" width="800" height="90">
</div>

<div align="center">
<img src="https://github.com/TaberNater96/Data-Science/blob/main/NFL%20Big%20Data%20Bowl%202024/images/TOW%20Plot%20Animation.gif" width="800" height="400">
</div>

Owing to the intrinsic capabilities of a neural network, particularly its adeptness in detecting nuanced variations, the direction and orientation of each player were pivotal in enabling the model to discern and adapt to subtle shifts. These shifts are essential for the model to recognize and adhere to an emergent pattern, as the features exhibit significant, yet controlled, variation across successive frames. This variation is not arbitrary, but rather demonstrative of a tackler pursuing their target with precision. The controlled variability within these features provides the model with critical data points, allowing it to effectively learn and predict the dynamics of a tackler's movement in relation to their target. Visualizing the distribution of each player's orientation and direction in the EDA phase, and noticing the non-random variation, is what gave rise to the idea of focusing on this specific concept in parallel with the tackle opportunity window. 

<div align="center">
<img src="https://github.com/TaberNater96/Data-Science/blob/main/NFL%20Big%20Data%20Bowl%202024/images/Polar%20Histogram.png" width="600" height="500">
</div>

To speak plainly, I am indeed the first to develop the TOW metric within the field of football analytics, demonstrating an ability to innovate and pioneer new methodologies in the realm of data science. This metric offers a fresh perspective by focusing on the precise moments a player remains within a strategic distance to make a successful tackle, thereby introducing a novel way to evaluate defensive actions. The project not only showcases my technical skills in handling large-scale data and complex statistical models but also highlights my creativity in generating new solutions to analyze performance metrics. By identifying and crafting a new evaluative metric, I have set a new standard in sports analytics for projects to build off of when analyzing tackle metrics, proving my capacity to lead and innovate in the field. This initiative demonstrates not just analytical acumen but also a visionary approach to transforming how data insights drive strategic decisions.

[⬆️ Back to table of contents](#top)

<div id="dashboards" align="center">
  <h2>Dashboards</h2>
</div>
<div>
  <p>
    All dashboards can be downloaded and interacted with from the <a href="https://github.com/TaberNater96/Data-Science/tree/main/Dashboards/Interactive%20Dashboards" target="_blank">Dashboards</a> repository. The following is just a preview.
  </p>
</div>

## Power BI
![Global Renewable Energy Production](https://github.com/user-attachments/assets/bd44313e-a84b-4ef7-b9e5-94472c60f2c7)
![Amazon Video Dashboard](https://github.com/user-attachments/assets/3af68fe9-f5d2-4dd8-afe2-e5e1410f9e64)

## Tableau
![Hollywood Stories](https://github.com/user-attachments/assets/8058a127-055b-4008-a0cd-6c88b1c90fab)

## Excel
![Coffee Orders](https://github.com/TaberNater96/Data-Science/blob/main/Dashboards/Images/Coffee%20Orders.png?raw=true)

[⬆️ Back to table of contents](#top)

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

[⬆️ Back to table of contents](#top)

<div id="us-wind-power-production-with-arima" align="center">
  <h2>US Wind Power Production with ARIMA</h2>
</div>

The "US Wind Power Production with ARIMA" notebook offers a detailed analysis aimed at understanding and forecasting US wind power production. After the data is prepared and cleaned, it addresses missing values and refines the dataset for analysis. A significant portion is dedicated to exploratory data analysis (EDA), identifying **Texas** as the leading state in wind power production and examining the correlation between production outputs across states. This suggests a shared influence of climatic conditions or energy policies. The notebook's core objective is to forecast future wind power production values using the ARIMA (AutoRegressive Integrated Moving Average) model, indicating a focus on predictive analytics to discern temporal patterns and guide future renewable energy strategies. The focus here is to showcase an ability to perform a deep and detailed analysis given a relatively small and limited dataset.

![Correlation Matrix](https://github.com/TaberNater96/Data-Science/blob/main/US%20Wind%20Power%20Production%20with%20ARIMA/Images/Correlation%20Matrix.png?raw=true)

Values between 0.90 and 0.99 indicate a very high positive correlation in wind power output across states, often due to similar climatic conditions and regional energy policies, leading to synchronous growth in wind energy. Conversely, dark blue values indicate a moderately inverse relationship, where an increase in wind energy in one state corresponds to a decrease in another, likely due to geographical and weather system diversity. Understanding these correlations is vital for energy planning, as states with high correlations might not ensure energy security during dips, whereas those with inverse relationships could help stabilize the national energy grid. 

#### ARIMA Model Results For National Wind Power Generation

![ARIMA Output](https://github.com/TaberNater96/Data-Science/blob/main/US%20Wind%20Power%20Production%20with%20ARIMA/Images/ARIMA%20Output.png?raw=true)

[⬆️ Back to table of contents](#top)

<div id="nasa-meteorite-analysis" align="center">
  <h2>NASA Meteorite Analysis</h2>
</div>

![Global Mass Distribution](https://github.com/TaberNater96/Data-Science/blob/main/NASA%20Meteorites/images/Global%20Mass%20Distribution.png?raw=true)

This project delved into NASA's meteorite dataset, uncovering unique insights into global meteorite characteristics, especially ancient, larger, relict meteorites found mainly near Earth's poles. The above screenshot is just a single photo of the interactive map that allows users to see the exact coordinates, mass, and name of all known meteorites on Earth when they hover over. This program is computationally expensive, hence a screenshot. The results highlighted how certain environmental conditions foster preservation, minimizing erosion. An advanced XGBoost Regression model was developed to predict future meteorite impacts with very high accuracy, showcasing the potential of machine learning in understanding celestial phenomena. 

![ML Output](https://github.com/TaberNater96/Data-Science/blob/main/NASA%20Meteorites/images/ML%20Output.png?raw=true)

[⬆️ Back to table of contents](#top)

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

[⬆️ Back to table of contents](#top)







