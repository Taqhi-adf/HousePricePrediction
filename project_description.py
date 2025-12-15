 project summary** and a **column-wise description** that you can directly use in a **project report, README.md, or presentation**.


# 📌 Project Summary: Synthetic House Price Dataset

This project focuses on the creation of a **synthetic real estate dataset** designed to simulate housing market data for analytical and machine learning purposes. The dataset contains **1,000 records** representing residential properties across multiple cities in the United States.

The data is generated using **random sampling techniques** to ensure variability and realism in property characteristics such as location, house type, size, number of rooms, and pricing. Listing dates span several years, allowing for potential **time-based analysis**.

This dataset is suitable for:

* Exploratory Data Analysis (EDA)
* House price prediction models
* Regression and feature-engineering practice
* Visualization and dashboard development
* Academic and learning-oriented machine learning projects

Since the dataset is synthetically generated, it contains **no real personal or financial data**, making it safe for experimentation and demonstration.

# 📊 Dataset Overview

* **Total Records:** 1,000
* **Data Type:** Structured tabular data
* **Domain:** Real Estate / Housing Market
* **Cities Covered:** New York, Los Angeles, Chicago, Texas, San Francisco

# 🧾 Column Descriptions

| Column Name      | Data Type   | Description                                                                                    |
| ---------------- | ----------- | ---------------------------------------------------------------------------------------------- |
| **City**         | Categorical | Name of the city where the property is located. Randomly selected from five major U.S. cities. |
| **House_type**   | Categorical | Type of residential property such as Apartment, Condo, Villa, or Townhouse.                    |
| **Built_year**   | Integer     | Year in which the house was constructed, ranging from 1950 to 2022.                            |
| **Price**        | Integer     | Listing price of the house in USD. Values range between 50,000 and 100,000.                    |
| **Area_sqft**    | Integer     | Total built-up area of the house measured in square feet.                                      |
| **Bedrooms**     | Integer     | Number of bedrooms available in the house (1 to 5).                                            |
| **Bathrooms**    | Integer     | Number of bathrooms in the house (1 to 4).                                                     |
| **Garage**       | Categorical | Indicates whether the house has a garage (`Yes` or `No`).                                      |
| **Listing_Date** | Date        | Date when the property was listed for sale. Dates span multiple years.                         |
| **Agent_Name**   | Categorical | Name of the real estate agent responsible for the listing.                                     |

## 🎯 Potential Use Cases

* House price prediction using regression models
* Feature importance analysis
* Time-series analysis on listing trends
* City-wise and property-type price comparison
* Building Streamlit dashboards for real estate insights

# the end