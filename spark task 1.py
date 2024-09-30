import time
import pandas as pd
from pyspark.sql.window import Window
from pyspark.sql.functions import dayofmonth, year
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import month
from pyspark.sql.functions import to_date
from pyspark.sql.functions import weekofyear
from pyspark.sql.functions import col, lag, mean
import matplotlib.pyplot as plt


start = time.time()
findspark.init()

# create SparkSession
spark = SparkSession \
    .builder \
    .appName("CovidAnalysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .config("spark.executor.memory", "16g") \
    .getOrCreate()

# read csv
covid_df = spark.read.csv('time_series_covid19_confirmed_global.csv', header=True, inferSchema=True)

# drop the columns we don't need
covid_df = covid_df.drop("Lat", "Long")

# data processing
covid_df = covid_df.na.fill({"Province/State": "Unknown"})
covid_df = covid_df.na.fill(0)

#unpivot the data
def unpivot_data(df):
    date_columns = df.columns[2:]
    pivot_columns = [f"{col}" for col in date_columns]
    return df.unpivot(["Province/State", "Country/Region"], pivot_columns, "Date", "Cumulative_Positive_Cases")


covid_df = unpivot_data(covid_df)
covid_df = covid_df.withColumn('Date', to_date(covid_df['Date'], 'MM/dd/yy'))
covid_df = covid_df.orderBy("Country/Region","Province/State","Date")


# make the date into three columns
def date(df):
    df = df.withColumn("Month", month(df["Date"]))
    df = df.withColumn("Day", dayofmonth(df["Date"]))
    df = df.withColumn("Year", year(df["Date"]))
    df = df.withColumn("Year_Week", year(df["Date"]))
    df = df.withColumn("Week", weekofyear(df["Date"]))
    df = df.select("Province/State", "Country/Region", "Day", "Month", "Year", "Week",
                   "Year_Week", "Cumulative_Positive_Cases")
    return df


covid_df = date(covid_df)



# define the window function
windowSpec = Window.partitionBy("Country/Region", "Month", "Year").orderBy("Day")

# calculate daily increase cases
covid_df = covid_df.withColumn("Prev_Confirmed", lag("Cumulative_Positive_Cases").over(windowSpec))
covid_df = covid_df.withColumn("Daily_Increase", col("Cumulative_Positive_Cases") - col("Prev_Confirmed"))

# delete the empty column of "Daily_Increase"
covid_df = covid_df.filter(col("Daily_Increase").isNotNull())

# group and count the mean daily increase
mean_daily_increase = covid_df.groupBy("Country/Region", "Month", "Year").agg(
    mean("Daily_Increase").alias("Mean_Daily_Increase")
)

mean_daily_increase.show()

sorted_mean_daily_increase = mean_daily_increase.orderBy(["Country/Region", "Year", "Month"])


sorted_mean_daily_increase.show(truncate=False)

end = time.time()-start
print("Query 1 Processing time:", end, "seconds")

#sorted_mean_daily_increase.coalesce(1).write.csv("task 1 result.csv", header=True)

# change spark dataframe to pandas dataframe
mean_daily_increase_pandas = sorted_mean_daily_increase.toPandas()

# Assuming that 'Month' and 'Year' are integer columns, convert them to string and create 'Month-Year'
mean_daily_increase_pandas['Year-Month'] = mean_daily_increase_pandas['Year'].astype(str) + '-' + mean_daily_increase_pandas['Month'].astype(str)


top_countries = mean_daily_increase_pandas.sort_values(by='Mean_Daily_Increase', ascending=False).head(20)

top_countries['Year-Month'] = pd.Categorical(top_countries['Year-Month'], categories=sorted(top_countries['Year-Month'].unique()), ordered=True)

top_countries.sort_values(by='Year-Month', inplace=True)


plt.figure(figsize=(15, 10))

# Plot for each of the top 20 countries
for country in top_countries['Country/Region'].unique():
    # Filter the data for the current country
    country_data = top_countries[top_countries['Country/Region'] == country]
    plt.bar(country_data['Year-Month'], country_data['Mean_Daily_Increase'], label=country)

# Customization of the plot
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Most Monthly Average Confirmed COVID-19 Cases')
plt.xlabel('Year-Month')
plt.ylabel('Average Confirmed Cases')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

