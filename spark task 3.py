import time

from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
import cartopy.crs as ccrs
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.window import Window
from pyspark.sql.functions import dayofmonth, year
import pandas as pd
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import month
from pyspark.sql.functions import to_date
from pyspark.sql.functions import weekofyear
from sklearn.linear_model import LinearRegression
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

start = time.time()

findspark.init()

# creare SparkSession
spark = SparkSession \
    .builder \
    .appName("CovidAnalysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .config("spark.executor.memory", "16g") \
    .getOrCreate()

# read csv
covid_df = spark.read.csv('time_series_covid19_confirmed_global.csv', header=True, inferSchema=True)

# data processing
covid_df = covid_df.na.fill({"Province/State": "Missing"})
covid_df = covid_df.na.fill(0)

def unpivot_data(df):
    date_columns = df.columns[4:]
    pivot_columns = [f"{col}" for col in date_columns]
    return df.unpivot(["Province/State", "Country/Region", "Lat", "Long"], pivot_columns, "Date", "Cumulative_Positive_Cases")


covid_df = unpivot_data(covid_df)
covid_df = covid_df.withColumn('Date', to_date(covid_df['Date'], 'MM/dd/yy'))
covid_df = covid_df.orderBy("Country/Region","Province/State","Date")


def date(df):
    df = df.withColumn("Month", month(df["Date"]))
    df = df.withColumn("Day", dayofmonth(df["Date"]))
    df = df.withColumn("Year", year(df["Date"]))
    df = df.withColumn("Year_Week", year(df["Date"]))
    df = df.withColumn("Week", weekofyear(df["Date"]))
    df = df.select("Province/State", "Country/Region", "lat", "Long", "Day", "Month", "Year", "Week",
                   "Year_Week", "Cumulative_Positive_Cases")
    return df


covid_df = date(covid_df)

print(covid_df.columns)

# calculate the length of month of each states
windowSpec = Window.partitionBy("Country/Region", "Province/State").orderBy("Day")
covid_df = covid_df.withColumn("Prev_Confirmed", F.lag("Cumulative_Positive_Cases").over(windowSpec))
covid_df = covid_df.withColumn("Daily_Increase", F.when(F.col("Cumulative_Positive_Cases") - F.col("Prev_Confirmed") < 0, 0)
                                            .otherwise(F.col("Cumulative_Positive_Cases") - F.col("Prev_Confirmed")))
covid_df = covid_df.drop("Prev_Confirmed")


# calculate the mean lat and long
monthly_increase_df = covid_df.groupBy("Country/Region", "Year", "Month").agg(
    F.sum("Daily_Increase").alias("Monthly_Increase"),
    F.avg("Lat").alias("Lat"),
    F.avg("Long").alias("Long")
)

monthly_increase_df = monthly_increase_df.filter((monthly_increase_df["Monthly_Increase"] > 0) &
                                                 (monthly_increase_df["Monthly_Increase"].isNotNull()))

# drop the null data
monthly_increase_df = monthly_increase_df.filter(monthly_increase_df["Lat"].isNotNull() &
                                                 monthly_increase_df["Long"].isNotNull())


def calculate_trend_slope(df):
    pandas_df = df.toPandas()
    slope_results = pd.DataFrame(columns=['Country/Region', 'Slope', 'Lat', 'Long'])

    for country in df.select("Country/Region").distinct().collect():
        country_name = country['Country/Region']
        country_df = pandas_df[pandas_df['Country/Region'] == country_name]

        if not country_df.empty:
            country_df = country_df.sort_values(['Year', 'Month'])

            country_df['Month_Num'] = range(1, len(country_df) + 1)
            X = country_df[['Month_Num']]
            y = country_df['Monthly_Increase']

            model = LinearRegression()
            model.fit(X, y)

            avg_lat = country_df['Lat'].mean()
            avg_long = country_df['Long'].mean()

            slope_results = slope_results._append({
                'Country/Region': country_name,
                'Slope': model.coef_[0],
                'Lat': avg_lat,
                'Long': avg_long
            }, ignore_index=True)

    # according to slope to rank
    slope_results = slope_results.sort_values(by='Slope', ascending=False)

    return slope_results


# calculate the slope
trend_slope_df = calculate_trend_slope(monthly_increase_df)
trend_slope_df = spark.createDataFrame(trend_slope_df)
trend_slope_df.show()

# pick the top 50
top_50_df = trend_slope_df.orderBy(F.col("Slope").desc()).limit(50)
print("Columns in top_50_df:", top_50_df.columns)

vecAssembler = VectorAssembler(inputCols=["Slope"], outputCol="features")
top_50_features = vecAssembler.transform(top_50_df)
print("Columns in top_50_features:", top_50_features.columns)

# use kmean
kmeans = KMeans(featuresCol="features", k=4)
model = kmeans.fit(top_50_features)

predictions = model.transform(top_50_features)

# Rename cluster column
cluster_results = predictions.withColumnRenamed("prediction", "Cluster")

end = time.time()-start
print("Query 3 Processing time:", end, "seconds")


#cluster_results.coalesce(1).write.csv("task 3 result.csv", header=True)


cluster_results_pandas = cluster_results.select("Country/Region", "Lat", "Long", "Cluster").toPandas()
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()


colors = ['blue', 'green', 'red', 'orange']
cluster_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

# create a map to make sure clusters added the legeng
added_to_legend = {cluster: False for cluster in range(4)}

# plot countries' location
for index, row in cluster_results_pandas.iterrows():
    cluster = row['Cluster']
    if not added_to_legend[cluster]:
        ax.scatter(row['Long'], row['Lat'], color=colors[cluster], label=cluster_labels[cluster], s=50)
        added_to_legend[cluster] = True
    else:
        ax.scatter(row['Long'], row['Lat'], color=colors[cluster], s=50)

plt.legend()
plt.title('Four Clusters of 50 Places')
plt.show()


