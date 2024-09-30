import time
from pyspark.sql.window import Window
from pyspark.sql.functions import dayofmonth, year
import pandas as pd
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import month,  split
from pyspark.sql.functions import mean, sum, stddev, max, min
from pyspark.sql.functions import to_date
from pyspark.sql.functions import weekofyear
from pyspark.sql.functions import lag, col
from pyspark.sql.functions import dense_rank
from sklearn.linear_model import LinearRegression
from pyspark.sql.functions import when
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
covid_df = covid_df.na.fill({"Province/State": "Missing"})
covid_df = covid_df.na.fill(0)

#read the csv that can reflect the continents of different states
continent_df = pd.read_csv('country-continent-codes.csv', skiprows=1)

continent_map = {
    'Asia': 'Asia',
    'Europe': 'Europe',
    'Africa': 'Africa',
    'Oceania': 'Oceania',
    'North America': 'America',
    'South America': 'America',
    'Antarctica': 'Antarctica'
}

# make the dataset's country consistent
continent_df['country'] = continent_df['country'].str.split(',').str[0]

# update the continent by map
continent_df['continent'] = continent_df['continent'].map(continent_map)

country_to_continent = {row['country']: row['continent'] for index, row in continent_df.iterrows()}


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

#calculate the daily increase
def calculate_daily_increase(df):
    windowSpec = Window.partitionBy("Country/Region", "Province/State").orderBy("Year", "Month", "Day")
    df = df.withColumn("Prev_Confirmed", lag("Cumulative_Positive_Cases").over(windowSpec))
    df = df.withColumn("Daily_Increase", when(col("Cumulative_Positive_Cases") - col("Prev_Confirmed") < 0, 0)
                                            .otherwise(col("Cumulative_Positive_Cases") - col("Prev_Confirmed")))
    df = df.drop("Prev_Confirmed")
    return df.na.fill({"Daily_Increase": 0})


covid_df_daily_increase = calculate_daily_increase(covid_df)

def calculate_trend_slope(df):
    pandas_df = df.toPandas()

    slope_results = pd.DataFrame(columns=['Country/Region', 'Slope'])

    for country in pandas_df['Country/Region'].unique():
        country_df = pandas_df[pandas_df['Country/Region'] == country]

        # sort the day of data
        country_df = country_df.sort_values(by='Day')

        X = country_df.index.values.reshape(-1, 1)
        y = country_df['Daily_Increase'].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        slope_results = slope_results._append({
            'Country/Region': country,
            'Slope': model.coef_[0][0]
        }, ignore_index=True)

    slope_results = slope_results.sort_values(by='Slope', ascending=False)

    return slope_results


# calculate daily increase
daily_slope_df = calculate_trend_slope(covid_df_daily_increase)

spark_df = spark.createDataFrame(daily_slope_df)


top_100_df = spark_df.orderBy(col("slope").desc()).limit(100)


# use window function
window = Window.orderBy(col("slope").desc())
top_100_df = top_100_df.withColumn("Rank", dense_rank().over(window))

top_100_df = top_100_df.dropDuplicates(["Country/Region"])


# make sure no repeat in the dataframe
continent_spark_df = spark.createDataFrame(continent_df)
continent_spark_df = continent_spark_df.withColumn('country', split(col('country'), ',').getItem(0)).dropDuplicates(["country"])
country_to_continent_df = continent_spark_df.selectExpr('country as Country', 'continent as Continent')

year_week_df = covid_df.select("Country/Region", "Year", "Week").distinct()


top_100_with_year_week = top_100_df.join(year_week_df, ["Country/Region"], "left_outer")

# To get continents data to reflect on top_100_with_continent
top_100_with_continent = top_100_with_year_week.join(country_to_continent_df, top_100_with_year_week["Country/Region"] == country_to_continent_df["Country"], "left_outer")


top_100_with_continent = top_100_with_continent.select("Rank", "Country/Region", "Year", "Week", "Slope", "Continent")


# Include year and week data
covid_df_daily_increase = covid_df_daily_increase.withColumnRenamed("Country/Region", "Country_Region_Original") \
                                                 .withColumnRenamed("Year", "Year_Original") \
                                                 .withColumnRenamed("Week", "Week_Original")

# Join data
top_100_daily_increase = top_100_with_continent.join(covid_df_daily_increase,
                                                     (top_100_with_continent["Country/Region"] == covid_df_daily_increase["Country_Region_Original"]) &
                                                     (top_100_with_continent["Year"] == covid_df_daily_increase["Year_Original"]) &
                                                     (top_100_with_continent["Week"] == covid_df_daily_increase["Week_Original"]),
                                                     "inner")


top_100_daily_increase = top_100_daily_increase.select("Rank", "Country/Region", "Year", "Week", "Slope", "Continent", "Daily_Increase")
# Sort the rank
window = Window.orderBy(col("slope").desc())
top_100_with_continent = top_100_with_continent.withColumn("Rank", dense_rank().over(window))

# use additional map to make sure there are no null in top 100
additional_mapping = {
    "Taiwan*": "Asia",
    "Korea, South": "Asia",
    "US": "America",
    "Vietnam": "Asia",
    "Czechia": "Europe",
    "Slovakia": "Europe",
    "Brunei": "Asia",
    "United Kingdom": "Europe",
    "Kosovo": "Europe",
    "Russia": "Europe",
    "Laos": "Asia"
}

for country, continent in additional_mapping.items():
    new_row = pd.DataFrame({"country": [country], "continent": [continent]})
    continent_df = continent_df._append(new_row, ignore_index=True)


continent_df = continent_df.drop_duplicates(subset=['country'])

continent_spark_df = spark.createDataFrame(continent_df)
country_to_continent_df = continent_spark_df.selectExpr('country as Country', 'continent as Continent')


# Rank the slope
top_100_df = spark_df.orderBy(col("slope").desc())
window = Window.orderBy(col("slope").desc())
top_100_df = top_100_df.withColumn("Rank", dense_rank().over(window))

top_100_df = top_100_df.filter(col("Rank") <= 100).dropDuplicates(["Country/Region"])



# Calculate each continent values
weekly_stats_df = top_100_daily_increase.groupBy("Continent", "Year", "Week").agg(
    mean("Daily_Increase").alias("Mean_Weekly_Increase"),
    stddev("Daily_Increase").alias("Std_Weekly_Increase"),
    max("Daily_Increase").alias("Max_Weekly_Increase"),
    min("Daily_Increase").alias("Min_Weekly_Increase")
)



end = time.time()-start
print("Query 2 Processing time:", end, "seconds")


#weekly_stats_df.coalesce(1).write.csv("task 2 result.csv", header=True)

# use top 20 to visualasation
top_20_df = top_100_df.limit(20)
top_20_pandas_df = top_20_df.toPandas()


continent_colors = {
    'Asia': 'orange',
    'Europe': 'blue',
    'Africa': 'black',
    'Oceania': 'green',
    'America': 'red',
    'Antarctica': 'gray'
}

plt.figure(figsize=(10, 6))
plt.barh(top_20_pandas_df['Country/Region'], top_20_pandas_df['Slope'])
plt.xlabel('Slope')
plt.ylabel('Country/Region')
plt.title('Top 20 Countries by COVID-19 Trend Slope')
plt.gca().invert_yaxis()
plt.show()


continent_stats_pandas_df = weekly_stats_df.toPandas()

continents = continent_stats_pandas_df['Continent'].unique()
years = continent_stats_pandas_df['Year'].unique()


for year in years:
    plt.figure(figsize=(10, 6))
    for continent in continents:
        subset = continent_stats_pandas_df[(continent_stats_pandas_df['Continent'] == continent) & (continent_stats_pandas_df['Year'] == year)].sort_values('Week')

        plt.plot(subset['Week'], subset['Mean_Weekly_Increase'], label=continent)

    plt.title(f'Mean Weekly Increase by Continent in {year}')
    plt.xlabel('Week')
    plt.ylabel('Mean Weekly Increase')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Standard Deviation of Weekly Increase
for year in years:
    plt.figure(figsize=(10, 6))
    for continent in continents:
        subset = continent_stats_pandas_df[(continent_stats_pandas_df['Continent'] == continent) & (continent_stats_pandas_df['Year'] == year)].sort_values('Week')
        plt.plot(subset['Week'], subset['Std_Weekly_Increase'], label=continent)
    plt.title(f'Standard Deviation of Weekly Increase in {year}')
    plt.xlabel('Week')
    plt.ylabel('Standard Deviation')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Max Weekly Increase
for year in years:
    plt.figure(figsize=(10, 6))
    for continent in continents:
        subset = continent_stats_pandas_df[(continent_stats_pandas_df['Continent'] == continent) & (continent_stats_pandas_df['Year'] == year)].sort_values('Week')
        plt.plot(subset['Week'], subset['Max_Weekly_Increase'], label=continent)
    plt.title(f'Max Weekly Increase in {year}')
    plt.xlabel('Week')
    plt.ylabel('Max Increase')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Min Weekly Increase
for year in years:
    plt.figure(figsize=(10, 6))
    for continent in continents:
        subset = continent_stats_pandas_df[(continent_stats_pandas_df['Continent'] == continent) & (continent_stats_pandas_df['Year'] == year)].sort_values('Week')
        plt.plot(subset['Week'], subset['Min_Weekly_Increase'], label=continent)
    plt.title(f'Min Weekly Increase in {year}')
    plt.xlabel('Week')
    plt.ylabel('Min Increase')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()



