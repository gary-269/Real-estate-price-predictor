import inline as inline
import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

matplotlib.rcParams["figure.figsize"] = (20, 10)

df1 = pd.read_csv("data.csv")
# df1.head()
# print(df1.groupby('area_type')['area_type'].agg('count'))

df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
# print(df2.head())

df3 = df2.dropna()
# print(df3.shape)

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# print(df3.head())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# print(df3[~df3['total_sqft'].apply(is_float)].head(10))


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)

df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']

df5.location = df5.location.apply(lambda x: x.strip())

location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)

location_stats_less_than_10 = location_stats[location_stats <= 10]

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# For removing outliers
df6 = df5[~(df5.total_sqft / df5.bhk < 300)]


def remove_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


df7 = remove_outliers(df6)


# print(df7.shape)

def scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total square feet area")
    plt.ylabel("Price per square feet ")
    plt.title(location)
    plt.legend()
    plt.show()

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }

        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df8 = remove_bhk_outliers(df7)

# scatter_chart(df8, "Hebbal")

# matplotlib.rcParams["figure.figsize"] = (20, 10)
# plt.hist(df8.price_per_sqft, rwidth=0.8)
# plt.xlabel("Price per square feet")
# plt.ylabel("Count")
# plt.show()

df9 = df8[df8.bath < df8.bhk + 2]

df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')

dummies = pd.get_dummies(df10.location)

df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')

df12 = df11.drop('location', axis='columns')

# Creating an independent variable

x = df12.drop('price', axis='columns')

y = df12.price

