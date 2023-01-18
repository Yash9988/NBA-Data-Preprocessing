import pandas as pd
import os
import requests

from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"


# write your code here
def clean_data(path):

    df = pd.read_csv(path)

    # Preprocessing
    df.b_day, df.draft_year = pd.to_datetime(df.b_day, format="%m/%d/%y"), pd.to_datetime(df.draft_year, format="%Y")
    df.team.fillna("No Team", inplace=True)
    df.height = df.height.apply(lambda x: x.split()[-1]).astype(float)
    df.weight = df.weight.apply(lambda x: x.split()[-2]).astype(float)
    df.salary = df.salary.apply(lambda x: x[1:]).astype(float)
    df.country = df.country.apply(lambda x: 'USA' if x == 'USA' else 'Not-USA')
    df.draft_round = df.draft_round.apply(lambda x: "0" if x == 'Undrafted' else x)

    return df


def feature_data(df):

    df.version = pd.to_datetime(df.version.apply(lambda x: x[3:].replace('k', '0')))
    df['age'] = (df.version - df.b_day).apply(lambda x: x.days / 365.25).astype(int) + 1
    df['experience'] = (df.version - df.draft_year).apply(lambda x: x.days / 365).astype(int)
    df['bmi'] = df.weight / df.height ** 2

    df.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'], inplace=True)
    df.drop(columns=['full_name', 'college', 'jersey', 'draft_peak'], inplace=True)

    return df


def multicol_data(df):

    corr = df.drop(columns=['salary']).corr()
    multi_cols = []
    for col in corr.columns:
        x = corr[col]
        if not x.loc[(x < -0.5) | (x > 0.5) & (x > -1) & (x < 1)].empty:
            multi_cols.append(col)

    multi_corr = [df[['salary', i]].corr().iloc[0, 1] for i in multi_cols]
    df.drop(columns=[multi_cols[multi_corr.index(min(multi_corr))]], inplace=True)

    return df


def transform_data(df):

    scaler, encoder = StandardScaler(), OneHotEncoder()
    num_feat_df = scaler.fit_transform(df.select_dtypes('number').drop('salary', axis=1))
    cat_feat_df = encoder.fit_transform(df.select_dtypes('object'))

    return pd.concat([pd.DataFrame(num_feat_df, columns=df.select_dtypes('number').drop('salary', axis=1).columns.tolist()), pd.DataFrame(cat_feat_df.toarray(), columns=sum([i.tolist() for i in encoder.categories_], []))], axis=1), df.salary
