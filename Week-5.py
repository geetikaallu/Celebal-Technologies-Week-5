import pandas as pd
import numpy as np

def preprocess():
    # read input files
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # separate target column
    y = train['SalePrice']
    train.drop(['SalePrice'], axis=1, inplace=True)

    # combine train + test
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # drop unnecessary columns
    combined.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)

    # fill missing values
    for col in combined.columns:
        if combined[col].dtype == 'object':
            combined[col].fillna(combined[col].mode()[0], inplace=True)
        else:
            combined[col].fillna(combined[col].median(), inplace=True)

    # add new features
    combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']
    combined['HouseAge'] = combined['YrSold'] - combined['YearBuilt']
    combined['RemodAge'] = combined['YrSold'] - combined['YearRemodAdd']
    combined['GarageAge'] = combined['YrSold'] - combined['GarageYrBlt']

    # drop unused columns
    combined.drop(columns=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'Id'], inplace=True)

    # one-hot encode data
    combined = pd.get_dummies(combined)

    # split data again
    X_train = combined.iloc[:len(y), :]
    X_test = combined.iloc[len(y):, :]

    # save processed files
    X_train.to_csv('X_train_processed.csv', index=False)
    X_test.to_csv('X_test_processed.csv', index=False)
    y.to_csv('y_train.csv', index=False)

    print("✅ Preprocessing complete. Files saved.")

# run the function
if __name__ == '__main__':
    preprocess()
