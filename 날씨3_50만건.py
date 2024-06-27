
import pandas as pd
train_data = pd.read_csv('./rainfall_train.csv')

test_data = pd.read_csv('./Test_rainfall_test_test.csv')

fc_year = 'rainfall_train.fc_year'
fc_month = 'rainfall_train.fc_month'
fc_day = 'rainfall_train.fc_day'
fc_hour = 'rainfall_train.fc_hour'
stn4contest = 'rainfall_train.stn4contest'
dh = 'rainfall_train.dh'
ef_year = 'rainfall_train.ef_year'
ef_month = 'rainfall_train.ef_month'
ef_day = 'rainfall_train.ef_day'
ef_hour = 'rainfall_train.ef_hour'
v01 = 'rainfall_train.v01'
v02 = 'rainfall_train.v02'
v03 = 'rainfall_train.v03'
v04 = 'rainfall_train.v04'
v05 = 'rainfall_train.v05'
v06 = 'rainfall_train.v06'
v07 = 'rainfall_train.v07'
v08 = 'rainfall_train.v08'
v09 = 'rainfall_train.v09'
vv = 'rainfall_train.vv'
class_interval = 'rainfall_train.class_interval'

column_mapping = {
    'rainfall_test.fc_year': 'rainfall_train.fc_year',
    'rainfall_test.fc_month': 'rainfall_train.fc_month',
    'rainfall_test.fc_day': 'rainfall_train.fc_day',
    'rainfall_test.fc_hour': 'rainfall_train.fc_hour',
    'rainfall_test.ef_year': 'rainfall_train.ef_year',
    'rainfall_test.ef_month': 'rainfall_train.ef_month',
    'rainfall_test.ef_day': 'rainfall_train.ef_day',
    'rainfall_test.ef_hour': 'rainfall_train.ef_hour',
    'rainfall_test.dh': 'rainfall_train.dh',
    'rainfall_test.stn4contest': 'rainfall_train.stn4contest',
    'rainfall_test.v01': 'rainfall_train.v01',
    'rainfall_test.v02': 'rainfall_train.v02',
    'rainfall_test.v03': 'rainfall_train.v03',
    'rainfall_test.v04': 'rainfall_train.v04',
    'rainfall_test.v05': 'rainfall_train.v05',
    'rainfall_test.v06': 'rainfall_train.v06',
    'rainfall_test.v07': 'rainfall_train.v07',
    'rainfall_test.v08': 'rainfall_train.v08',
    'rainfall_test.v09': 'rainfall_train.v09'
}

test_data = test_data.rename(columns=column_mapping)

print(train_data.dtypes)

print(test_data.dtypes)

train_data.isnull().sum()

test_data.isnull().sum()

# # 예: 특정 월 필터링
# train_data = train_data[(train_data['rainfall_train.fc_month'].isin([5, 6, 7, 8, 9])) &
#                         (train_data['rainfall_train.ef_month'].isin([5, 6, 7, 8, 9]))]

# test_data = test_data[(test_data['rainfall_train.fc_month'].isin([5, 6, 7, 8, 9])) &
#                       (test_data['rainfall_train.ef_month'].isin([5, 6, 7, 8, 9]))]

# 카테고리형 변환
train_data['rainfall_train.fc_year'] = train_data['rainfall_train.fc_year'].astype('category').cat.codes
train_data['rainfall_train.ef_year'] = train_data['rainfall_train.ef_year'].astype('category').cat.codes
train_data['rainfall_train.stn4contest'] = train_data['rainfall_train.stn4contest'].astype('category').cat.codes

test_data['rainfall_train.fc_year'] = test_data['rainfall_train.fc_year'].astype('category').cat.codes
test_data['rainfall_train.ef_year'] = test_data['rainfall_train.ef_year'].astype('category').cat.codes
test_data['rainfall_train.stn4contest'] = test_data['rainfall_train.stn4contest'].astype('category').cat.codes

# if 'class interval' in train_data.columns:
#     train_data['class interval'] = train_data['class interval'].astype(int)

# if 'class interval' in test_data.columns:
#     test_data['class interval'] = test_data['class interval'].astype(int)

train_data.head()

test_data.head()

features = [
    fc_year, fc_month, fc_day, fc_hour, stn4contest, dh,
       ef_year, ef_month, ef_day, ef_hour,
       v01, v02, v03, v04, v05, v06, v07, v08, v09]

X = train_data[features]
y = train_data[vv]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# 모델
rf = RandomForestRegressor(n_estimators=100, random_state=42)
etr = ExtraTreesRegressor(n_estimators=100, random_state=42)

models = {
    'rf' : RandomForestRegressor(n_estimators=100, random_state=42),
    'etr' : ExtraTreesRegressor(n_estimators=100, random_state=42)}

# # 학습
# rf.fit(X_train, y_train)
# etr.fit(X_train, y_train)

# # 예측값
# rf_pred = rf.predict(X_test)
# gbr_pred = gbr.predict(X_test)
# etr_pred = etr.predict(X_test)
# xgbr_pred = xgbr.predict(X_test)
# lgbmr_pred = lgbmr.predict(X_test)

# # 예측값
# rf_pred = rf.predict(X_test)
# etr_pred = etr.predict(X_test)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

sample_size = 200000
X_train_sample = X_train.sample(n=sample_size, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

# param_distributions = {
#     'n_estimators': randint(10, 200),
#     'max_features': randint(1, X_train_sample.shape[1]),
#     'max_depth': randint(1, 20),
#     'min_samples_split': randint(2, 20),
#     'min_samples_leaf': randint(1, 20)
#     }

# from sklearn.metrics import mean_squared_error, r2_score

# models = {'etr':etr, 'rf':rf}
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     train_score = model.score(X_train, y_train)
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     test_score = r2_score(y_test, predictions)
#     print(f'{name} Train Score: {train_score}')
#     print(f'{name} Test MSE: {mse}')
#     print(f'{name} Test Score: {test_score}')

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint

# sample_size = 100000
# X_train_sample = X_train_split.sample(n=sample_size, random_state=42)
# y_train_sample = y_train_split.loc[X_train_sample.index]

# param_distributions = {
#     'n_estimators': randint(10, 200),
#     'max_features': randint(1, X_train_sample.shape[1]),
#     'max_depth': randint(1, 20),
#     'min_samples_split': randint(2, 20),
#     'min_samples_leaf': randint(1, 20)
# }

# from sklearn.metrics import mean_squared_error, r2_score

# best_estimators = {}
# for name, model in models.items():
#     random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=3,
#                                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
#     random_search.fit(X_train_sample, y_train_sample)
#     best_estimators[name] = random_search.best_estimator_
#     print(f"{name} 최적 파라미터: {random_search.best_params_}")

#     predictions = random_search.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     print(f"{name} 검증 MSE: {mse}")

#     mse = mean_squared_error(y_test, predictions)
#     test_score1 = r2_score(y_test, predictions)

#     print(f'{name} Train Score: {train_score}')
#     print(f'{name} Test MSE: {mse}')
#     print(f'{name} Test Score r2: {test_score1}')

# best_estimators = {}
# for model_name, model in models.items():
#     grid_search = TQDMSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
#     grid_search.fit(X_train_split, y_train_split)
#     best_estimators[model_name] = grid_search.best_estimator_
#     print(f"{model_name} best params: {grid_search.best_params_}")
#     y_pred = grid_search.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"{model_name} validation MSE: {mse}")

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

models = {
    'etr': etr,
    'rf': rf
}

param_distributions = {
    'etr': {
        'n_estimators': np.arange(100, 500, 100),
        'max_features': ['sqrt', 'log2'],  # 'auto' 대신 'sqrt' 또는 'log2' 사용
        'max_depth': [None] + list(np.arange(10, 110, 10)),
        'min_samples_split': np.arange(2, 20, 2),
        'min_samples_leaf': np.arange(1, 20, 2)
    },
    'rf': {
        'n_estimators': np.arange(100, 500, 100),
        'max_features': ['sqrt', 'log2'],  # 'auto' 대신 'sqrt' 또는 'log2' 사용
        'max_depth': [None] + list(np.arange(10, 110, 10)),
        'min_samples_split': np.arange(2, 20, 2),
        'min_samples_leaf': np.arange(1, 20, 2)
    }
}

best_estimators = {}

for name, model in models.items():
    random_search = RandomizedSearchCV(model, param_distributions[name], n_iter=50, cv=3,
                                       scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    random_search.fit(X_train_sample, y_train_sample)
    best_estimators[name] = random_search.best_estimator_
    print(f"{name} 최적 파라미터: {random_search.best_params_}")

    predictions = random_search.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"{name} 검증 mse : {mse}")

    test_score1 = r2_score(y_test, predictions)
    print(f"{name} test score r2: {test_score1}")

predictions_ensemble = np.zeros_like(y_test, dtype=float)

# 최적 모델-전체 데이터 다시 학습
for name, best_estimators in best_estimators.items():
    best_estimators.fit(X_train, y_train)
    # 최적 모델-전체 데이터 성능 평가
    train_score = best_estimators.score(X_train, y_train)
    predictions = best_estimators.predict(X_test)

    predictions_ensemble += predictions / len(best_estimators)

    mse = mean_squared_error(y_test, predictions)
    test_score = r2_score(y_test, predictions)

    print(f"{name} 최종 모델 train score : {train_score}")
    print(f"{name} 최종 모델 test mse : {mse}")
    print(f"{name} 최종 모델 test score r2 : {test_score}")


# 앙상블 성능 평가
mse_ensemble = mean_squared_error(y_test, predictions_ensemble)
test_score_ensemble = r2_score(y_test, predictions_ensemble)

print(f"앙상블 모델 Test MSE: {mse_ensemble}")
print(f"앙상블 모델 Test Score r2: {test_score_ensemble}")

import pandas as pd

# 예측 결과를 pandas 시리즈로 변환
test_predictions_series = pd.Series(predictions)

# 예측 값 -> class_interval
def vv_to_class_interval(vv):
    if vv <= 0.1:
        return 0
    elif 0.1< vv <= 0.2:
        return 1
    elif 0.2 < vv <= 0.5:
        return 2
    elif 0.5 < vv <= 1.0:
        return 3
    elif 1.0 < vv <= 2.0:
        return 4
    elif 2.0 < vv <= 5.0:
        return 5
    elif 5.0 < vv <= 10.0:
        return 6
    elif 10.0 < vv <= 20.0:
        return 7
    elif 20.0 < vv <= 30.0:
        return 8
    else:
        return 9

# 예측 결과에 함수 적용
test_data['rainfall_test.class_interval'] = test_predictions_series.apply(vv_to_class_interval)

# 결과 확인
print(test_data['rainfall_test.class_interval'])

# 원래 컬럼명을 키로, 변경된 컬럼명을 값으로 하는 역 매핑 생성
reverse_column_mapping = {v: k for k, v in column_mapping.items()}

# 컬럼명을 원래대로 되돌리기
test_data = test_data.rename(columns=reverse_column_mapping)

test_data.to_csv('./240484.csv', index=False)