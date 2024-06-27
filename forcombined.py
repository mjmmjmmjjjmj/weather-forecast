import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

# 데이터 로드
train_data = pd.read_csv('./rainfall_train.csv')

test_data = pd.read_csv('./combined_data.csv')

# 열 이름 매핑
column_mapping = {
    'rainfall_train.fc_year': 'fc_year',
    'rainfall_train.fc_month': 'fc_month',
    'rainfall_train.fc_day': 'fc_day',
    'rainfall_train.fc_hour': 'fc_hour',
    'rainfall_train.ef_year': 'ef_year',
    'rainfall_train.ef_month': 'ef_month',
    'rainfall_train.ef_day': 'ef_day',
    'rainfall_train.ef_hour': 'ef_hour',
    'rainfall_train.dh': 'dh',
    'rainfall_train.stn4contest': 'stn4contest',
    'rainfall_train.v01': 'v01',
    'rainfall_train.v02': 'v02',
    'rainfall_train.v03': 'v03',
    'rainfall_train.v04': 'v04',
    'rainfall_train.v05': 'v05',
    'rainfall_train.v06': 'v06',
    'rainfall_train.v07': 'v07',
    'rainfall_train.v08': 'v08',
    'rainfall_train.v09': 'v09',
    'rainfall_train.vv': 'vv',
    'rainfall_train.class_interval': 'class_interval'
}

train_data = train_data.rename(columns=column_mapping)
test_data = test_data.rename(columns=column_mapping)

# 열 이름 변수
fc_year = 'fc_year'
fc_month = 'fc_month'
fc_day = 'fc_day'
fc_hour = 'fc_hour'
stn4contest = 'stn4contest'
dh = 'dh'
ef_year = 'ef_year'
ef_month = 'ef_month'
ef_day = 'ef_day'
ef_hour = 'ef_hour'
v01 = 'v01'
v02 = 'v02'
v03 = 'v03'
v04 = 'v04'
v05 = 'v05'
v06 = 'v06'
v07 = 'v07'
v08 = 'v08'
v09 = 'v09'
vv = 'vv'
class_interval = 'class_interval'

print(train_data.dtypes)
print(train_data.isnull().sum())

# 카테고리형 변환
train_data[fc_year] = train_data[fc_year].astype('category').cat.codes
train_data[ef_year] = train_data[ef_year].astype('category').cat.codes
train_data[stn4contest] = train_data[stn4contest].astype('category').cat.codes

train_data.head()

# 특징 선택
features = [
    fc_year, fc_month, fc_day, fc_hour, stn4contest, dh,
    ef_year, ef_month, ef_day, ef_hour,
    v01, v02, v03, v04, v05, v06, v07, v08, v09
]

X = train_data[features]
y = train_data[vv]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일러 저장
joblib.dump(scaler, 'scaler.pkl')

# 모델 초기화
rf = RandomForestRegressor(n_estimators=100, random_state=42)
etr = ExtraTreesRegressor(n_estimators=100, random_state=42)

models = {
    'rf': rf,
    'etr': etr
}

# 하이퍼파라미터 설정
param_distributions = {
    'etr': {
        'n_estimators': np.arange(100, 300, 100),
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None] + list(np.arange(10, 110, 10)),
        'min_samples_split': np.arange(2, 20, 2),
        'min_samples_leaf': np.arange(1, 20, 2)
    },
    'rf': {
        'n_estimators': np.arange(100, 300, 100),
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None] + list(np.arange(10, 110, 10)),
        'min_samples_split': np.arange(2, 20, 2),
        'min_samples_leaf': np.arange(1, 20, 2)
    }
}

best_estimators = {}

# 모델 최적화 및 학습
sample_size = 200000
X_train_sample = X_train.sample(n=sample_size, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

for name, model in models.items():
    random_search = RandomizedSearchCV(model, param_distributions[name], n_iter=50, cv=3,
                                       scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    random_search.fit(X_train_sample, y_train_sample)
    best_estimators[name] = random_search.best_estimator_
    print(f"{name} 최적 파라미터: {random_search.best_params_}")

    predictions = random_search.predict(X_test)
    test_score1 = r2_score(y_test, predictions)
    print(f"{name} test score r2: {test_score1}")

# 최적 모델-전체 데이터 다시 학습 및 저장
for name, estimator in best_estimators.items():
    estimator.fit(X_train, y_train)
    # 최적 모델-전체 데이터 성능 평가
    train_score = estimator.score(X_train, y_train)
    predictions = estimator.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    test_score = r2_score(y_test, predictions)

    print(f"{name} 최종 모델 train score : {train_score}")
    print(f"{name} 최종 모델 test mse : {mse}")
    print(f"{name} 최종 모델 test score r2 : {test_score}")

    # 모델 저장
    model_path = os.path.join(os.getcwd(), f'{name}_model.pkl')
    joblib.dump(estimator, model_path)
    print(f"{name} 모델이 '{model_path}' 파일로 저장되었습니다.")

import pandas as pd

# 예측 결과를 pandas 시리즈로 변환
test_predictions_series = pd.Series(predictions)

print(predictions)

print(f"{best_estimators} 최종 모델 train score : {train_score}")
print(f"{best_estimators} 최종 모델 test score : {test_score}")

# 예측 값 -> class_interval
def vv_to_class_interval(vv):
    if vv <= 0.1:
        return 0
    elif 0.1 < vv <= 0.2:
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

train_data[vv] = test_predictions_series
print(train_data[vv])

# 예측 결과에 함수 적용
train_data['rainfall_test.class_interval'] = test_predictions_series.apply(vv_to_class_interval)

# 결과 확인
print(train_data['rainfall_test.class_interval'])

# 결과 데이터 저장
train_data.to_csv('./combined_data_result.csv', index=False)
