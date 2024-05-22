import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt


env = pd.read_csv('OBS_ASOS_ANL_20240508163403_date.csv')
# 소수점 이하 자릿수 설정
np.set_printoptions(precision=3)

# 지수 표기법 사용 안함
np.set_printoptions(suppress=True)

def delNaN(data, *idx, start=0):
    li_idx = []
    for i in idx:
        li_idx.append(i)

    cnt = [len(data) for _ in range(len(li_idx[0]))]
    
    if len(li_idx[0]) == 1:
        col = li_idx[0][0]
        for i in range(start, len(data)):
            if pd.isnull(data.at[i, col]):
                data.at[i, col] = 0
                cnt[0] -= 1
    else:
        for col in li_idx[0]:
            for i in range(start, len(data)):
                if pd.isnull(data.at[i, col]):
                    data.at[i, col] = 0
                    cnt[li_idx[0].index(col)] -= 1

    return cnt, data

li_cnt, env_nan = delNaN(env, ['평균기온(°C)', '평균최저기온(°C)', '평균최고기온(°C)'], start=1)

li_env = []
date = 0
cnt = -1
li_cnt = []
li_cnt_n = 1
for i in range(len(env_nan)):
    if env_nan['일시'][i] != date:
        li_env.append([env_nan['일시'][i], env_nan['평균기온(°C)'][i], env_nan['평균최저기온(°C)'][i], env_nan['평균최고기온(°C)'][i]])
        date = env_nan['일시'][i]
        cnt+=1
        li_cnt.append(li_cnt_n)
        li_cnt_n = 1
    else:
        li_env[cnt][1] += env_nan['평균기온(°C)'][i]
        li_env[cnt][2] += env_nan['평균최저기온(°C)'][i]
        li_env[cnt][3] += env_nan['평균최고기온(°C)'][i]
        li_cnt_n += 1
li_cnt.append(li_cnt_n)
del li_cnt[0]

for i in range(len(li_env)):
    for j in range(1, len(li_env[i])):
        li_env[i][j] = round(li_env[i][j]/li_cnt[i], 2)

sky = pd.read_csv('전국_대기오염물질_배출량_20240513180909.csv')

li_sky = []
for i in range(len(sky)):
    li_sky.append([sky['연도'][i], sky['구분'][i], sky['배출량'][i]])

a = ''
for i in range(len(sky)):
    if a != li_sky[i][1]:
        a = li_sky[i][1]
        li_sky[i].append(0)
        continue
    else:
        li_sky[i].append(round(((li_sky[i][2]/li_sky[i-1][2])*100)-100, 2))

li_sky_1 = []
li_sky_2 = []
li_sky_3 = []
li_sky_4 = []
li_sky_5 = []
li_sky_6 = []
li_sky_7 = []
li_sky_8 = []
li_sky_9 = []

cnt = 0
prev_pollutant = ''
for data in li_sky:
    # 오염물질의 이름이 바뀌면 새로운 리스트에 추가
    if data[1] != prev_pollutant:
        prev_pollutant = data[1]
        cnt += 1
    # cnt 값을 사용하여 적절한 변수에 추가
    if cnt == 1:
        li_sky_1.append(data)
    elif cnt == 2:
        li_sky_2.append(data)
    elif cnt == 3:
        li_sky_3.append(data)
    elif cnt == 4:
        li_sky_4.append(data)
    elif cnt == 5:
        li_sky_5.append(data)
    elif cnt == 6:
        li_sky_6.append(data)
    elif cnt == 7:
        li_sky_7.append(data)
    elif cnt == 8:
        li_sky_8.append(data)
    elif cnt == 9:
        li_sky_9.append(data)

li_sky_f = [li_sky_1, li_sky_2, li_sky_3, li_sky_4, li_sky_5, li_sky_6, li_sky_7, li_sky_8, li_sky_9]

GHGs = pd.read_csv('CLM_온실가스_ANL_20240516202928.csv')
ghgs = pd.read_csv('CLM_반응가스_ANL_20240517141240.csv')

GHGs_1 = []
GHGs_2 = []
GHGs_3 = []
GHGs_4 = []
GHGs_5 = []
GHGs_6 = []
GHGs_7 = []
for i in range(len(GHGs)):
    GHGs_1.append([GHGs['일시'][i], '평균 이산화탄소(CO2) 배경대기농도(ppm)', GHGs['평균 이산화탄소(CO2) 배경대기농도(ppm)'][i]])
    GHGs_2.append([GHGs['일시'][i], '평균 메탄(CH4) 배경대기농도(ppb)', GHGs['평균 메탄(CH4) 배경대기농도(ppb)'][i]])
    GHGs_3.append([GHGs['일시'][i], '평균 아산화질소(N2O) 배경대기농도(ppb)', GHGs['평균 아산화질소(N2O) 배경대기농도(ppb)'][i]])
    GHGs_4.append([GHGs['일시'][i], '평균 염화불화탄소11(CFC11) 배경대기농도(ppt)', GHGs['평균 염화불화탄소11(CFC11) 배경대기농도(ppt)'][i]])
    GHGs_5.append([GHGs['일시'][i], '평균 염화불화탄소12(CFC12) 배경대기농도(ppt)', GHGs['평균 염화불화탄소12(CFC12) 배경대기농도(ppt)'][i]])
    GHGs_6.append([GHGs['일시'][i], '평균 염화불화탄소113(CFC113) 배경대기농도(ppt)', GHGs['평균 염화불화탄소113(CFC113) 배경대기농도(ppt)'][i]])
    GHGs_7.append([GHGs['일시'][i], '평균 육불화황(SF6) 배경대기농도(ppt)', GHGs['평균 육불화황(SF6) 배경대기농도(ppt)'][i]])

li_GHGs = [GHGs_1, GHGs_2, GHGs_3, GHGs_4, GHGs_5, GHGs_6, GHGs_7]

ghgs_1 = []
ghgs_2 = []
ghgs_3 = []
ghgs_4 = []
for i in range(len(ghgs)):
    ghgs_1.append([ghgs['일시'][i], '평균 일산화탄소(CO) 농도(ppb)', ghgs['평균 일산화탄소(CO) 농도(ppb)'][i]])
    ghgs_2.append([ghgs['일시'][i], '평균 지표오존(O₃) 농도(ppb)', ghgs['평균 지표오존(O₃) 농도(ppb)'][i]])
    ghgs_3.append([ghgs['일시'][i], '평균 이산화황(SO₂) 농도(ppb)', ghgs['평균 이산화황(SO₂) 농도(ppb)'][i]])
    ghgs_4.append([ghgs['일시'][i], '평균 질소산 화물(NO x) 농도(ppb)', ghgs['평균 질소산화물(NOｘ) 농도(ppb)'][i]])

li_ghgs = [ghgs_1, ghgs_2, ghgs_3, ghgs_4]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df_env = pd.DataFrame(li_env, columns=['연도', '평균기온', '평균최저기온', '평균최고기온'])
# 오염물질 데이터의 DataFrame 리스트 생성
df_pollutants = [pd.DataFrame(data, columns=['연도', '구분', '배출량', '변화율']) for data in li_sky_f]

df_GHGs = [pd.DataFrame(data, columns=['연도', '구분', '배출량']) for data in li_GHGs]

df_ghgs = [pd.DataFrame(data, columns=['연도', '구분', '배출량']) for data in li_ghgs]

# 환경 데이터와 각 오염물질 데이터 간의 상관관계 분석
correlation_matrix_pollutant = []
for df_pollutant in df_pollutants:
    # 오염물질 데이터프레임과 환경 데이터프레임을 합침
    df_merged = pd.merge(df_pollutant, df_env, on='연도')
    # 상관 계수 계산
    correlation_matrix_pollutant .append(df_merged[['배출량', '변화율', '평균기온', '평균최저기온', '평균최고기온']].corr())

correlation_matrix_GHGs = []
for df_GHG in df_GHGs:
    # 오염물질 데이터프레임과 환경 데이터프레임을 합침
    df_merged = pd.merge(df_GHG, df_env, on='연도')
    # 상관 계수 계산
    correlation_matrix_GHGs.append(df_merged[['배출량', '평균기온', '평균최저기온', '평균최고기온']].corr())

correlation_matrix_ghgs = []
for df_ghg in df_ghgs:
    # 오염물질 데이터프레임과 환경 데이터프레임을 합침
    df_merged = pd.merge(df_ghg, df_env, on='연도')
    # 상관 계수 계산
    correlation_matrix_ghgs.append(df_merged[['배출량', '평균기온', '평균최저기온', '평균최고기온']].corr())

# for df in df_GHGs:
#     X = df
#     y = df_env[94:118:]
#     print(X, y, sep='\n')
#     print(len(X), len(y), sep='\n')
#     print('---------------------------------------')

# 결과를 저장할 리스트
models = []
scores = []

for df in df_GHGs:
    # '연도', '구분', '배출량' 열을 제거하고 입력 변수 X 준비
    X = df.drop(columns=['구분'])
    
    # 타겟 변수 y 준비 (예: '기온' 열)
    y = df_env[94:118:]
    
    # 학습 데이터와 테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=892) #0.2 892
    
    # 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 모델 성능 평가
    score = model.score(X_test, y_test)
    
    # 모델과 성능 점수를 리스트에 저장
    models.append(model)
    scores.append(score)

# 각 모델의 성능 점수 출력
for i, score in enumerate(scores):
    print(f"Dataset {i+1} model score: {score}")

# 각 모델을 파일로 저장
for i, model in enumerate(models):
    joblib.dump(model, f"model_{i}.pkl")

# 각 온실가스의 이름과 초기값을 입력 받음
ghg_names = ['이산화탄소', '메탄', '아산화질소', '염화불화탄소11', '염화불화탄소12', '염화불화탄소113', '육불화황']
ghg_values = []

for ghg in ghg_names:
    value = float(input(f"{ghg}의 초기값을 입력하세요: "))
    ghg_values.append(value)

# 입력 받은 각 온실가스 초기값을 20년 동안 유지하는 데이터 생성
GHGs_value = np.zeros((20, len(ghg_values) + 1))

# 연도 정보 추가
for i in range(20):
    GHGs_value[i][-1] = i + 2023  # 예측 시작 연도인 2025년부터 시작

# 각 연도별로 입력된 온실가스 초기값을 추가
for i in range(20):
    GHGs_value[i][:len(ghg_values)] = ghg_values

# 결과를 저장할 리스트
predict_data = []

# 각 모델을 사용하여 예측
for cnt in range(len(ghg_names)):
    # 모델 파일의 경로
    model_path = f"model_{cnt}.pkl"

    # 모델 로드
    model = joblib.load(model_path)
    predictions = model.predict([[GHGs_value[i][-1], GHGs_value[i][cnt]] for i in range(len(GHGs_value))])  # 연도와 온실가스 값의 쌍으로 학습한 모델이므로 마지막 열 제외

    predictions_rounded = np.round(predictions, 3)

    # 예측 결과 출력
    print(f"예측 결과 {cnt + 1}:", predictions_rounded)
    predict_data.append(predictions_rounded)
    print('----------------------------------------')
# print(len(predict_data[1]))
predict_avg = []
for i in range(len(predict_data[0])):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for j in range(len(predict_data)):
        sum1 += predict_data[j][i][1] * correlation_matrix_GHGs[j]['배출량']['평균기온']
        sum2 += predict_data[j][i][2] * correlation_matrix_GHGs[j]['배출량']['평균최저기온']
        sum3 += predict_data[j][i][3] * correlation_matrix_GHGs[j]['배출량']['평균최고기온']
    predict_avg.append([2023 + i, round(sum1/len(predict_data[j]), 3), round(sum2/len(predict_data[j]), 3), round(sum3/len(predict_data[j]), 3)])

for i in predict_avg:
    print(i)

# for i in correlation_matrix_GHGs:
#     print(i['배출량']['평균기온'])
