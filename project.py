import pandas as pd

env = pd.read_csv('OBS_ASOS_ANL_20240508163403_date.csv')

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
print(sky.head())

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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df_env = pd.DataFrame(li_env, columns=['연도', '평균기온', '평균최저기온', '평균최고기온'])
# 오염물질 데이터의 DataFrame 리스트 생성
df_pollutants = [pd.DataFrame(data, columns=['연도', '구분', '배출량', '변화율']) for data in li_sky_f]

# 환경 데이터와 각 오염물질 데이터 간의 상관관계 분석
for df_pollutant in df_pollutants:
    # 오염물질 데이터프레임과 환경 데이터프레임을 합침
    df_merged = pd.merge(df_pollutant, df_env, on='연도')
    # 상관 계수 계산
    correlation_matrix = df_merged[['배출량', '변화율', '평균기온', '평균최저기온', '평균최고기온']].corr()
    print(f"오염물질: {df_pollutant['구분'][0]}")
    print(correlation_matrix)