import pandas as pd

env = pd.read_csv('OBS_ASOS_ANL_20240508163403.csv')
print(env.head())

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

cnt, env_nan = delNaN(env, ['평균기온(°C)', '평균최저기온(°C)', '평균최고기온(°C)'], start=1)

li_env = []

for i in range(len(env_nan['일시'])):
    for j in range(len(env_nan['일시'])):
        if i == 0:
            li_env.append([env_nan['일시'][j], env_nan['평균기온(°C)'][j], env_nan['평균최저기온(°C)'][j], env_nan['평균최고기온(°C)'][j]])
            continue
        else:
            li_env[j][1] += env_nan['평균기온(°C)'][j]
            li_env[j][2] += env_nan['평균최저기온(°C)'][j]
            li_env[j][3] += env_nan['평균최고기온(°C)'][j]

for i in range(len(li_env)):
    for j in range(1, len(li_env)):
        li_env[i][j] /= cnt[j-1]
        li_env[i][j] = round(li_env[i][j], 2)

print(li_env)