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

print(li_env)