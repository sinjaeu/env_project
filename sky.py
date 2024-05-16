import pandas as pd

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

for i in range(len(li_sky)):
    print(*li_sky[i])