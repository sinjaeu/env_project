import pandas as pd

GHGs = pd.read_csv('CLM_온실가스_ANL_20240516202928.csv')
ghgs = pd.read_csv('CLM_반응가스_ANL_20240517141240.csv')

print(GHGs.columns)
print(ghgs.columns)

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