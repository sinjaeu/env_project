# 환경 데이터 분석 및 예측 프로젝트

## 프로젝트 개요
이 프로젝트는 기상 데이터와 대기 오염물질 데이터를 분석하고, 머신러닝을 활용하여 미래 환경 변화를 예측하는 시스템입니다.

## 주요 기능
1. 기상 데이터 분석
   - 평균 기온, 최저 기온, 최고 기온 데이터 처리
   - 결측치 처리 및 데이터 정제

2. 대기 오염물질 분석
   - 온실가스 (CO2, CH4, N2O 등) 농도 분석
   - 반응가스 (CO, O3, SO2, NOx) 농도 분석
   - 오염물질 배출량 변화율 계산

3. 머신러닝 기반 예측
   - 선형 회귀 모델을 사용한 환경 변화 예측
   - 교차 검증을 통한 모델 성능 평가
   - 20년 단위 미래 예측

## 사용된 기술
- Python
- pandas: 데이터 처리 및 분석
- numpy: 수치 계산
- scikit-learn: 머신러닝 모델 구현
- matplotlib: 데이터 시각화
- xgboost: 고급 머신러닝 알고리즘

## 프로젝트 구조
```
├── project.py          # 메인 프로젝트 파일
├── env.py             # 환경 데이터 처리 모듈
├── GHGs.py            # 온실가스 데이터 처리 모듈
├── sky.py             # 대기 오염물질 처리 모듈
├── model_*.pkl        # 학습된 머신러닝 모델 파일들
└── 데이터 파일들
    ├── OBS_ASOS_ANL_*.csv           # 기상 관측 데이터
    ├── CLM_온실가스_ANL_*.csv        # 온실가스 데이터
    ├── CLM_반응가스_ANL_*.csv        # 반응가스 데이터
    └── 전국_대기오염물질_배출량_*.csv  # 대기 오염물질 배출량 데이터
```

## 설치 및 실행 방법
1. 필요한 패키지 설치:
```bash
pip install pandas numpy scikit-learn matplotlib xgboost joblib
```

2. 프로젝트 실행:
```bash
python project.py
```

## 데이터 설명
1. 기상 데이터
   - 평균 기온, 최저 기온, 최고 기온 정보
   - 일별 관측 데이터

2. 온실가스 데이터
   - 이산화탄소(CO2)
   - 메탄(CH4)
   - 아산화질소(N2O)
   - 염화불화탄소(CFC11, CFC12, CFC113)
   - 육불화황(SF6)

3. 반응가스 데이터
   - 일산화탄소(CO)
   - 지표오존(O3)
   - 이산화황(SO2)
   - 질소산화물(NOx)

## 주의사항
- 데이터 파일은 반드시 프로젝트 루트 디렉토리에 위치해야 합니다.
- 예측 모델 사용 시 초기값 입력이 필요합니다.
- 대용량 데이터 처리 시 메모리 사용량에 주의하세요.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다. 
