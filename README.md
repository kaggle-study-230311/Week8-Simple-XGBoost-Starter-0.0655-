# wee8-Simple-XGBoost-Starter-0.0655-
### 데이터 처리

1. float 64 → float 32 로 변경
    - properties_2016 data 값이 대부분 ‘0.0’형태라 dtype을 float32로 바꾸어 메모리 사용량 절약
    
    ```python
    for c, dtype in zip(prop.columns, prop.dtypes):
       if dtype == np.float64:
          prop[c] = prop[c].astype(np.float32)
    ```
    
2. data drop
    - 키값인 '`parcelid`' 타겟값인 '`logerror`’ drop
    - `transactiondate` : 이후 거래 시기와 상관없이 예측하므로 drop
    - object data중 아래 컬럼만 drop
    - `propertyzoningdesc , propertycountylandusecode`
        
        null값이 더 적은 컬럼을 삭제한 이유? object 타입이라 대체할 값을 정하기 어려워서..?
        
    
    ```markdown
    
    #object 컬럼의 null값 개수
    hashottuborspa               2916203
    propertycountylandusecode      12277
    propertyzoningdesc           1006588
    fireplaceflag                2980054
    taxdelinquencyflag           2928755
    ```
    
    ```python
    x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = df_train['logerror'].values
    ```
    
3. nan 값 처리
    - nan값을 mean값이나 다른 값으로 대체하는 것이 아닌 ‘False’값으로 대체
    
    ```python
    # dtype이 object인 컬럼값 처리  nan => False
    for c in x_train.dtypes[x_train.dtypes == object].index.values:
        x_train[c] = (x_train[c] == True)
    ```
    
4. xgboost 학습을 위한 DMatrix 형태로 변환
    
    ```python
    d_train = xgb.DMatrix(x_train, label = y_train)
    	d_valid = xgb.DMatrix(x_valid, label = y_valid)
    ```
    
5. train test split
    - train_test_split가 아닌 split할 값을 지정하고 인덱싱
    
    ```python
    # train, test data split
    split = 80000  # split를 80000으로 하면 약 88%로 분리됨
    x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
    ```
    

### 모델링 : XGBoost

1. objective : reg:squarederror
    
    기본값
    
    오차 제곱이 최소화되는 방향으로 학습
    
2. eval_metric : mae
    - **MAE(Mean Absolute Error)예측값과 Ground Truth 사이의 차이의 절대값의 평균**
        
        ![https://blog.kakaocdn.net/dn/EfDhM/btrokrAoiRB/UTcf849KTXjNMGAOwnGhC1/img.png](https://blog.kakaocdn.net/dn/EfDhM/btrokrAoiRB/UTcf849KTXjNMGAOwnGhC1/img.png)
        
    - **MSE(Mean Squared Error)예측값과 Ground Truth 사이의 차이의 제곱의 평균제곱 때문에 Anomaly Data에 매우 민감함(크기가 어마어마하게 커질 수 있음)**
        
        ![https://blog.kakaocdn.net/dn/d7rPwe/btrolVnIk8y/4oHxCl1FNVC4W1MZ150osk/img.png](https://blog.kakaocdn.net/dn/d7rPwe/btrolVnIk8y/4oHxCl1FNVC4W1MZ150osk/img.png)
        
    - **RMSE(Root Mean Squared Error)MSE에 루트를 씌운 값Anomaly에 상대적으로 덜 민감함**
        
        ![https://blog.kakaocdn.net/dn/cfDigs/btroibdgLVq/GuXv624KufG9HmnNOTdU80/img.png](https://blog.kakaocdn.net/dn/cfDigs/btroibdgLVq/GuXv624KufG9HmnNOTdU80/img.png)
        
    - **RMSLE(Root Mean Squared Logarithmic Error)예측값과 정답값에 로그를 취한 뒤 계산한 RMSE**
        
        ![https://blog.kakaocdn.net/dn/AnjNh/btroiaZHSpQ/aYnUTUoKBUPijyLTtc09JK/img.png](https://blog.kakaocdn.net/dn/AnjNh/btroiaZHSpQ/aYnUTUoKBUPijyLTtc09JK/img.png)
        

### zillow 주택가격 예측 실패 사례

- 하나금융연구소 보고서 ‘기업은 왜 데이터 분석에 성공하기 어려울까? 중 빅데이터 분석 실패사례로 소개
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c5e6f824-70ba-4a19-892d-818dc444610b/Untitled.png)
    
- 관련 기사: [https://www.seattlen.com/hot/8827](https://www.seattlen.com/hot/8827)
- *부동산시장 동향 예측력이 홈플리핑 사업의 성공을 좌우하는 중요한 요소가 된다. 구매부터 판매까지 걸리는 시간을 단축하는 것도 수익에 영향을 미친다.*
    
    *질로우는 홈플리핑 사업을 접게 된 이유에 대해 자체 설계 알고리즘의 실패를 들었다. 질로우는 자체 알고리즘(Zestimate)을 활용해 주택 구매 가격을 결정해 왔다.*
