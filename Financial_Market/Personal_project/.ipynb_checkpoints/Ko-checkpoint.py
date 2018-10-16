import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



def Preprocessing(raw, column):
    
    '''
    지표 값이 담겨있는 데이터프레임이 대한 전처리를 수행하는 함수
    '''
    
    CD_list = list(raw.iloc[0,1:])
    name_list = list(raw.iloc[1,1:])
    month_list = list(raw.iloc[6:,0].apply(lambda x: x.replace('-','')))
    
    if column == '기업명':
        df = pd.DataFrame(raw.iloc[6:,1:].values,
                         index = month_list,
                         columns = name_list)
    elif column == '심볼' or column =='Symbol':
        df = pd.DataFrame(raw.iloc[6:,1:].values,
                         index = month_list,
                         columns = CD_list)
        
    def __remove_comma(column):
        column = [float(x.replace(',','')) if type(x) == str else float(x) for x in column]
        return column
    
    # 첫줄은 Symbol이니까 제외
    df = df.apply(lambda x: __remove_comma(x)).T
    
    df = df.loc[df.sum(axis = 1) != 0]
    
    return df



# 퍼센트로 표시된 수익률을 곱셈이 편한 형태로 변환해주는 함수
def rt_transform(column):
    return column.apply(lambda x: x/100 +1)




# 1개월 전 성과지표를 기준으로, 이번 달의 포트폴리오를 구성하고 매월 1일날 투자를 수행한다.
def portfolio_selection(df):
    
    # 높을수록 1그룹
    # 낮을수록 10 그룹
    
    data = df.copy()
    
    month_list = df.columns
    name_list = df.index

    # t월의 지표 횡단면을 분석하여, t+1월의 1일에 투자하는 것을 목표로 한다.
    # 따라서 t월 30일을 기준으로 포트폴리오를 그룹핑하고, t+1월 1일 부터 투자한다.
    # t+1월에 새로 상장된 주식의 경우 t월 성과가 없기 때문에 t+1월에는 투자되지 않는다 (0으로 처리)
    # t+1월에 사라진 주식은 t월 성적이 존재하지만 투자가 불가능하므로, t+1월에 제거하여 0으로 처리한다
    
    for cnt, month in enumerate(month_list):
        
        temp_t = df[month].dropna()
        
        # 첫월은 제외한다. 전부 0으로 처리 (0은 어느 포트폴리오에도 속하지 않은 주식들의 그룹)
        if cnt == 0:
            data[month] = 'x'
            t_index_list = temp_t.index
            
        # 마지막 월을 제외하고 결과를 확인한다
        if not cnt == len(month_list) - 1:
            
            temp_t_plus_1 = df[month_list[cnt+1]].dropna()
            
            
            t_index_list = temp_t.index
            t_plus_1_index_list = temp_t_plus_1.index
            
            
            
            # t월에 존재하지 않았던 펀드들은 t+1월에 x그룹으로 따로 분류한다 (t월 : x, t+1월 : 0)
            t_no_existence_list = [x for x in name_list if not x in t_index_list]
            
            # t월에 존재하였지만 t+1월에 사라진 종목들은 t+1월에 0그룹으로 분류한다 (t월 : 존재, t+1월: 사라짐)
            t_1_disappear_list = [x for x in t_index_list if not x in t_plus_1_index_list]
            

            # t월에 존재하지 않았던 종목은 t+1월에 0으로 처리한다
            data.loc[t_no_existence_list, month_list[cnt+1]] = 'x'
            
            # t월에 존재했지만 t+1월에 사라진 종목들은 역시 0로 따로 처리한다.
            ## 처리하면 안된다. 왜냐하면, 다음 달에 성과지표가 -로 가서 사라진건지, 상장 폐지된건지 알 수 없기 때문.. 일단 x로 처리하진 말자
            #data.loc[t_1_disappear_list, month_list[cnt+1]] = 'x'

            
            # 그룹핑을 위한 분위값 계산
            Q_value_list = [0]
            Q_value_list += [np.percentile(temp_t.values, i*10)for i in range(1,11)]
            
            
            for i, Q in enumerate(Q_value_list):
                
                if not i == 10:
                    df_temp = temp_t[temp_t > Q] <= Q_value_list[i+1]
                    df_temp_index = df_temp[df_temp == True].index
                    
                    data.loc[df_temp_index, month_list[cnt+1]] = i
                    
    
    return data.iloc[:,:-1] # 마지막 달 제거


def performance_analysis(data, rt_df):

    month_list = data.columns

    group_list = ['GROUP_%s'%(i) for i in range(10)]
    group_rt_dict = dict(zip(group_list, [[] for _ in range(10)]))
    
    for cnt,month in enumerate(month_list):

        # 최초월 1일에는 1원씩 투자
        if cnt == 0 :
            for i in range(10):
                group_rt_dict['GROUP_%s'%i].append(1)

        # 둘째 달부터는 1일에 전달 기록에 근거한 투자를 한다
        else:
            temp = data[month]
            
            # 마지막 달은 제외하고
            if not cnt == len(month_list) -1:
                rt_t = rt_df[month_list[cnt]].dropna()
                rt_t_plus_1 = rt_df[month_list[cnt+1]].dropna()

                # t-1월에 수익률이 존재하였지만 t월에 사라진 종목들은 상장폐지된 종목으로 결정하고 -100% 수익률로 측정한다.
                Abolished_index = [x for x in rt_t.index if not x in rt_t_plus_1.index]
                #print(month, Abolished_index)
                rt_df.loc[Abolished_index, month] = 0
                       
            for i in range(10): # i는 1,2,3,4 ~ 10
                group_index = temp[temp == i].index
                
                #print(month, i,  len(group_index))

                group_mean_rt = np.mean(rt_df.loc[group_index, month].dropna().values)
                group_rt_dict['GROUP_%s'%i].append(group_mean_rt)
    
    return pd.DataFrame(group_rt_dict)
    