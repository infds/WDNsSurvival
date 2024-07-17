import pandas as pd
from sksurv.util import Surv
import numpy as np


"""
代码均在实验室环境下，读取数据从csv文件中读取，在实际项目中需要写sql语句
"""


def read_csv(path):
    df = pd.read_csv(path,index_col=None)
    df.dropna(subset=['管段编号'], inplace=True)
    return df

def standardize_time_format(df:pd.DataFrame)->pd.DataFrame:
    df.dropna(subset=['竣工接收时'],inplace=True)
    df.dropna(subset=['接报时间'],inplace=True)
    df['竣工接收时'] = pd.to_datetime(df['竣工接收时'], format='%Y/%m/%d %H:%M:%S.%f')
    df['接报时间'] = pd.to_datetime(df['接报时间'], format='%Y-%m-%d %H:%M:%S')
    df['管龄(月)'] = ((df['接报时间'] - df['竣工接收时'])/ pd.to_timedelta(30, unit='D')).round(1)
    df=df.drop(columns=['接报时间'])
    return df

def properties_mapping(df:pd.DataFrame)->pd.DataFrame:
    material_mapping = {'PPR管': 0, '球墨铸铁': 1, '钢塑管': 2, 'UPVC管': 3, '玻璃钢管': 4, '镀锌管': 5, '钢管': 6,
                        '灰口铸铁': 7, '砼管': 8, 'PE管': 9}
    connection_mapping = {'热熔': 0, '承插式胶圈接口': 1, '螺纹连接': 2, '承插式石棉水泥接口': 3,
                          '承接式石棉水泥接口': 4, '法兰': 5, '沟槽连接': 6, '焊接': 7}
    df['材质'] = df['材质'].map(material_mapping)
    df['连接类型'] = df['连接类型'].map(connection_mapping)

    return df


def data_merge(df1_wdns_path:str,df2_repairs_records_path:str)->pd.DataFrame:
    wdns = read_csv(df1_wdns_path)
    repair_records = read_csv(df2_repairs_records_path)
    not_censored_pipline= wdns[~wdns['管段编号'].isin(repair_records['管段编号'])]
    sampled_pipline = not_censored_pipline.sample(n=min(1000, len( not_censored_pipline)), random_state=42)
    repair_records.sort_values(by='接报时间')
    repairs_df_cleaned = repair_records.drop_duplicates(subset=["管段编号"],keep='first')
    censored_pipline=wdns[wdns['管段编号'].isin(repair_records['管段编号'])]
    censored_pipline=censored_pipline.merge(repairs_df_cleaned[['管段编号','接报时间']],on='管段编号',how='left')
    censored_pipline=standardize_time_format(censored_pipline)
    sampled_pipline['Event']=0
    censored_pipline['Event']=1
    merged_data=pd.concat([sampled_pipline,censored_pipline])
    merged_data=merged_data.drop(columns=['竣工接收时','运行状态'])
    merged_data=merged_data.rename(columns={"管龄(月)": "Time"})
    merged_data=merged_data[merged_data['Time']>=0]
    merged_data=properties_mapping(merged_data)
    return merged_data


class PiplineRecoder():
    def __init__(self,raw_data:pd.DataFrame,feature_list:list):

        self.pipline_features=None
        self.pipline_Number=None
        self.pipline_idx=None
        self.label=None
        self.feature_list=feature_list
        self.raw_data=raw_data
        self.data_init()

    def data_init(self):
        self.pipline_features=self.raw_data[self.feature_list]
        self.pipline_Number=self.raw_data['管段编号']
        self.label=Surv.from_dataframe("Event","Time",self.raw_data)
        i=0
        idx=dict()
        for number in self.pipline_Number:
            idx[number]=i
            i=i+1
        self.pipline_idx=idx

    def get_pipline_features(self):
        return self.pipline_features

    def get_pipline_idx(self):
        return self.pipline_idx

    def get_pipline_Number(self):
        return self.pipline_Number

    def get_label(self):
        return self.label

    def get_pipline_feature_by_pipline_Number(self,pipline_number:str)->np.ndarray:
        idx=self.pipline_idx[pipline_number]
        return self.pipline_features.iloc[[idx]].values[0]

    def get_pipline_label_by_pipline_Number(self,pipline_number:str)->pd.DataFrame:
        idx = self.pipline_idx[pipline_number]
        return self.label[idx]




if __name__ == '__main__':
    data=data_merge("Wdns.csv","repairs.csv")[:100]
    pipline_recoder=PiplineRecoder(data,feature_list=["公称直径", "材质", "管长", "连接类型"])
    print(pipline_recoder.get_pipline_feature_by_pipline_Number("841429GX4045756"))





