import numpy as np
import pandas as pd
from math import log, pi, sqrt, floor, ceil
from datetime import datetime as d, timedelta

class preprocess:
    def __init__(self, vbb_data, sp_data):
        self.vbb_data = vbb_data
        self.sp_data = sp_data

    def combine(self):
        data1 = self.vbb_data.to_numpy()
        data2 = self.sp_data.to_numpy()
        assert len(data1) == len(data2)
        combined_data = np.zeros(shape = (len(data1),4), dtype='O')
        for i,row in enumerate(data1):
            combined_data[i][0] = row[0] 
            combined_data[i][1] = data1[i][1] + data2[i][1]
            combined_data[i][2] = data1[i][2] + data2[i][2]
            combined_data[i][3] = data1[i][3] + data2[i][3]

        d = {'t': [vec[0] for vec in combined_data], 'Z': [vec[1] for vec in combined_data], 
            'N': [vec[2] for vec in combined_data], 'E': [vec[3] for vec in combined_data], 
            'Label': [0 for vec in combined_data]} 
        df_result = pd.DataFrame(data = d)

        return df_result

    def __binary_search(self,dates, date, acc = 5):
        n = len(dates)
        L = 0
        R = n-1

        date_range = [date+timedelta(milliseconds=acc)-
                    timedelta(milliseconds=x) for x in range(2*acc)]

        while L <= R:
            mid = floor((L+R)/2)
            if d.strptime(dates[mid], '%Y-%b-%d %H:%M:%S.%f') in date_range:
                return mid
            elif d.strptime(dates[mid], '%Y-%b-%d %H:%M:%S.%f') < date:
                L = mid+1
            elif d.strptime(dates[mid], '%Y-%b-%d %H:%M:%S.%f') > date:
                R = mid - 1
        return -1

    def get_events(self, time_data, label_data, frame=0, lim=99, label = 1, false_events = []):
        n = len(time_data)
        event_data = []
        label_data = label_data.to_numpy()
        for i,r in enumerate(label_data):
            date_from = d.strptime(r[0], '%Y-%m-%d %H:%M:%S')
            date_to = d.strptime(r[1], '%Y-%m-%d %H:%M:%S')
            if date_from >= d.strptime(time_data['t'][0], '%Y-%b-%d %H:%M:%S.%f') and date_to <= d.strptime(time_data['t'][n-1], '%Y-%b-%d %H:%M:%S.%f'):
                pos1 = self.__binary_search(time_data['t'], date_from)
                pos2 = self.__binary_search(time_data['t'], date_to)
                if pos2-pos1 >= lim and i not in false_events:
                    if frame == 0:
                        event_data.append(time_data.loc[pos1:pos2, ['Z','N','E']])
                        time_data.loc[pos1:pos2, 'Label'] = 1
                    else:
                        assert type(frame)==int
                        event_data.append(time_data.loc[pos1:pos1+frame, ['Z','N','E']])
                        time_data.loc[pos1:pos1+frame, 'Label'] = label 
        return event_data,time_data

    def get_negatives(self, time_data, n, start=0, N = 501):
        neg_data = time_data.loc[time_data['Label']==0]
        neg_data = neg_data.drop('t', axis=1)
        neg_data = neg_data.drop('Label',axis=1)
        neg_data_arr = neg_data.to_numpy()
        res = [None]*n
        for i in range(n):
            data = neg_data_arr[start+(i*N):start+((i*N)+N)]
            d = {'Z': [vec[0] for vec in data], 'N': [vec[1] for vec in data], 
                'E': [vec[2] for vec in data]} 
            res[i] = pd.DataFrame(data=d)
        return res

    def create_target(self,num_events,num_negs):
        target_events = np.array([1 for i in range(num_events)])
        target_negs = np.array([0 for i in range(num_negs)])
        target_data = np.concatenate((target_negs,target_events))
        return target_data

#data_vbb = pd.read_csv('C:/Users/rohan/Documents/FYP/FYPdata/VBB_data.txt')
#data_sp = pd.read_csv('C:/Users/rohan/Documents/FYP/FYPdata/SP_data.txt')
#label_data = pd.read_csv('C:/Users/rohan/Documents/FYP/FYPdata/Label Dates')

#p = preprocess(data_vbb,data_sp)
#tdbl = p.combine()
#time_data = p.label_events(tdbl,label_data,frame=500)
#print(time_data)
#donk_data = p.get_events(time_data, N=501)
#neg_data = p.get_negatives(time_data, len(donk_data))
#target_data = p.create_target(len(donk_data),len(neg_data))

#print(target_data, len(target_data))
#assert len(target_data) == 162