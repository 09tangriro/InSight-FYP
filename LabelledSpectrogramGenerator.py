import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from math import log, pi, sqrt, floor, ceil
from datetime import datetime as d, timedelta
from scipy import signal, fftpack


class LSG:
    def __init__(self,vbb_data = None,sp_data=None,label_data=pd.DataFrame(),col=None):
        self.vbb_data = vbb_data
        self.sp_data = sp_data
        self.label_data = label_data
        self.col = col
        self.time_data = None
        self.fig = None


    def __combine(self):
        #Combine the VBB and SP datasets into one.
        #OUTPUT: pandas dataframe of combined data.

        data1 = self.vbb_data.to_numpy()
        data2 = self.sp_data.to_numpy()
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
        #Search for labelled dates in the data.
        #OUTPUT: an integer of the index in the data relating to the label.

        n = len(dates)
        L = 0
        R = n-1

        #acc represents a precision value. search for the date in a certain date_range.
        #A lower acc -> more precise but less consistent results 
        #(possible to never find the value due to discrete nature of data).
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


    def plot_graphs(self, fs=100):
        #Generate the spectrogram plot with the labels if there are any.
        #OUTPUT: the spectrogram matplotlib figure

        plt.rcParams["figure.figsize"] = (8,4)
        x = np.arange(len(self.time_data['t']))
        x = np.array([int(n/100) for n in x])
        plt.plot(x,self.time_data[self.col])
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (ms^-2)')
        plt.title('Time Series '+self.col+' Axis')
        plt.grid(b=True, which='both')
        plt.show()

        f, t, Sxx_VB = signal.spectrogram(x = self.time_data[self.col], fs = fs,
                                        window = 'hann', nperseg = 256,
                                        noverlap = 128)
        logSxx_VB = np.log10(np.sqrt(Sxx_VB))

        if type(self.label_data) == np.ndarray:
            for _,r in enumerate(self.label_data):
                pos1 = 1501*r
                pos2 = (1501*r)+1501
                x = pos2/128
                if x > logSxx_VB.shape[1]-1: x = logSxx_VB.shape[1]-1
                logSxx_VB[:,floor(pos1/128)] = -9.6 
                logSxx_VB[:,floor(x)] = -9.55 

        elif not self.label_data.empty:
            n = len(self.time_data)
            label_data_np = self.label_data.to_numpy()
            for _,r in enumerate(label_data_np):
                date_from = d.strptime(r[0], '%Y-%m-%d %H:%M:%S')
                date_to = d.strptime(r[1], '%Y-%m-%d %H:%M:%S')
                if date_from >= d.strptime(self.time_data['t'][0], '%Y-%b-%d %H:%M:%S.%f') and date_to <= d.strptime(self.time_data['t'][n-1], '%Y-%b-%d %H:%M:%S.%f'):
                    pos1 = self.__binary_search(self.time_data['t'], date_from)
                    pos2 = self.__binary_search(self.time_data['t'], date_to)
                    logSxx_VB[:,floor(pos1/128)] = -9.6
                    logSxx_VB[:,floor(pos2/128)] = -9.55
        
        jet = cm.get_cmap('jet', 256)
        newcolors = jet(np.linspace(0, 1, 1024))
        black = np.array([0, 0, 0, 1])
        white = np.array([1,1,1,1])
        newcolors[:5, :] = black
        newcolors[5:15,:] = white
        jet_ = ListedColormap(newcolors)
        fig, ax = plt.subplots()
        cmap = ax.pcolormesh(t, f, logSxx_VB, cmap = jet_, vmin = -9.6,vmax = -6)
        fig.colorbar(cmap)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency[Hz]')
        plt.title('Spectrogram '+self.col+' Axis')
        #plt.savefig('E:/Users/rohan/Documents/FYP/Report Pics/spectrogram.eps', format='eps')
        plt.show()
        return fig    

    def save_fig(self, filename):
        #Save a generated figure to permanent memory as a pickled plot.
        
        filename = str(filename)
        pickle.dump(self.fig, open(filename, 'wb'))

    def __show_figure(self,fig):
        # Create a dummy figure and use its manager to display "fig".

        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = fig
        fig.set_canvas(new_manager.canvas)

    def load_fig(self,filename):
        #Load a pre-saved figure from permanent memory.
        #One of two main interfacing functions.

        self.fig = pickle.load(open(filename, 'rb'))
        plt.show(self.__show_figure(self.fig))

    def generate(self, save=True, filename = None):
        #Generate a plot from scratch using data.
        #One of the two main interfacing functions.
        
        self.time_data = self.__combine()
        self.fig = self.plot_graphs()
        if save == True:
            self.save_fig(filename)

data_vbb = pd.read_csv('E:/Users/rohan/Documents/FYP/FYPdata/VBB_data_98.txt')
data_sp = pd.read_csv('E:/Users/rohan/Documents/FYP/FYPdata/SP_data_98.txt')
#label_data = pd.read_csv('E:/Users/rohan/Documents/FYP/FYPdata/Label Dates')
#label_data = np.load('E:/Users/rohan/Documents/FYP/FYPdata/donk_frames_hhtcc_hmm_98.npy')

#Generate: data_vbb, data_sp, label_data, 'Z'
#Load: optional, leave empty
lsg = LSG(data_vbb, data_sp, col='Z')

lsg.generate(save=False, filename = 'E:/Users/rohan/Documents/FYP/Spectrogram_Z_Result_98.pickle')

#lsg.load_fig('E:/Users/rohan/Documents/FYP/Spectrogram_Z_Result_98.pickle')
#lsg.load_fig(str(sys.argv[1]))