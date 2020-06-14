import pip
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from math import log, pi, sqrt, floor, ceil
from datetime import datetime as d, timedelta
subprocess.check_call([sys.executable, "-m", "pip", "install", "pyhht"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa"])
from pyhht import emd, utils, visualization
from librosa import filters

class marsHHT:
    def plot_imfs(self,imfs):
        plt.figure
        for i,imf in enumerate(imfs):
            plt.subplot(len(imfs),1,i+1)
            plt.plot(imf)
        plt.show()

    def IOt(self,imfs,x):
        dp = np.dot(imfs, np.conj(imfs).T)
        mask = np.logical_not(np.eye(len(imfs)))
        s = np.abs(dp[mask]).sum()
        return s / (2 * np.sum(x ** 2))

    def __orthogonal_factor(self,c,d):
        num = np.dot(c,d.T)
        den = np.dot(d,d.T)
        return num/den

    def orthogonal_emd(self,imfs):
        num_imfs = len(imfs)
        t = len(imfs[0])
        res = np.zeros((num_imfs,t))

        for i,imf in enumerate(reversed(imfs)):
            if i == 0:
                res[i] = imf
            else:
                of = np.zeros(shape=(i,))
                f = np.zeros(shape=(i,t))
                for j in range(i):
                    of[j] = self.__orthogonal_factor(imf,res[i-(j+1)])
                    f[j] = of[j]*res[i-(j+1)]
                d = np.sum(f,axis=0)
                res[i] = np.array([c-d_ for c,d_ in zip(imf,d)])
        return res

    def __get_bin(self,freq,n_bins,bins):
        if freq < 0:
            return -1
        ratio = int(n_bins/bins[len(bins)-1])
        index = int(freq*ratio)
        return index

    def __check_inputs(self,imfs):
        if len(imfs) < 1:
            return -1
        else: 
            return 1

    def __get_freq_amp(self,imf,fs):
        x_a = signal.hilbert(imf)
        p = np.unwrap(np.angle(x_a))
        inst_f = np.diff(p)/(2*np.pi)*fs #diff[i] = a[i+1] - a[i]
        inst_a = np.abs(x_a)[:len(imf)-1] 
        return inst_f,inst_a

    def hht(self, imfs, fs=100, n_freq=201):
        if len(imfs.shape) == 1:
            t = np.arange(len(imfs)-1)
        else:
            t = np.arange(len(imfs[0])-1)
        f = np.linspace(0,int(fs/2),n_freq)
        A = [[0 for i in range(len(t))] for j in range(len(f))]
        inst_a = [None]*(len(imfs))
        inst_f = [None]*(len(imfs))

        if len(imfs.shape) == 1:
            inst_f,inst_a = self.__get_freq_amp(imfs,fs)
            for i,c in enumerate(inst_a):
                freq = inst_f[i]
                freq_bin = self.__get_bin(freq,n_freq,f)
                if freq_bin != -1:
                    A[freq_bin][i] += c
        else:
            for i,imf in enumerate(imfs):
                inst_f[i],inst_a[i] = self.__get_freq_amp(imf,fs)
            for i,r in enumerate(inst_a):
                for j,c in enumerate(r):
                    freq = inst_f[i][j]
                    freq_bin = self.__get_bin(freq,n_freq,f)
                    if freq_bin != -1:
                        A[freq_bin][j] += c
        A = np.array(A)
        return A,f,t

    def plot_hs(self,A,f,t,title='Hilbert Spectrum',mode='log',fs=100):
        if mode == 'log': A = np.log(A)
        if mode == 'sqrt': A = np.log(np.sqrt(A))
        cmap = plt.pcolormesh(t/fs, f, A, cmap = 'jet')
        plt.colorbar(cmap)
        plt.title(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency[Hz]')
        plt.show()

    def get_corr(self,sig,imfs):
        res = [None]*len(imfs)
        for i,imf in enumerate(imfs):
            res[i] = abs(np.corrcoef(imf,sig)[0,1])
        return res

    def get_threshold(self,corr_coefs):
        return max(corr_coefs)/(10*max(corr_coefs)-3)

    def filter_imfs(self,corr_coefs, threshold, imfs):
        res = []
        for i,corr in enumerate(corr_coefs):
            if corr > threshold:
                res.append(imfs[i])
        res = np.array(res)
        return res

    def get_coeffs(self,A,num_ceps=13,num_filters=16,f_bins=400,fs=100,normalize=True,corr=False):
        fbank = filters.mel(fs,f_bins,num_filters, norm=None)
        fbank_coeffs = np.dot(fbank,A).T
        cc = fftpack.dct(fbank_coeffs, type=2, norm='ortho')[:, 1 : (num_ceps + 1)]
        if normalize == True:
            cc -= (np.mean(cc, axis=0) + 1e-8)
            fbank_coeffs -= (np.mean(fbank_coeffs, axis=0) + 1e-8)
        if corr == False:
            return cc
        else:
            return fbank_coeffs

    def plot_coeffs(self, coeffs, title = 'Cepstral Spectrogram'):
        t = np.arange(coeffs.shape[0])
        t = t*(15/len(t))
        mel = np.arange(coeffs.shape[1])
        coeffs = coeffs.T
        cmap = plt.pcolormesh(t, mel, coeffs, cmap = 'jet')
        plt.colorbar(cmap)
        plt.title(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Coefficient')
        plt.show()

    def delta(self,feat,N=2):
        M = len(feat)
        denominator = 2 * sum([i**2 for i in range(1, N+1)])
        delta_feat = np.empty_like(feat)
        padded = np.pad(feat, ((N, N), (0, 0)), mode='edge') 
        for t in range(M):
            delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator 
        return delta_feat

    def emd_postprocessing(self, eemds):
        num_signals = len(eemds)
        t = len(eemds[0])
        imfs = []
        sum_sig = eemds[0]
        residue = [0]*t
        for i in range(num_signals):
            decomposer = emd.EMD(sum_sig,n_imfs=2)
            imfs_sig = decomposer.decompose()
            if len(imfs_sig) == 1:
                residue = [0]*t
            else:
                residue = imfs_sig[1]
                imfs.append(imfs_sig[0])
            if i < num_signals-1:
                sum_sig = np.add(residue, eemds[i+1])
        decomposer = emd.EMD(np.array(residue),n_imfs=2)
        imfs_sig = decomposer.decompose()
        while(len(imfs_sig) > 1):
            imfs.append(imfs_sig[0])
            decomposer = emd.EMD(imfs_sig[0],n_imfs=2)
            imfs_sig = decomposer.decompose()
        residue = imfs_sig
        imfs = np.array(imfs)
        return imfs,residue

    def eemd(self, sig, trials=100, noise_width = 0.05):
        n = len(sig)
        imfs = []
        num_additions = []
        std = noise_width*np.std(sig)
        for i in range(trials):
            noise = np.random.normal(0,std,size=n)
            noisy_sig = np.add(sig,noise)
            decomposer = emd.EMD(noisy_sig)
            noisy_imfs = decomposer.decompose()
            #remove residue!
            noisy_imfs = noisy_imfs[:len(noisy_imfs)-1]
            if i == 0:
                imfs = noisy_imfs.tolist()
                num_additions = [1]*len(imfs)
            else:
                while len(noisy_imfs) > len(imfs):
                    imfs.append(0)
                    num_additions.append(0)
                for i,imf in enumerate(noisy_imfs):
                    imfs[i] = np.add(imfs[i],imf)
                    num_additions[i] += 1
        for i,imf in enumerate(imfs):
            imfs[i] = [x / num_additions[i] for x in imf]
        imfs = np.array(imfs)
        imfs = self.emd_postprocessing(imfs)[0]
        return imfs