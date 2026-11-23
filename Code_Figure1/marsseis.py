#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:32:10 2024

@author: planet_science
"""
import numpy as np
import pandas as pd
from obspy import UTCDateTime

class lmst(object):
    def __init__(self,sol=0,hour=0,minute=0,sec=0):
        self.sol = sol
        self.hour = hour
        self.min = minute
        self.sec = sec
    def __str__(self):
        return ('Sol-''%.5d''M''%.2d'':''%.2d'':''%05.2f') %(self.sol,self.hour,self.min,self.sec)
    __repr__ = __str__
    def __sub__(self,other):
        #time difference in seconds
        sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
        sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
        sec_m = sol2_start - sol1_start - 0.000005
        sec_e = 86400
        t1 = self.conv2utc()
        t2 = other.conv2utc()
        return (t1-t2)*sec_e/sec_m
    def __add__(self,dt_m):
        sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
        sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
        sec_m = sol2_start - sol1_start - 0.000005
        sec_e = 86400
        dt_e = dt_m*sec_m/sec_e
        t1 = self.conv2utc()
        t2 = t1+dt_e
        tmp = lmst()
        tmp.utc2lmst(t2)
        return tmp
    def _display(self):
        # = print
        print('Sol-''%.5d''M''%.2d'':''%.2d'':''%05.2f' %(self.sol,self.hour,self.min,self.sec))
    def get_sol_hour_min(self):
        return ('Sol''%.3d''\n''%.2d'':''%.2d') %(self.sol,self.hour,self.min)
    def get_hour(self):
        return self.hour
    def get_hour_min(self):
        return ('%.2d''.''%.2d') %(self.hour,self.min)
    def conv2utc(self):
        ref = UTCDateTime("2018-11-26T05:10:50.336037Z")
        sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
        sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
        sec_m = sol2_start - sol1_start - 0.000005
        sec_e = 86400
        # calculate secs on Mars
        dur_m = self.sol*24*3600+self.hour*3600+self.min*60+self.sec
        dur_e = dur_m*sec_m/sec_e
        return ref+dur_e
    def utc2lmst(self,utc_time):        
        sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
        sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
        sec_m = sol2_start - sol1_start - 0.000005
        sec_e = 86400
        dur_e = utc_time-UTCDateTime("2018-11-26T05:10:50.336037Z")
        dur_m = dur_e*sec_e/sec_m
        self.sol = dur_m // 86400
        mod = dur_m % 86400
        self.hour = mod // 3600
        mod = mod %3600
        self.min = mod // 60
        self.sec = mod % 60
    def move_to_start(self):
        if (self.min == 0)&((self.sec-0.0)<1e-3):
            return
        else:
            self.min = 0
            self.sec = 0
            if self.hour == 23:
                self.hour = 0
                self.sol += 1
            else:
                self.hour += 1
    def move_to_end(self):
        self.min = 0
        self.sec = 0
    def check(self):
        if round(self.sec) == 60:
            self.sec = 0
            self.min += 1
        if self.min == 60:
            self.min = 0
            self.hour += 1
        if self.hour == 24:
            self.hour = 0
            self.sol += 1

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def rssq(x):
    return np.sqrt(np.sum(x**2)/len(x))

def snr(x, y):
    spow = rssq(x)**2
    npow = rssq(y)**2
    if npow == 0:
        return np.nan
    else:
        return 10 * np.log10(spow / npow)