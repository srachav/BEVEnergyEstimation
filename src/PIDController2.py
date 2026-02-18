from VelocityProfileGenerator2 import generateVelProf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.gridspec as gridspec


class vehModel:
    def __init__(self, nodePD, add_noise=True, sf_curve=10, sf_stop=4, stop_dist_csv="StoppingDistData.csv",
                 aMax=2, aMin=-4, jMax=10, jMin=-10, kphToMs=0.277778,
                 noise_std_spdlim=2, noise_std_curv=1, noise_clip_spdlim=5.0, noise_clip_curv=2.0,
                 horizon_extra=50, horizon_array_len=100,
                 dvdt_gain=0.5, dadt_gain=4, djdt_gain=15, zero_vel_thresh=0.01,
                 low_speed_threshold=3, stop_margin=5, curvFacThd=None, intrsHeadThd=None):
        """Vehicle model that builds speed-limit track from provided `nodePD`.

        Args:
            nodePD: pandas.DataFrame or path to CSV containing node attributes
            add_noise: bool, whether to add random noise to command velocity
            sf_curve: scaling factor for curve events (default 10)
            sf_stop: scaling factor for stop events (default 4)
            stop_dist_csv: path to stopping distance CSV (default 'StoppingDistData.csv')
            aMax/aMin/jMax/jMin: acceleration/jerk limits
            kphToMs: conversion factor from kph to m/s
            noise_std_spdlim/noise_std_curv: noise std dev for cmdVel in different states
            horizon_extra: extra horizon distance added to stopping distance
            horizon_array_len: preallocated horizon array length
            dvdt_gain/dadt_gain/djdt_gain: gains used in state derivative calculations
            zero_vel_thresh: threshold below which velocity is considered zero
            low_speed_threshold: speed threshold used in stop-event handling
            stop_margin: additional margin for stop decision
            curvFacThd/intrsHeadThd: optional thresholds passed to `generateVelProf`
        """
        self.sf_curve = sf_curve
        self.sf_stop = sf_stop
        self.stopDistData = pd.read_csv(stop_dist_csv)
        self.aMax = aMax
        self.aMin = aMin
        self.jMax = jMax
        self.jMin = jMin
        # Use VelocityProfileGenerator2 to generate speed-limit profile from nodePD
        # pass through optional curve/intersection thresholds if provided
        if curvFacThd is not None or intrsHeadThd is not None:
            self.curvFacThd = curvFacThd
            self.intrsHeadThd = intrsHeadThd
            self.rawSpdLimData = np.array(generateVelProf(nodePD, curvFacThd=self.curvFacThd, intrsHeadThd=self.intrsHeadThd))
        else:
            self.curvFacThd = curvFacThd
            self.intrsHeadThd = intrsHeadThd
            self.rawSpdLimData = np.array(generateVelProf(nodePD))
        self.currIdx = 0
        self.currDist = 0
        self.currVel = 0
        self.currAcc = 0
        self.currJerk = 0
        self.currState = 0
        self.stoppingDistInterp = interp1d(self.stopDistData['Speed (m/s)'], self.stopDistData['Stopping Distance(m)'], kind='linear', fill_value='extrapolate')
        self.stoppingDistInterpLowAcc = interp1d(self.stopDistData['Speed (m/s)'], self.stopDistData['Stopping Distance 2(m)'], kind='linear', fill_value='extrapolate')
        self.cmdVel = 0
        self.switchFlag = False
        self.eventStartIdx = 0
        self.zeroReached = False
        self.simStopFlag = False
        # control whether random noise is added to command velocity
        self.add_noise = add_noise
        # configurable parameters
        self.kphToMs = kphToMs
        self.noise_std_spdlim = noise_std_spdlim
        self.noise_std_curv = noise_std_curv
        self.noise_clip_spdlim = noise_clip_spdlim
        self.noise_clip_curv = noise_clip_curv
        self.horizon_extra = horizon_extra
        self.horizon_array_len = horizon_array_len
        self.dvdt_gain = dvdt_gain
        self.dadt_gain = dadt_gain
        self.djdt_gain = djdt_gain
        self.zero_vel_thresh = zero_vel_thresh
        self.low_speed_threshold = low_speed_threshold
        self.stop_margin = stop_margin

    def getSimStopFlag(self):
        return self.simStopFlag

    def getStateDeriv(self, t, y):
        # Ensure y is numeric array and has finite values
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).all():
            raise ValueError(f"State vector 'y' contains non-finite values at t={t}: {y}")
        self.currDist = y[0]  # Distance
        self.currVel = y[1]  # Velocity 
        self.currAcc = y[2]  # Acceleration
        self.currJerk = y[3]  # Jerk
        # print(self.currDist)
        self.updateCmdVel()
        dydt = np.zeros(4, dtype=float)
        dydt[0] = self.currVel # Velocity
        dydt[1] = self.currAcc # Acc
        dydt[2] = self.currJerk # Jerk
        dvdt = min(max(self.dvdt_gain*(self.cmdVel - self.currVel), self.aMin), self.aMax)
        dadt =  min(max(self.dadt_gain*(dvdt - self.currAcc), self.jMin), self.jMax)
        djdt =  self.djdt_gain*(dadt - self.currJerk)
        dydt[3] = djdt  # Jerk derivative

        if not np.isfinite(dydt).all():
            raise ValueError(f"Computed derivative 'dydt' contains non-finite values at t={t}, y={y}, dydt={dydt}")

        return dydt

    def updateCmdVel(self):
        while self.rawSpdLimData[self.currIdx,0] <= self.currDist:
            self.currIdx += 1
            if self.currIdx >= len(self.rawSpdLimData):
                self.simStopFlag = True
                self.cmdVel = 0
                return
        self.currIdx = self.currIdx - 1
        if self.currIdx < 0:
            self.currIdx = 0
        if self.currState == 0:
            self.cmdVel = self.cmdVelFromSpdLimTrack()
            if self.add_noise:
                n = np.random.normal(0, self.noise_std_spdlim)
                n = float(np.clip(n, -self.noise_clip_spdlim, self.noise_clip_spdlim))
                self.cmdVel += n  # Add clipped noise to cmdVel for more realistic simulation
        elif self.currState == 1:
            self.cmdVel = self.cmdVelFromStopEvent()
        elif self.currState == 2:
            self.cmdVel = self.cmdVelFromCurvTrack()
            if self.add_noise:
                n = np.random.normal(0, self.noise_std_curv)
                n = float(np.clip(n, -self.noise_clip_curv, self.noise_clip_curv))
                self.cmdVel += n  # Add clipped noise to cmdVel for more realistic simulation
        return

    def cmdVelFromSpdLimTrack(self):
        cmdVel = self.currVel
        horizon = self.stoppingDistInterp(self.rawSpdLimData[self.currIdx, 1]*self.kphToMs) + self.horizon_extra
        futIdx = self.currIdx + 1
        horizonData = np.zeros((5,self.horizon_array_len)) # [SpdLims, SpdLimTypes, Idx, DistToSpdLim, DistReqForSpdLim]
        horizonIdx = 0
        if futIdx < len(self.rawSpdLimData):
            while self.rawSpdLimData[futIdx, 0] <= self.currDist + horizon:
                if self.rawSpdLimData[futIdx, 1] != self.rawSpdLimData[self.currIdx, 1]:
                    horizonData[0, horizonIdx] = self.rawSpdLimData[futIdx, 1]
                    horizonData[1, horizonIdx] = self.rawSpdLimData[futIdx, 2]
                    horizonData[2, horizonIdx] = futIdx
                    horizonData[3, horizonIdx] = self.rawSpdLimData[futIdx, 0] - self.currDist
                    speedDrop = self.currVel  - self.rawSpdLimData[futIdx, 1]*self.kphToMs
                    if self.rawSpdLimData[futIdx, 2] != 3:  
                        speedDropDistReq = self.stoppingDistInterp(self.currVel)*speedDrop/max(0.1,self.currVel)
                    else:
                        speedDropDistReq = self.stoppingDistInterp(self.currVel)
                    horizonData[4, horizonIdx] = speedDropDistReq
                    horizonIdx += 1
                    # if horizonIdx >= self.horizon_array_len:
                    #     break
                futIdx += 1
                if futIdx >= len(self.rawSpdLimData):
                    break

        if horizonIdx == 0:
            horizonIdx = 1
        
        if not (horizonData[1, horizonIdx - 1] == 0):  
            horizonIdx -= 1
            while horizonIdx >= 0:
                if horizonData[1, horizonIdx] == 3 and horizonData[3, horizonIdx] < horizonData[4, horizonIdx] + self.sf_stop:
                    self.currState = 1
                    self.eventStartIdx = np.int32(horizonData[2, horizonIdx])
                    return cmdVel
                elif horizonData[1, horizonIdx] == 2 and horizonData[3, horizonIdx] < horizonData[4, horizonIdx]*self.sf_curve:
                    self.currState = 2
                    self.eventStartIdx = np.int32(horizonData[2, horizonIdx])
                    return cmdVel
                elif horizonData[1, horizonIdx] == 2 and horizonData[3, horizonIdx] < 0 and horizonData[3, horizonIdx] <= abs(horizonData[4, horizonIdx])*self.sf_curve*1.5:
                    cmdVel = horizonData[0, horizonIdx]*self.kphToMs
                
                horizonIdx -= 1

        if self.rawSpdLimData[self.currIdx, 1] == 0:  
            self.currIdx = self.currIdx + 1
        
        cmdVel = self.rawSpdLimData[self.currIdx, 1]*self.kphToMs
        return cmdVel

    def cmdVelFromStopEvent(self):
        self.simStopFlag = False
        cmdVel = self.currVel
        if self.currVel <= self.zero_vel_thresh and self.currAcc < 0:
            self.zeroReached = True

        if self.currDist <= self.rawSpdLimData[self.eventStartIdx, 0] and self.zeroReached :
            if self.eventStartIdx + 1 >= len(self.rawSpdLimData):
                self.simStopFlag = True
            else:
                if self.rawSpdLimData[self.eventStartIdx+1, 1]*self.kphToMs < self.low_speed_threshold:
                    cmdVel = self.rawSpdLimData[self.eventStartIdx+1, 1]*self.kphToMs
                else:
                    cmdVel = self.low_speed_threshold
        else:
            distToStop = self.rawSpdLimData[self.eventStartIdx, 0] - self.currDist
            if self.currAcc > 0.5 :
                stoppingDist = self.stoppingDistInterp(self.currVel)
            else:
                stoppingDist = self.stoppingDistInterpLowAcc(self.currVel)
            if self.currDist <= self.rawSpdLimData[self.eventStartIdx, 0] :
                if distToStop <= stoppingDist + self.stop_margin and not self.switchFlag:
                    cmdVel = 0
                    self.switchFlag = True
                elif self.switchFlag:
                    cmdVel = 0
                else:
                    cmdVel = self.rawSpdLimData[self.currIdx, 1]*self.kphToMs
            
            else:
                cmdVel = 0
                self.currState = 0
                self.zeroReached = False
                self.switchFlag = False

        return cmdVel    


    def cmdVelFromCurvTrack(self):
        cmdVel = self.cmdVel
        horizon = self.stoppingDistInterp(self.rawSpdLimData[self.currIdx, 1]*self.kphToMs) + self.horizon_extra
        futIdx = self.currIdx + 1
        horizonData = np.zeros((5,self.horizon_array_len))    # [SpdLims, SpdLimTypes, Idx, DistToSpdLim, DistReqForSpdLim]
        horizonIdx = 0
        if futIdx < len(self.rawSpdLimData):
            while self.rawSpdLimData[futIdx, 0] <= self.currDist + horizon:
                if self.rawSpdLimData[futIdx, 2] == 2 or self.rawSpdLimData[futIdx, 2] == 3:
                    horizonData[0, horizonIdx] = self.rawSpdLimData[futIdx, 1]
                    horizonData[1, horizonIdx] = self.rawSpdLimData[futIdx, 2]
                    horizonData[2, horizonIdx] = futIdx
                    horizonData[3, horizonIdx] = self.rawSpdLimData[futIdx, 0] - self.currDist
                    speedDrop = self.currVel  - self.rawSpdLimData[futIdx, 1]*self.kphToMs
                    if self.rawSpdLimData[futIdx, 2] != 3:  
                        speedDropDistReq = self.stoppingDistInterp(self.currVel)*speedDrop/self.currVel
                    else:
                        speedDropDistReq = self.stoppingDistInterp(self.currVel)
                    horizonData[4, horizonIdx] = speedDropDistReq
                    horizonIdx += 1
                futIdx += 1
                if futIdx >= len(self.rawSpdLimData):
                    break
        
        if horizonIdx == 0:
            horizonIdx2 = 1
        else:
            horizonIdx2 = horizonIdx
        
        if not (horizonData[1, horizonIdx2 - 1] == 0):
            horizonIdx = horizonIdx2 - 1
            while horizonIdx >= 0:
                if horizonData[1, horizonIdx] == 3 and horizonData[3, horizonIdx] <= horizonData[4, horizonIdx] + self.sf_stop:
                    self.currState = 0
                    break
                elif horizonData[1, horizonIdx] == 2 and horizonData[3, horizonIdx] <= horizonData[4, horizonIdx]*self.sf_curve:
                    if self.currVel >= horizonData[0, horizonIdx]*self.kphToMs:
                        scf = 0.2
                        cmdVel = horizonData[0, horizonIdx]*self.kphToMs - abs(horizonData[0, horizonIdx]*self.kphToMs*np.tanh(horizonData[4, horizonIdx]*scf/horizonData[3, horizonIdx]))
                        break
                horizonIdx -= 1

        if (self.rawSpdLimData[self.currIdx, 2] != 2 and self.rawSpdLimData[self.currIdx-1, 2] == 2) or (horizonData[1, horizonIdx2 - 1] == 0):
            cmdVel = self.rawSpdLimData[self.currIdx, 1]*self.kphToMs
            self.currState = 0
        elif self.cmdVel <= 0.1:
            cmdVel = self.rawSpdLimData[self.currIdx, 1]*self.kphToMs
            self.currState = 0
        
        return cmdVel
    


def EulInt(t0, y0, h, dydt, getStopFlag):
    t = t0
    y = np.array(y0)
    tdisp = [t0]
    ydisp = [y0]
    stopFlag = getStopFlag()
    while not stopFlag:
        slope = np.array(dydt(t, y))
        y = y + h * slope
        t += h
        tdisp.append(t)
        ydisp.append(y)
        stopFlag = getStopFlag()
    ydisp = np.array(ydisp)
    return [tdisp, ydisp]

if __name__ == "__main__":
    # demo: load nodes.csv and run simulation
    try:
        nodePD = pd.read_csv("nodes_phase1.csv")
    except Exception as e:
        raise SystemExit("nodes_phase1.csv not found for demo")
    vehModelObj = vehModel(nodePD)
    t0 = 0
    y0 = [0, 0, 0, 0]
    h = 0.01
    [tdisp, ydisp] = EulInt(t0, y0, h, vehModelObj.getStateDeriv, vehModelObj.getSimStopFlag)
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax1 = plt.subplot(gs[0],  sharex=None)
    ax1.plot(ydisp[:,0], ydisp[:, 1]/vehModelObj.kphToMs, label='Generated Velocity')
    ax1.step(vehModelObj.rawSpdLimData[:,0], vehModelObj.rawSpdLimData[:,1], where='post', label='Raw Speed limits')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Speed (kph)')
    ax1.set_title('Vehicle Speed Profile')
    ax1.grid(True)
    ax1.minorticks_on()
    ax1.grid(which='minor', linestyle='--', linewidth=0.5)
    ax1.legend()
    ax2 = plt.subplot(gs[1],  sharex=ax1)
    ax2.step(vehModelObj.rawSpdLimData[:,0], vehModelObj.rawSpdLimData[:,2], where='post',label='Speed limit type')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Speed Limit Type')
    ax2.grid(True)
    ax2.minorticks_on()
    ax2.grid(which='minor', linestyle='--', linewidth=0.5)
    ax2.legend()
    plt.show()
