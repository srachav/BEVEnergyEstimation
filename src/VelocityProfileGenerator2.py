import pandas as pd

def generateVelProf(nodePD, curvFacThd=2, intrsHeadThd=60):
    """Generate speed-limit profile from a nodes DataFrame or path.

    Args:
        nodePD: pandas.DataFrame or str path to CSV containing node attributes.
        curvFacThd: curvature factor threshold used to mark reduced speed segments (default 2).
        intrsHeadThd: heading difference threshold (degrees) for identifying intersections (default 60).

    Returns:
        List of [range, speed_limit, speed_limit_type] entries (same format as original).
    """
    if isinstance(nodePD, str):
        nodePD = pd.read_csv(nodePD)

    spdLimData = []
    # thresholds are taken from function arguments; defaults provided in signature

    for node in nodePD.itertuples():
        nodeSpdLimData = [node.range]
        speedLim = node.speed_limit
        if speedLim == 0:
            speedLim = node.edge_speed
        speedLimType = 1
        if node.edge_curvature > curvFacThd or node.round_about == True:
            speedLimType = 2
        nodeSpdLimData.append(speedLim)
        nodeSpdLimData.append(speedLimType)
        spdLimData.append(nodeSpdLimData)

        if (node.traffic_signal == True or node.stop_sign == True or node.yield_sign == True) \
            or (node.percent_edge == 0 and ( not pd.isna(node.intersection_type))  and angleDiff(nodePD.iloc[max(0,node.Index - 1)]["heading"], node.heading) > intrsHeadThd) \
            or (node.percent_edge == 1 and ( not pd.isna(node.intersection_type))  and angleDiff(nodePD.iloc[min(node.Index + 1, len(nodePD) - 1)]["heading"], node.heading) > intrsHeadThd):
            stopSpeedLim = 0
            stopSpeedLimType = 3

            nodeSpdLimData = [node.range+0.1]
            nodeSpdLimData.append(stopSpeedLim)
            nodeSpdLimData.append(stopSpeedLimType)
            spdLimData.append(nodeSpdLimData)

            nodeSpdLimData = [node.range+0.2]
            nodeSpdLimData.append(speedLim)
            nodeSpdLimData.append(speedLimType)
            spdLimData.append(nodeSpdLimData)

    spdLimData.append([spdLimData[-1][0] + 0.1, 0, 3])
    # spdLimData = createConvexHull(spdLimData)
    

    return spdLimData


def createConvexHull(spdLimData):
    i = 0
    while i < len(spdLimData) :
        if spdLimData[i][2] == 2:
            istart = i
            while i < len(spdLimData) and spdLimData[i][2] == 2 :
                i += 1
            minSpdLim = min(spdLim[1] for spdLim in spdLimData[istart:i])    
            for spdLim in spdLimData[istart:i]:
                spdLim[1] = minSpdLim
        i += 1
    return spdLimData


def angleDiff(angle1, angle2):
    diff = abs(angle1 - angle2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff
