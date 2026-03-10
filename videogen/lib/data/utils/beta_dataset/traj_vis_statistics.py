import matplotlib.cm as cm


ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds
ColorListLeft = [ (0, 0, 255), (255, 255, 0), (0, 255, 255)]
ColorListRight = [ (255, 0, 255), (255, 0, 0), (0, 255, 0)]



EndEffectorPts = [
    [0, 0, 0, 1],
    [0.1, 0, 0, 1],
    [0, 0.1, 0, 1],
    [0, 0, 0.1, 1]
]

Gripper2EEFCvt = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.23],
    [0, 0, 0, 1]
]

EEF2CamLeft = [0,0,-0.5236]
EEF2CamRight = [0,0,0.5236]
