traffic_light = 1
flag = 0
flag1 = 0
h = 1
if (h<20 and flag == 0 and flag1 == 0):
    speed = 0
    flag = 1
    time.sleep(0.3)
elif (traffic_light == 1 and flag == 1 and flag1 == 0):
    speed = 0.5
    angle = 0.35
    flag1 = 1
    #delay
    time.sleep(2)
    #---------
elif(traffic_light == 0 and flag == 1 and flag1 == 0):
    speed = 0
    angle = 0
    #delay
    time.sleep(0.4)
else:
    
