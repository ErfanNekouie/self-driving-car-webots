# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vehicle_driver_altino controller."""

from vehicle import Driver
from controller import Camera , Keyboard , LED
import time
import cv2
import numpy as np


driver = Driver()
basicTimeStep = int(driver.getBasicTimeStep())

keyboard = Keyboard()  
print("keyboard object called") 
keyboard.enable(basicTimeStep) 
print("keyboard enabled")

camera = Camera('jetcamera')
camera.enable(10)


sensorTimeStep = 4 * basicTimeStep
# speed refers to the speed in km/h at which we want Altino to travel
speed = 0
# angle refers to the angle (from straight ahead) that the wheels
# currently have
angle = 0

# This the Altino's maximum speed
# all Altino controllers should use this maximum value
maxSpeed = 2
minSpeed = -2
maxAngle = 0.65
minAngle = -0.65
# ensure 0 starting speed and wheel angle
driver.setSteeringAngle(angle)
driver.setCruisingSpeed(speed)
# defaults for this controller
right = False
printCounter = 0
i = 0

turn = True
n = 0
while driver.step() != -1:
    # print("speed:",speed)
    # print("Hello World!")
    # if right :
        # i += 1
    # else:
        # i -= 1
        
    # if i > 60 :
        # right = False
    # elif i < -60:
        # right = True
    # if(i < 0):
        # driver.setSteeringAngle(0.3)
    # else:
        # driver.setSteeringAngle(-0.3)
    key = keyboard.getKey()
    
    
    if(key == ord('W')):
       if(speed < maxSpeed):
           speed = speed + 0.01
       
    elif(key == ord('S')):
       if(speed > minSpeed):
           speed = speed - 0.01
    # else:
        # if(speed > 0):
            # speed -= 0.01
        # else:
            # speed += 0.01
    if(key == ord('Q')):
        speed = 0      
            

    if(key == ord('W') + ord('D') or key == ord('D')):
        if(angle < maxAngle):
           angle = angle + 0.05
           
    elif(key == ord('S') + ord('A') or key == ord('A')):
        if(angle > minAngle):
            angle = angle - 0.05
            
    else:
        if(angle > 0.1):
            angle -= 0.1
        elif(angle < -0.1):
            angle += 0.1
        else:
            angle = 0
    
    n += 1
    img = camera.getImage()
    # image = np.frombuffer(img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    
    # cv2.imshow("frome" , image)
    
    # cv2.waitKey(1)
    
    
    driver.setSteeringAngle(angle)
    driver.setCruisingSpeed(speed)
    
writer.release()
cv2.destroyAllWindows()
