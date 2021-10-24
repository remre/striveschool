
import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('solidWhiteCurve.jpg')


def cannyfunc(image):
  #copy the image and change it to Gray
  
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  #apply gaussian blur
  gaussian = cv2.GaussianBlur(image,(5,5),0)
  #apply canny edge detection
  cannyedge = cv2.Canny(gaussian, 150,155)
  return cannyedge
#plt.figure(figsize = (15,15))
canny = cannyfunc(img)
#plt.imshow(canny, cmap= 'gray')

#now specify the region of interest


def mcoordinates(image, line_parameters):
  slope, intercept = line_parameters
  y1 =  image.shape[0]
  y2 = int(y1*(3.1/5))
  x1 = (y1-intercept)/slope
  x2 = (y2-intercept)/slope
  return np.array([x1,y1,x2,y2])
def average_slope(image,lines):
  left_fit = []
  right_fit = []

  for line in lines:
    x1,y1,x2,y2 = line.reshape(4)
    parameter = np.polyfit((x1,x2),(y1,y2),1)
    slope = parameter[0]
    intercept = parameter[1]
    if slope < 0:
      left_fit.append((slope, intercept))
    else:
      right_fit.append((slope,intercept))
  left_fit_average = np.average(left_fit, axis =0)
  right_fit_average = np.average(right_fit, axis = 0)
  left_line = mcoordinates(image,left_fit_average)
  right_line = mcoordinates(image,right_fit_average)
  arrray = np.array([left_line, right_line])
  int_array = arrray.astype('int')
  return int_array

def display_lines(image,lines):
  line_image = np.zeros_like(image)
  if lines is not None:
    for x1,y1,x2,y2  in lines:
      cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)
  return line_image

def roi(image):
  #we will enclose the triangle region
  height, weight = image.shape[0],image.shape[1]
  polygons = np.array([
                       [(140,height),(900,height),(500,300)]
                       ])#in order bottom left, bottom right, top point of the roi


  #mask the area whole black
  mask = np.zeros_like(image)
  

  #fill the polygon are fi
  cv2.fillPoly(mask, polygons,255)

  bitw = cv2.bitwise_and(mask,image)

  
  return bitw

#canny = cannyfunc(img)
#roii = roi(canny)
#lines = cv2.HoughLinesP(roii,2,np.pi/180, 120,np.array([]), minLineLength = 40, maxLineGap=5) #keep rho small but not too small theta in radiant


# averaged_lines1,averaged_lines2 = average_slope(img1,lines)
# averaged_lines  = averaged_lines1,averaged_lines2

# lane_image = display_lines(img1,(averaged_lines))

# combo = cv2.addWeighted(img1,0.7,lane_image,0.8,2)
# cv2.imshow('basdasd',combo)  
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture('solidWhiteRight.mp4')

while(cap.isOpened()):
    _,frame = cap.read()
    canny = cannyfunc(frame)
    roii = roi(canny)
    lines = cv2.HoughLinesP(roii,2,np.pi/180, 120,np.array([]), minLineLength = 40, maxLineGap=5) #keep rho small but not too small theta in radiant

    averaged_lines = average_slope(frame,lines)
   

    lane_image = display_lines(frame,averaged_lines)

    combo = cv2.addWeighted(frame,0.7,lane_image,0.8,2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.imshow('basdasdd',combo)  
    cv2.waitKey(0)

