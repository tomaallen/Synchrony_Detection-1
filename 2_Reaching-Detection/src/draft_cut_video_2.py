import cv2
from datetime import datetime

filename="data/c003/C0004.mp4"   #filename
cap = cv2.VideoCapture(filename) #Read Frame
fps=cap.get(cv2.CAP_PROP_FPS)    #Extract the frame per second (fps)

height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #width

origin="00:00:00"          #the origin
start="00:57:00"           #specify start time in hh:mm:ss 0*3600+57*60, 0*3600+59*60
end = "00:59:00"           #specify end time in hh:mm:ss

origintime=datetime.strptime(origin,"%H:%M:%S") #origin 
starttime=datetime.strptime(start,"%H:%M:%S")  #start time
endtime=datetime.strptime(end,"%H:%M:%S")      #end time

startframe=fps*(starttime-origintime).total_seconds()  #get the start frame
endframe=fps*(endtime-origintime).total_seconds()  #get the end frame

#video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out1 = cv2.VideoWriter("data/c003/C0004_stt.avi",fourcc, fps, (width,height))

counter =1 #set counter
while(cap.isOpened()):           #while the cap is open

    ret, frame = cap.read()       #read frame
    if frame is None:             #if frame is None
        break  
    
    frame=cv2.resize(frame, (width,height))  #resize the frame
    if counter>=startframe and counter<=endframe:  #check for range of output
        out1.write(frame)  #output 

    # cv2.imshow("Frame", frame)  #display frame
    key = cv2.waitKey(1) & 0xFF

    counter+=1  #increase counter
    if counter%500 == 0:
        print(counter)
    
#release the output and cap  
out1.release()
cv2.destroyAllWindows()