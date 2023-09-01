#import cv2, os, csv, shutil, numpy as np, datetime, time ,pandas as pd
from tkinter import *
from PIL import Image
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as font


window=tk.Tk()
window.title('face recognizer')
window.geometry('1280x720')
dialog_title= 'QUIT'
dialog_text='Are you sure?'
window.configure(background='blue')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message=tk.Label(window,text='Face Recognition Based Attendance Management System', bg='green' , fg='white',width=50, height=3 ,font=('times',30 ,'italic bold underline'))
message.place(x=100, y=20)

lb1=tk.Label(window, text='Enter ID',width=20, height=2, fg='red', bg='yellow',font=("times",15,'bold'))
lb1.place(x=200, y=200)
txt=tk.Entry(window, width=20,bg='yellow', fg='black', font=('times',25, 'bold'))
txt.place(x=550, y=210)

lb2=tk.Label(window, text='Name',width=20, fg='red', bg='yellow', height=2,font=('times', 15, 'bold'))
lb2.place(x=200, y=300)
txt2=tk.Entry(window ,width=20,bg='yellow', fg='black', font=('times',25, 'bold'))
txt2.place(x=550, y=310)

lb3=tk.Label(window, text='Notification',width=20, fg='red', bg='yellow', height=2,font=('times', 15, 'bold underline'))
lb3.place(x=200, y=400)

message=tk.Label(window, text='',width=30, fg='black', bg='yellow', height=2,activebackground='yellow' ,font=('times', 15, 'bold'))
message.place(x=550, y=400)

lb3=tk.Label(window, text='Attendance',width=20, fg='red', bg='yellow', height=2,font=('times', 15, 'bold underline'))
lb3.place(x=200, y=620)

message2=tk.Label(window, text='',width=30, fg='black', bg='yellow', height=2, activebackground='green', font=('times', 15, 'bold'))
message2.place(x=550, y=620)



def clear():
    txt.delete( 0,'end')
    res=""
    message.configure(text=res)

def clear2():
    txt2.delete( 0,'end')
    res=""
    message.configure(text=res)
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def TakeImage():
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        sampleNum = 0
        while(True):
            ret,img=cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                roi_color = img[y:y+h, x:x+w]
                sampleNum = sampleNum + 1
                cv2.imwrite("dataset/" +name+"."+ Id + "."  +str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('img', img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            elif sampleNum > 50:
                break
        cam.release()
        cv2.destroyAllWindows()

        res="Image Saved for Id:"+ Id+ "name:"+ name
        row=(Id,name)
        with open('studentDetails/stuentdetails.csv', 'w') as csvFile:
            writer=csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if(is_number(Id)):
            res="Enter Alphabetic NAME"
            message.configure(text=res)
        if (name.isalpha()):
            res="Enter numeric"
            message.configure(text=res)

def TrainImages():
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces,Id=getImageandLabels('dataset')
    recognizer.train(faces,np.array(Id))
    recognizer.save("trainer/trainer.yml")
    res="image Trained" #+".".join(str(f) for f in Id)
    message.configure(text=res)

def getImageandLabels(path):
    imagePaths= [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    Ids=[]
    for imagepath in imagePaths:
        pilimage=Image.open(imagepath).convert('L')
        imagenp=np.array(pilimage, 'uint8')
        Id = int(os.path.split(imagepath)[-1].split(".")[1])

        faces.append(imagenp)
        Ids.append(Id)
    return faces, Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    df = pd.read_csv('studentDetails/stuentdetails.csv')
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns = col_names)
    cam=cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0, 0), 2)
          
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if 'Id' in df and (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M,%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timestamp]
            else:
                Id = '1-karanrampal'
                tt = str(Id)
                if (conf > 75):
                    noOfFile = len(os.listdir("image1-karanrampal")) + 1
                    cv2.imwrite("image1-karanrampal/image" + str(noOfFile) + ".jpg", img[y:y + h, x:x + w])
                cv2.putText(img,str(tt),(x,y+h),font, 1, (225,225,255),2)
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        cv2.imshow('img', img)
        if (cv2.waitKey(1)==ord('q')):
            break
    ts=time.time()
    date=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timestamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timestamp.split(":")
    filename='attendance/attendance_'+date+"_" +Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(filename,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text=res)

clearbutton=tk.Button(window, text='clear', command=clear, width=20, fg='red', bg='yellow', height=2, activebackground='red', font=('times', 15, 'bold'))
clearbutton.place(x=950, y=210)

clearbutton2=tk.Button(window, text='clear', command=clear2, width=20, fg='red', bg='yellow', height=2, activebackground='red', font=('times', 15, 'bold'))
clearbutton2.place(x=950, y=310)

takeimg=tk.Button(window, text='Take image', command=TakeImage , width=20, fg='red', bg='yellow', height=3, activebackground='red', font=('times', 15, 'bold'))
takeimg.place(x=90, y=500)

trainimg=tk.Button(window, text='Train Image', command=TrainImages , width=20, fg='red', bg='yellow', height=3, activebackground='red', font=('times', 15, 'bold'))
trainimg.place(x=390, y=500)

trackimg=tk.Button(window, text='Track Imges', command=TrackImages , width=20, fg='red', bg='yellow', height=3, activebackground='red', font=('times', 15, 'bold'))
trackimg.place(x=690, y=500)

quitwindow=tk.Button(window, text='Quit', command=window.destroy, width=20, fg='red', bg='yellow', height=3, activebackground='red', font=('times', 15, 'bold'))
quitwindow.place(x=990, y=500)


window.mainloop()






