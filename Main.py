import tkinter as tk
from tkinter import Message ,Text
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.font as font
from tkinter import filedialog
import tkinter.messagebox as tm
from tkinter import ttk
import time
import matplotlib.pyplot as plt
import preprocess as pre
import Training as tr
import predict as pred


fontScale=1
fontColor=(0,0,0)
cond=0

bgcolor="#d7837f"
fgcolor="white"

window = tk.Tk()
window.title("Image Decompression Using CNN")

 
window.geometry('1280x720')
window.configure(background=bgcolor)
#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message1 = tk.Label(window, text="Image Decompression Using CNN" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
message1.place(x=100, y=10)

lbl = tk.Label(window, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl.place(x=10, y=200)

txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=300, y=215)

lbl1 = tk.Label(window, text="Select Image",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl1.place(x=10, y=300)

txt1 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt1.place(x=300, y=315)

lbl2 = tk.Label(window, text="Original Image Size",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl2.place(x=10, y=400)

txt2 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt2.place(x=300, y=415)

lbl3 = tk.Label(window, text="Compressed Image Size",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl3.place(x=10, y=450)

txt3 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt3.place(x=300, y=465)

lbl4 = tk.Label(window, text="Restored Image Size",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl4.place(x=10, y=500)

txt4 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt4.place(x=300, y=515)



def browse():
	path=filedialog.askdirectory()
	print(path)
	txt.delete(0, 'end')
	txt.insert('end',path)
	if path !="":
		print(path)
	else:
		tm.showinfo("Input error", "Select Dataset")	

def browse1():
	path=filedialog.askopenfilename()
	print(path)
	txt1.delete(0, 'end')
	txt1.insert('end',path)
	if path !="":
		print(path)
	else:
		tm.showinfo("Input error", "Select Image")	
	
def clear():
	txt.delete(0, 'end') 
	txt1.delete(0, 'end') 
	txt2.delete(0, 'end') 
	txt3.delete(0, 'end') 
	txt4.delete(0, 'end') 

def preprocess():
	sym=txt.get()
	if sym != "" :
		pre.process(sym)
		tm.showinfo("Input", "Preprocess Successfully Finished")
	else:
		tm.showinfo("Input error", "Select Dataset Path")

def trprocess():
	tr.process()
	tm.showinfo("Input", "Training Successfully Finished")
	
def predictprocess():
	sym=txt1.get()
	txt2.delete(0, 'end') 
	txt3.delete(0, 'end') 
	txt4.delete(0, 'end') 


	if sym != "" :
		s1,s2,s3=pred.process(sym)
		print(s1)
		print(s2)
		print(s3)
		txt2.insert('end',s1+"  Bytes")
		txt3.insert('end',s2+"  Bytes")
		txt4.insert('end',s3+"  Bytes")
		
	else:
		tm.showinfo("Input error", "Select Image")


browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
browse.place(x=650, y=200)

browse = tk.Button(window, text="Browse", command=browse1  ,fg=fgcolor  ,bg=bgcolor  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
browse.place(x=650, y=300)

pre1 = tk.Button(window, text="Preprocess", command=preprocess  ,fg=fgcolor  ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
pre1.place(x=400, y=600)

texta = tk.Button(window, text="Training", command=trprocess  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta.place(x=600, y=600)

texta1 = tk.Button(window, text="Predict", command=predictprocess  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta1.place(x=820, y=600)


quitWindow = tk.Button(window, text="QUIT", command=window.destroy  ,fg=fgcolor ,bg=bgcolor  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1030, y=600)

 
window.mainloop()
