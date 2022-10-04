import tkinter as tk  
from tkinter import filedialog as fd
from PIL import ImageTk, Image
from RetrieverSSA import RetrieverSSA
from RetrieverVGG import RetrieverVGG
import time
window = tk.Tk()
window.title="SSA Image Retrieval" 

chooser = tk.Label(text="Please choose dataset:", width=20, height=3)
choosermethod = tk.Label(text="Please choose method:", width=20, height=3)
image_label=tk.Label(text="Selected Image",width=30,height=30,fg="blue",bg="white")
out_frame=tk.Frame(master=window) 

DataSet=""
Method=""
FileName=""
AllResult=[]
def choosecars():
    global dataset
    dataset="cars"
    chooser["text"]="Dataset: cars" 
    return

def choosebirds():
    global dataset
    dataset="cubs"
    chooser["text"]="Dataset: birds " 
    return

def choosedogs(): 
    global dataset
    dataset='dogs'
    chooser["text"]="Dataset: dogs " 
    return

def choosSSAMul():
    global Method
    Method="mul"
    choosermethod["text"]="Method: SSA Mul"
    return
def choosSSAPlus():
    global Method
    Method="plus"
    choosermethod["text"]="Method: SSA Plus"
    return

def clear_result():
    global out_frame
    global AllResult
    AllResult=[] 
    for child in out_frame.winfo_children():
        child.destroy()
    out_frame=tk.Frame(master=window,width=100,height=50) 
    out_frame.grid(row=0,rowspan=5,column=3)

    return
def chooseImage():
    global filename
    filename = fd.askopenfilename() 
    img = ImageTk.PhotoImage(Image.open(filename)) 
    ximg,yimg=size_of_view(img.width(),img.height())
    img=ImageTk.PhotoImage(Image.open(filename).resize((ximg,yimg)))
    image_label.config(image=img,width=img.width(),height=img.height()).pack()
    image_label.pack()
    image_label.grid(row=5, column=0, columnspan=2)
    return

def process():
    clear_result()
    button_process.config(text="Processing...")
    if dataset==0:
        tk.messagebox.showerror("choose dataset", "choose dataset")
        button_process.config(text="Retrieve")
        return
    if Method=="":
        tk.messagebox.showerror("choose method","choose method")
        button_process.config(text="Retrieve")
        return
    if filename=="":
        tk.messagebox.showerror("choose input file","choose input file")
        button_process.config(text="Retrieve")
        return 

    show_result(RetrieverSSA(filename,dataset,Method))
    button_process.config(text="Retrieve")
    return
    

def size_of_view(width, height):
    x,y=0,0 
    # if width>height:
    x=200
    y=(height/width)*200
    # else:
    y=100
    x=(width/height)*100
    return int(x),int(y)

def show_result(image_array):  
    row=0
    real_row=0
    column=0
    AllResult.clear()
    for img in image_array:
        tkimg=ImageTk.PhotoImage(Image.open(str(img.replace(" ",""))))
        ximg,yimg=size_of_view(tkimg.width(),tkimg.height())
        tkimg=ImageTk.PhotoImage(Image.open(str(img).replace(" ","")).resize((ximg,yimg))) 
        img_out_label=tk.Label(master=out_frame,width=ximg,height=yimg )
        img_out_label.configure(image=tkimg)
        img_out_label.image=tkimg 
        img_out_label.grid(row=row,column=column)   
        # row=row+1
        # label_of_title.grid(row=row,column=column)
        AllResult.append(img_out_label)
        if row==3:
            column=column+1
            row=0
            real_row=real_row+1
        else:
            row=row+1
    btnclear=tk.Button(master=out_frame,width=10,height=2,text="clear", fg="blue",command=clear_result)
    btnclear.grid(row=real_row+1,column=0)
    print('done')
    time.sleep(1) 
    return
 
buttondogs = tk.Button(
    text="dogs",
    width=10,
    height=2,
    bg="white",
    fg="blue",command= choosedogs
) 
buttoncars=tk.Button(
    text="cars",
    width=10,
    height=2,
    bg="white",
    fg="blue",
    command=choosecars 
)
buttonbirds=tk.Button(
    text="birds",
    width=10,
    height=2,
    bg="white",
    fg="blue",
    command=choosebirds

)

button_ssa = tk.Button(
    text="SSA Mul",
    width=10,
    height=2, 
    bg="gray",
    fg="blue",  command=choosSSAMul
)
button_vgg = tk.Button(
    text="SSA Plus",
    width=10,
    height=2,
    bg="gray",
    fg="blue", command= choosSSAPlus
) 
button_choose_input=tk.Button(
    text="Choose Input Image",
    width=15, height=2 , bg="gray", fg="green",command=chooseImage
)
 
button_process=tk.Button(
    text="Retrieve",bg="black", fg="green", width=20, height=2, command= process
) 
 
out_frame.grid(row=0,rowspan=5,column=3)
chooser.grid(row=0,column=0,columnspan=2)
buttoncars.grid(row=1,column=0)
buttonbirds.grid(row=1,column=1)
buttondogs.grid(row=1,column=2)
choosermethod.grid(row=2,column=0,columnspan=2)
button_ssa.grid(row=3,column=1)
button_vgg.grid(row=3,column=0)
button_choose_input.grid(row=4,column=0,columnspan=2)
image_label.grid(row=5, column=0, columnspan=2)
button_process.grid(row=6,column=0,columnspan=2) 
window.mainloop()

