import customtkinter
import os #for file paths
from PIL import Image
from tkinter import filedialog
import cv2
import numpy as np

customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

lower_h = 0
lower_s = 0
lower_v = 0
upper_h = 255
upper_s = 255
upper_v = 255
gaussian = 1
dilate = 1
break_state = 0
objects = 0
resulting_image = None
replace = False
def getTrackbarValues():
    global lower_h
    global lower_s
    global lower_v
    global upper_h
    global upper_s
    global upper_v
    global gaussian
    global dilate

    return [lower_h,lower_s,lower_v,upper_h,upper_s,upper_v,gaussian,dilate]

#this function shall be implemented for calling the gaussian trackbar values and dilation values 
#because their values should always be positive
def isOdd(x):
    if (x%2==0):
        return x+1
    return x

# def detect(input_image):
#if the parameters are passed to the function
def detect(img, hsv_image, l_h, l_s, l_v, u_h, u_s, u_v, g_b, d_k):
    
    #the values for the lower and upper bounds of the image
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound) #this is the binary image?
    
    masked_image = cv2.bitwise_and(img, img, mask=mask) #this gets the binary picture with the objects in white
    
    #next step is to convert the resulting image to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    
    #getting the gaussian blur of the image. But what is the best kernel size?
    blur = cv2.GaussianBlur(gray, (g_b, g_b), 1)
    canny = cv2.Canny(blur, 30, 150, 3)

    dilated = cv2.dilate(canny, (d_k, d_k), iterations=1)
    (cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    
    return [rgb, cnt]


#this function calls the main algorithm, please set up here the code for getting the values of the trackbars
def caller_function(img,image,self):
    global objects
    global resulting_image
    global replace
    #the function should be in a loop to always have the preview of the output image,
    #then if the button is clicked to detect, the while loop shall be broken, and the result shall be printed
    
    #first set up the getting of values from the trackbar
    ranges =  getTrackbarValues()#l_h, l_s, l_v, u_h, u_s, u_v, g_b, d_k
    #lower Hue, lower Saturation, lower value, Upper Hue, Upper Saturation, Upper value, gaussian kernel size, dilation kernel size

    result = detect(img,image, ranges[0], ranges[1], ranges[2], ranges[3], ranges[4], ranges[5], ranges[6], ranges[7]) 
        
        #display the image in the window
        
        #result[0] is the image with contours on
        #result[1] is the number of objects detected
    if(replace):
        self.output_label.destroy()
    cv2.imwrite("objects.jpeg",result[0])
      #dito nag kaka problema
    if (result[0] is not None):

        image = cv2.imread("objects.jpeg")
        h, w, c = img.shape
        pil_image=Image.fromarray(image)
        image = customtkinter.CTkImage(pil_image, size=(w, h))
        self.output_label = customtkinter.CTkLabel(self, text="", image=image)
        self.output_label.grid(row=1, column=1, padx=20, pady=10)
        objects = result[1] 


#reading of image opened in opencv, as soon as an image is read, the algorithm will start
def read_image(path,self):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # img = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
    h, w, c = img.shape
    img=cv2.resize(img, (int(w*0.15),int(h*0.15)))
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    caller_function(img,hsv_image,self)



#GUI PART
class App(customtkinter.CTk):
  def __init__(self):
    super().__init__()
    
    #Setting up of the placeholder image for the start of the algorithm
    image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./")
    self.add_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "add.png")), size=(100, 100))

    # Frame for the application
    self.title("Custom Pollen Detection algorithm")
    self.geometry('{}x{}'.format(1280, 720))

    # Configuring of layout
    self.grid_columnconfigure(1, weight=1)
    self.grid_columnconfigure((2, 3), weight=0)
    self.grid_rowconfigure((0, 1, 2), weight=1)

    # Sidebar frame to place buttons and trackbars
    self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
    self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
    self.sidebar_frame.grid_rowconfigure(4, weight=1)

    # Defining trackbars in the code
    self.options_label = customtkinter.CTkLabel(self.sidebar_frame, text="Settings", font=customtkinter.CTkFont(size=24, weight="bold"))
    self.options_label.grid(row=0, column=0, padx=20, pady=(20, 10))

    # Buttons for the program
    self.add_image_button = customtkinter.CTkButton(self.sidebar_frame, command=self.add_image_func)
    self.add_image_button.grid(row=1, column=0, padx=20, pady=10)
    self.remove_image_button = customtkinter.CTkButton(self.sidebar_frame, command=self.remove_image_func)
    self.remove_image_button.grid(row=2, column=0, padx=20, pady=10)
    self.examine_image_button = customtkinter.CTkButton(self.sidebar_frame, command=self.show_results)
    self.examine_image_button.grid(row=3, column=0, padx=20, pady=10)

    # Trackbars for the ranges of Hue, Saturation, Value, Gaussian Kernel Size, Dilation Kernel Size

    # For Hue
    self.hue_label = customtkinter.CTkLabel(self.sidebar_frame, text="Hue range: 0-255", font=customtkinter.CTkFont(size=12))
    self.hue_label.grid(row=5, column=0, padx=20)
    self.lower_h_slider = customtkinter.CTkSlider(self.sidebar_frame, from_=0, to=255, command=lambda value:self.trackbar_set("lower_h",value))
    self.lower_h_slider.set(0)
    self.lower_h_slider.grid(row=6, column=0, padx=20)
    self.lower_h_slider = customtkinter.CTkSlider(self.sidebar_frame, from_=0, to=255, command=lambda value:self.trackbar_set("upper_h",value))
    self.lower_h_slider.set(255)
    self.lower_h_slider.grid(row=7, column=0, padx=20)
    
    # For Saturation
    self.saturation_label = customtkinter.CTkLabel(self.sidebar_frame, text="Saturation range: 0-255", font=customtkinter.CTkFont(size=12))
    self.saturation_label.grid(row=8, column=0, padx=20)
    self.lower_s_slider = customtkinter.CTkSlider(self.sidebar_frame, from_=0, to=255, command=lambda value:self.trackbar_set("lower_s",value))
    self.lower_s_slider.set(0)
    self.lower_s_slider.grid(row=9, column=0, padx=20)
    self.lower_s_slider = customtkinter.CTkSlider(self.sidebar_frame, from_=0, to=255, command=lambda value:self.trackbar_set("upper_s",value))
    self.lower_s_slider.set(255)
    self.lower_s_slider.grid(row=10, column=0, padx=20)

    # For Value
    self.value_label = customtkinter.CTkLabel(self.sidebar_frame, text="Value range: 0-255", font=customtkinter.CTkFont(size=12))
    self.value_label.grid(row=11, column=0, padx=20)
    self.lower_v_slider = customtkinter.CTkSlider(self.sidebar_frame, from_=0, to=255, command=lambda value:self.trackbar_set("lower_v",value))
    self.lower_v_slider.set(0)
    self.lower_v_slider.grid(row=12, column=0, padx=20)
    self.lower_v_slider = customtkinter.CTkSlider(self.sidebar_frame, from_=0, to=255, command=lambda value:self.trackbar_set("upper_v",value))
    self.lower_v_slider.set(255)
    self.lower_v_slider.grid(row=13, column=0, padx=20)

    # For the gaussian Kernel Size
    self.gaussian_label = customtkinter.CTkLabel(self.sidebar_frame, text="Gaussian Kernel: 1", font=customtkinter.CTkFont(size=12))
    self.gaussian_label.grid(row=14, column=0, padx=20)
    self.gaussian_slider = customtkinter.CTkSlider(self.sidebar_frame, from_=1, to=21, command=lambda value:self.trackbar_set("gaussian",value))
    self.gaussian_slider.set(1)
    self.gaussian_slider.grid(row=15, column=0, padx=20)

    # For the Dilation Kernel Size
    self.dilation_label = customtkinter.CTkLabel(self.sidebar_frame, text="Dilation Kernel: 1", font=customtkinter.CTkFont(size=12))
    self.dilation_label.grid(row=16, column=0, padx=20)
    self.dilation_slider = customtkinter.CTkSlider(self.sidebar_frame, from_=1, to=21, command=lambda value:self.trackbar_set("dilation",value))
    self.dilation_slider.set(1)
    self.dilation_slider.grid(row=17, column=0, padx=20)


    #This will show the result when the detect button is clicked
    self.objectser_label = customtkinter.CTkLabel(self.sidebar_frame, text="Number of pollens found:", anchor="w")
    self.objectser_label.grid(row=19, column=0, padx=20)
    self.object_number = customtkinter.CTkButton(self.sidebar_frame)
    self.object_number.grid(row=20, column=0, padx=20,pady=(0, 10))


    # Default configuration of buttons at the start of the program
    self.add_image_button.configure(state="enabled", text="Add new image")
    self.remove_image_button.configure(state="disabled", text="Remove image")
    self.examine_image_button.configure(state="disabled", text="Detect Pollens")
    self.object_number.configure(state="normal", text="")


    # Placeholder image at the start of the algorithm
    self.image_label = customtkinter.CTkLabel(self, text="", image=self.add_image)
    self.image_label.grid(row=0, column=1, rowspan = 10, padx=20, pady=10)
    self.output_label = customtkinter.CTkLabel(self, text="", image=self.add_image)
    self.output_label.grid(row=1, column=1, rowspan = 10, padx=20, pady=10)


    # This shall be the filename of the image
    self.filename = ""

    # Reseting the state of the trackbars and their values
  def trackbar_reset(self):
        self.lower_h_slider.set(0)
        self.lower_s_slider.set(0)
        self.lower_v_slider.set(0)
        self.lower_h_slider.set(255)
        self.lower_s_slider.set(255)
        self.lower_v_slider.set(255)
        self.gaussian_slider.set(1)
        self.dilation_slider.set(1)
        self.object_number.configure(state="normal", text="")
        self.hue_label.configure(state="normal", text="Hue Range: 0-255")
        self.saturation_label.configure(state="normal", text="Saturation Range: 0-255")
        self.value_label.configure(state="normal", text="Value Range: 0-255")
        self.gaussian_label.configure(state="normal", text="Gaussian Range: 1")
        self.dilation_label.configure(state="normal", text="Dilation Range: 1")
        self.remove_image_button.configure(state="disabled", text="Remove image")
        self.examine_image_button.configure(state="disabled", text="Detect Pollens")

    # Updating of the values of the trackbars
  def trackbar_set(self, name, value):
        global lower_h
        global lower_s
        global lower_v
        global upper_h
        global upper_s
        global upper_v
        global gaussian
        global dilate
        value = int(value)

        print(name,": ",value)
        if (name == "lower_h"):
            lower_h = value
            text = "Hue range: "+str(lower_h)+"-"+str(upper_h)
            self.hue_label.configure(state="normal", text=text)
        elif (name == "lower_s"):
            lower_s = value
            text = "Saturation range: "+str(lower_s)+"-"+str(upper_s)
            self.saturation_label.configure(state="normal", text=text)
        elif (name == "lower_v"):
            lower_v = value
            text = "Value range: "+str(lower_v)+"-"+str(upper_v)
            self.value_label.configure(state="normal", text=text)
        elif (name == "upper_h"):
            upper_h = value
            text = "Hue range: "+str(lower_h)+"-"+str(upper_h)
            self.hue_label.configure(state="normal", text=text)
        elif (name == "upper_s"):
            upper_s = value
            text = "Saturation range: "+str(lower_s)+"-"+str(upper_s)
            self.saturation_label.configure(state="normal", text=text)
        elif (name == "upper_v"):
            upper_v = value
            text = "Value range: "+str(lower_v)+"-"+str(upper_v)
            self.value_label.configure(state="normal", text=text)
        elif (name == "gaussian"):
            gaussian = isOdd(value)
            text = "Gaussian Kernel: "+str(gaussian)
            self.gaussian_label.configure(state="normal", text=text)
        elif (name == "dilation"):
            dilate = isOdd(value)
            text = "Dilation Kernel: "+str(dilate)
            self.dilation_label.configure(state="normal", text=text)
        if (self.filename != ""):
            read_image(self.filename,self)

    #Importing of the image and first call of the image processing function
  def add_image_func(self):
        global replace
        
        self.trackbar_reset()
        self.filename = filedialog.askopenfilename(initialdir="./", title="Please select an appropriate image", filetypes=(("JPG files", "*.jpg"),("JPEG files", "*.jpeg"), ("PNG files", "*.png")))
        if (self.filename != ""):
            self.image_label.destroy()
            replace = True

        image = cv2.imread(self.filename)
        h, w, c = image.shape
        pil_image=Image.fromarray(image)
        self.pollen_image = customtkinter.CTkImage(pil_image, size=(w*.15, h*.15))
        self.image_label = customtkinter.CTkLabel(self, text="", image=self.pollen_image)
        self.image_label.grid(row=0, column=1, padx=20, pady=10)

        read_image(self.filename,self)
        self.remove_image_button.configure(state="enabled", text="Remove image")
        self.examine_image_button.configure(state="enabled", text="Detect Pollens")
        
    #removing of the image the the previous results of the algorithm
  def remove_image_func(self):
        global objects
        global resulting_image
        global replace
        objects = 0
        resulting_image = None

        replace = False
        self.image_label.destroy()
        self.output_label.destroy()
        self.output_label = customtkinter.CTkLabel(self, text="", image=self.add_image)
        self.output_label.grid(row=1, column=1, rowspan = 10, padx=20, pady=10)
        self.filename = ""
        self.image_label = customtkinter.CTkLabel(self, text="", image=self.add_image)
        self.image_label.grid(row=0, column=1, rowspan = 10, padx=20, pady=10)
        self.trackbar_reset()

    #showing of results of the detection
  def show_results(self):
        global objects
        if(self.filename != ""):
            self.object_number.configure(state="enabled", text=str(len(objects)))


if __name__ == "__main__":
    app = App()
    app.mainloop()