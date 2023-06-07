from tkinter import *
import customtkinter
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pygame

class DigitRecogniser(object):
    
    def __init__(self):
        self.root = customtkinter.CTk()
        self.root.title("Digit Recogniser")
        self.root.resizable(False,False)
        self.c = Canvas(self.root, bg='white', width=400, height=400)
        self.c.pack()
        self.button = customtkinter.CTkButton(self.c, text=None, fg_color="white", hover_color="white", width=30, height=30,
                                              image=ImageTk.PhotoImage(Image.open("icons/record.png").resize((40,40), Image.ANTIALIAS)), command=self.guess_digit)
        self.button_window = self.c.create_window(200, 350, anchor=CENTER, window=self.button)
        self.CNN = tf.keras.models.load_model("CNN_digit_recogniser_model")
        self.setup()
        self.setup_audio()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 8
        self.colour = "black"
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
    
    def setup_audio(self):
        pygame.init()
        self.nine = pygame.mixer.Sound("audio/9.wav")
        self.eight = pygame.mixer.Sound("audio/8.wav")
        self.seven = pygame.mixer.Sound("audio/7.wav")
        self.six = pygame.mixer.Sound("audio/6.wav")
        self.five = pygame.mixer.Sound("audio/5.wav")
        self.four = pygame.mixer.Sound("audio/4.wav")
        self.three = pygame.mixer.Sound("audio/3.wav")
        self.two = pygame.mixer.Sound("audio/2.wav")
        self.one = pygame.mixer.Sound("audio/1.wav")
        self.zero = pygame.mixer.Sound("audio/0.wav")

    def paint(self, event):
        if self.old_x and self.old_y:
            self.line = self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.colour,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def save_digit(self):
        x=self.c.winfo_rootx()+self.c.winfo_x()
        y=self.c.winfo_rooty()+self.c.winfo_y()
        x1=x+self.c.winfo_width()
        y1=y+self.c.winfo_height()
        image = ImageGrab.grab().crop((x,y,x1,y1-100)).resize((28,28), Image.HAMMING)
        return image
   
    def get_pixels(self, image):
        img = image.convert('L') 
        WIDTH, HEIGHT = img.size
        data = list(img.getdata()) #
        image_pixels = np.array([data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)])  
        image_pixels = np.abs(image_pixels-255)/255
        return image_pixels
   
    def guess_digit(self):
        image = self.save_digit()
        self.c.delete(ALL)
        self.button = customtkinter.CTkButton(self.c, text=None, fg_color="white", hover_color="white", width=30, height=30,
                                              image=ImageTk.PhotoImage(Image.open("icons/record.png").resize((40,40), Image.ANTIALIAS)), command=self.guess_digit)
        self.button_window = self.c.create_window(200, 350, anchor=CENTER, window=self.button)
        image_pixels = self.get_pixels(image)
        prediction = self.CNN_predict(image_pixels)
        self.play_voice(prediction)
        # plt.imshow(image_pixels)
        # plt.show()

    def CNN_predict(self, input):
        cor_input = input.reshape(-1,28,28,1)
        pred = self.CNN.predict(cor_input)
        return np.argmax(pred)
    
    def play_voice(self, digit_prediction):
        match digit_prediction:
            case 9:
                self.nine.play()
            case 8:
                self.eight.play()
            case 7:
                self.seven.play()
            case 6:
                self.six.play()
            case 5:
                self.five.play()
            case 4:
                self.four.play()
            case 3:
                self.three.play()
            case 2:
                self.two.play()
            case 1:
                self.one.play()
            case 0:
                self.zero.play()
                
if __name__ == '__main__':
    DigitRecogniser()