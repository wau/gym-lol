import math
import random
import numpy as np
import scipy
import urllib.request
import pyautogui
import keyboard
import h5py
import threading
from threading import Thread, Lock
from PIL import Image
import PIL.ImageOps
import time
#ocr
import cv2
import pytesseract

#screencapture
import os
import win32gui
import win32ui, win32con
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'


############################################################
#  Bounding Boxes
############################################################


class BoundingBox():
    """A bounding box object for ocr.

    """
    def __init__(self, x,y,width,height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        return

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

############################################################
#  OCR Google's optical character recognition
############################################################



class Ocr():
    """Compute strings using ocr of bounding boxes.
    boundingboxes: [[x,y,width,height],[x,y,width,height],...]

    Returns: string of all ocr aqcuired text.
    """
    def __init__(self, boundingboxes):
        self.boundingboxes = boundingboxes
        return

    def capturetext(self, image):
        text = []
        for box in self.boundingboxes:
            im = Image.new('RGB', (600, 300), (255, 255, 255))
            roi = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            img = Image.fromarray(roi, 'RGB')
            img = img.convert('L').resize([8 * _ for _ in img.size], Image.BICUBIC)
            img.show() 
            img = img.point(lambda x: 255 if x<110 else 0, '1')
            im.paste(img, (50,50))
            im.show()
            text = pytesseract.image_to_string(im)
            print(text)
        return

############################################################
#  Input Record Windows 
############################################################

class InputRecord():

    def __init__(self, save_path):
        self.recording_array = []
        self.image_array = []
        self.recording = False
        self.sv = ScreenCapture()
        self.save_path = save_path
        self.recording_session = len(os.listdir(save_path))
        self.incremental_array = [0] * 9
        return

    def begin_recording(self):
        start_time = time.time()
        self.recording = True


        self.sv.GetHWND('League of Legends (TM) Client')
        bounds = self.sv.getScreenBounds()
        self.sv.Start()
        #100ms array
        keyboard.add_hotkey('Q', self.addKey, [0], suppress=False)
        keyboard.add_hotkey('W', self.addKey, [1], suppress=False)
        keyboard.add_hotkey('E', self.addKey, [2], suppress=False)
        keyboard.add_hotkey('T', self.addKey, [3], suppress=False)
        keyboard.add_hotkey('D', self.addKey, [4], suppress=False)
        keyboard.add_hotkey('F', self.addKey, [5], suppress=False)
        keyboard.add_hotkey('R', self.addKey, [6], suppress=False) # mouse click substitute
        # 7 mouse x
        # 8 mouse y

        # start at 100ms
        while self.recording == True:
            if(time.time() - start_time >= 0.100): #ms
                start_time = time.time()
                currentMouseX, currentMouseY = pyautogui.position() 
                self.incremental_array = [0,0,0,0,0,0,0, currentMouseX-bounds[0], currentMouseY-bounds[2]]    
                #record screen too
                img = self.sv.GetScreen() #lol using wrong method
                self.image_array.append(img)                      
                self.recording_array.append(self.incremental_array)
                #reset
                img = None   
        return

    def addKey(self, keynum):
        self.incremental_array[keynum] = 1
        return

    def stop_recording(self):
        print('Stopping recording session....')
        self.sv.Stop()
        self.recording = False
        keyboard.unhook_all()
        #save both arrays as h5py?
        with h5py.File(self.save_path+'/recording_{0}.h5'.format(self.recording_session), 'w') as hf:
            hf.create_dataset("Inputs",  data=self.recording_array)
            hf.create_dataset("Images",  data=self.image_array)
        print(self.recording_array)
        print(len(self.image_array))
        return



############################################################
#  Screen Capture
############################################################

#Asynchronously captures screens of a window. Provides functions for accessing
#the captured screen.
class ScreenCapture:
 
    def __init__(self):
        self.mut = Lock()
        self.hwnd = None
        self.its = None         #Time stamp of last image 
        self.i0 = None          #i0 is the latest image; 
        self.i1 = None          #i1 is used as a temporary variable
        self.cl = False         #Continue looping flag
        #Left, Top, Right, and bottom of the screen window
        self.l, self.t, self.r, self.b = 0, 0, 0, 0
        #Border on left and top to remove
        self.bl, self.bt, self.br, self.bb = 13, 31, 13, 2

    #Begins recording images of the screen
    def Start(self):
        #if self.hwnd is None:
        #    return False
        self.cl = True
        thrd = Thread(target = self.ScreenUpdateT)
        thrd.start()
        return True
        
    #Stop the async thread that is capturing images
    def Stop(self):
        self.cl = False
        
    #Thread used to capture images of screen
    def ScreenUpdateT(self):
        #Keep updating screen until terminating
        while self.cl:
            #t1 = time.time()
            self.i1 = self.GetScreenImg()
            #print('Elapsed: ' + str(time.time() - t1))
            self.mut.acquire()
            self.i0 = self.i1               #Update the latest image in a thread safe way
            self.its = time.time()
            self.mut.release()

    #Gets handle of window to view
    #wname:         Title of window to find
    #Return:        True on success; False on failure
    def GetHWND(self, wname):
        self.hwnd = win32gui.FindWindow(None, wname)
        if self.hwnd == 0:
            self.hwnd = None
            return False
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        return True
         
    #Get's the latest image of the window
    def GetScreen(self):
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.i0
        self.mut.release()
        return s
         
    #Get's the latest image of the window along with timestamp
    def GetScreenWithTime(self):
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.i0
        t = self.its
        self.mut.release()
        return s, t
         
    #Gets the screen of the window referenced by self.hwnd
    def GetScreenImg(self):
        if self.hwnd is None:
            raise Exception("HWND is none. HWND not called or invalid window name provided.")
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        #Remove border around window (8 pixels on each side)
        #Remove 4 extra pixels from left and right 16 + 8 = 24
        w = self.r - self.l - self.br - self.bl
        #Remove border on top and bottom (31 on top 8 on bottom)
        #Remove 12 extra pixels from bottom 39 + 12 = 51
        h = self.b - self.t - self.bt - self.bb
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        #First 2 tuples are top-left and bottom-right of destination
        #Third tuple is the start position in source
        cDC.BitBlt((0,0), (w, h), dcObj, (self.bl, self.bt), win32con.SRCCOPY)
        bmInfo = dataBitMap.GetInfo()
        im = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype = np.uint8)
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        #Bitmap has 4 channels like: BGRA. Discard Alpha and flip order to RGB
        #For 1920x1080 images:
        #Remove 12 pixels from bottom + border
        #Remove 4 pixels from left and right + border
        return im.reshape(bmInfo['bmHeight'], bmInfo['bmWidth'], 4)[:, :, -2::-1]

    # returns the left, right, top, and bottom coordinates of the window
    def getScreenBounds(self):
        return [self.l, self.r, self.t, self.b]

