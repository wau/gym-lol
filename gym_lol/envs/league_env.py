import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import pyautogui
import numpy as np
import keyboard
from threading import Thread, Lock
from flask import Flask, request, jsonify

try:
    import pytesseract
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you must install pytesseract on your machine to use this feature.)'".format(e))

try:
    import win32gui
    import win32con
    import win32ui
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you must be running on a Windows machine.)'".format(e))

ALLOWED_ACTIONS = ['Q','W','E','T','D','F','R'] # where R is right click



import logging
logger = logging.getLogger(__name__) 

class LeagueEnv(gym.Env):

    def __init__(self, game='sr', frameskip=(2, 5), repeat_action_probability=0.):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""
        self.sv = ScreenCapture()        
        self.sv.GetHWND('League of Legends (TM) Client')
        self.bounds = self.sv.getScreenBounds()
        self.screen = None
    
    def init(self, client_pool=None, start_minecraft=None,
             continuous_discrete=True, add_noop_command=None,
             max_retries=90, retry_sleep=10, step_sleep=0.100, skip_steps=0, 
             recordDestination=None, obs_resolution=[1024,786],
             recordObservations=None, recordRewards=None,
             recordCommands=None, recordMP4=None):
        self.res_x = obs_resolution[0] # returned resolution, not necessarily game res
        self.res_y = obs_resolution[1]
        self._key_thread()
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.step_sleep = step_sleep
        self.skip_steps = skip_steps
        self.continuous_discrete = continuous_discrete
        self.add_noop_command = add_noop_command
        self.sv.Start()

        #initialize mouse
        currentMouseX, currentMouseY = pyautogui.position() 
        self.current_action = [0,0,0,0,0,0,0,currentMouseX-self.bounds[0], currentMouseY-self.bounds[2]]


        self.episode_over = False
        self.window_dim = obs_resolution

        # TODO: produce observation space dynamically based on requested features

        self.last_image = np.zeros((self.res_y, self.res_x, 3), dtype=np.uint8)


    def _step(self, machine_action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """



        # take the last frame from world state
        image = self._get_video_frame()

        # append it to the list of prveious frames 
    

        reward = self._get_reward(self.current_action, machine_action, 12) #also resets

        episode_over = self.episode_over # not needed

        info = dict()

        return image, reward, episode_over, info

    def _reset(self): 
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        return 

    def shutdown():
        """ Request a server shutdown - currently used by the integration tests to repeatedly create and destroy fresh copies of the server running in a separate thread"""
        f = request.environ.get('werkzeug.server.shutdown')
        f()
        return 'Server shutting down'

    def _take_action(self, keys):
        # length 9 
        send_keys = []
        for i in range(7):
            if(keys[i] >= .5):
                send_keys+=ALLOWED_ACTIONS[i]
        self._move_mouse(keys[7], keys[8])
        pyautogui.press(send_keys)
        print('{0}:{1} Keys:{2}'.format(keys[7],keys[8], send_keys))
        return


    def _get_video_frame(self):
        #record screen
        img = self.sv.GetScreen() 
        if self.res_x!= np.size(img,0):
            img = cv2.resize(img, (self.res_x,self.res_y), interpolation=cv2.INTER_CUBIC) #res is tuple
        return img


    def _get_current_action(self):
        currentMouseX, currentMouseY = pyautogui.position()
        action = self.current_action 
        self.current_action = [0,0,0,0,0,0,0,currentMouseX-self.bounds[0], currentMouseY-self.bounds[2]]
        return action


    def _close(self):
        self.sv.Stop()
        keyboard.unhook_all()

    def _get_reward(self, human_keys, predicted_keys, key_bias):
        '''
        TODO: Improve this algorithm. It is the reward function for your model.
        '''
        #mse between the two 
        total_error = 0
        array_length = len(human_keys)
        for i in array_length:
            if i < 7: #weight key mse properly
                total_error+= (human_keys[i] - predicted_keys[i])**2
            else: # lower mouse mse based on specified key bias
                total_error+= ((human_keys[i] - predicted_keys[i])/key_bias)**2
        return total_error/array_length


    def _move_mouse(self, x, y):
        pyautogui.moveTo(x+self.bounds[0], y+self.bounds[2])
        return

    def _key_thread(self):
        for i in range(len(ALLOWED_ACTIONS)):
            keyboard.add_hotkey(ALLOWED_ACTIONS[i], self._add_key, [i], suppress=False)
        return

    def _add_key(self, keynum):
        self.current_action[keynum] = 1
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


    def start(self):
        # thrd = 
        return

    def record(self):
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


class Controller:

    def __init__(self, model_path):
        self.sv = utils.ScreenCapture()
        self.sv.GetHWND('League of Legends (TM) Client')
        self.bounds = self.sv.getScreenBounds()  
        self.translated_keys = ['Q','W','E','T','D','F','R']  
        self.recording = False  
        self.model = agentmodel.AgentModel()
        self.model.load_model(model_path)   
        return

    def begin_eval(self, duration):
        #capture screen
        self.recording = True
        self.sv.Start()

        image_queue = np.zeros(shape=(4, 382, 502, 3))
        start_time = time.time()
        begin_time = start_time
        while self.recording == True:
             if(time.time() - start_time >= .4): #100 ms delay?
                start_time = time.time()
                img = self.sv.GetScreen() #add capture to front of image queue and drop last image
                #eval 
                img = cv2.resize(img, dsize=(502, 382),interpolation=cv2.INTER_CUBIC) 
                image_queue = np.insert(image_queue, 0, img, axis=0) 
                image_queue = np.delete(image_queue, -1, axis=0) #remove last entry
                keys = self.model.eval(image_queue[np.newaxis,:])
                keys = np.append(keys[0], keys[1], axis=1)
                keys = keys[0]
                self.SendKeys(keys)
                if(duration <= time.time()-begin_time):
                    self.recording = False
                    return
        return
    
    def SendKeys(self, keys):
        # length 9 
        send_keys = []
        for i in range(7):
            if(keys[i] >= .5):
                send_keys+=self.translated_keys[i]
        self.moveMouse(keys[7], keys[8])
        print('{0}:{1}'.format(keys[7],keys[8]))
        pyautogui.press(send_keys)
        print(send_keys)
        return
    
    def moveMouse(self, x, y):
        pyautogui.moveTo(x+self.bounds[0], y+self.bounds[2])
        return