import win32gui
import win32con, win32ui
import win32api
import config
import pyautogui
import gym
from gym import error, spaces
from gym import utils
import pygame
import numpy as np
import keyboard
import time
import os
import cv2
import pytesseract
from dataclasses import dataclass
from threading import Thread, Lock
import subprocess
from utilz import utils_chroma

#store key press with timestamp
@dataclass
class ActionKey:
    key: chr
    time: int

colors =	{
  "fuschia": (255, 0, 128),  # Transparency color
  "dark_red": (100, 0, 0),
  "bright_green": (0,230,0)
}



ACTIONS = ['q', 'w', 'e', 'r', 'd', 'f']
COOLDOWN = []
 
class LeagueEnv(gym.Env):

    
    def __init__(self, username, password, champion, gamemode, position, ban,
             menu_res=[1080,1920], obs_resolution=[720,1280],
             recordObservations=None, recordRewards=None,
             recordCommands=None, recordMP4=None):



        self.username = username
        self.password = password
        self.champion = champion
        self.gamemode = gamemode
        self.position = position
        self.ban = ban
        self.chroma = utils_chroma.KeyboardChroma()
        self.chroma.startup_animation()

        self._setup_env()

        self.sv = ScreenCapture()        
        self.sv.GetHWND('League of Legends (TM) Client')
        self.bounds = self.sv.getScreenBounds()
        self.screen = None

           

        self.res_x = obs_resolution[0] # returned resolution, not necessarily game res
        self.res_y = obs_resolution[1]

        self._key_thread() #add all keys 

        self.sv.Start()


        self.action_space = spaces.Box(0,1, shape=(9,))
        self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.res_y, self.res_x, 3))

        #initialize action array
        self.current_action = [0,0,0,0,0,0,0]



        self.episode_over = False
        self.window_dim = obs_resolution

        # TODO: produce observation space dynamically based on requested features

        self.last_image = np.zeros((self.res_y, self.res_x, 3), dtype=np.uint8)

    def _setup_env(self):
        #TODO: autolaunch app and get into game
            #launch app on rest call
        gamemodes = {
            'blind': 725,
            'draft': 825,
            'solo': 870,
            'flex': 925
        }
        positions = {
            'jungle': (625,620),
            'mid': (850,570)
        }

        subprocess.Popen(['C:\\Riot Games\\League of Legends\\LeagueClient.exe'])
        time.sleep(8)

        self.sv = ScreenCapture()        
        self.sv.GetHWND('League of Legends')
        l, r, t, b = self.sv.getScreenBounds()

        pyautogui.click(x=l+1740,y=t+290) #username
        pyautogui.typewrite(self.username)
        pyautogui.click(x=l+1740,y=t+380) #password
        pyautogui.typewrite(self.password)
        pyautogui.click(x=l+1740,y=t+840) #click on login

        time.sleep(30)

        pyautogui.click(x=l+150,y=t+50) #click on play
        time.sleep(4)
        pyautogui.click(x=l+150,y=t+gamemodes.get(self.gamemode)) #click on gamemode
        pyautogui.click(x=l+800,y=t+1030) #play
        time.sleep(4)

        for i in range(2):
            pyautogui.click(x=l+725+i*125,y=t+720)
            pyautogui.click(x=l+positions.get(self.position[i])[0],
                y=t+positions.get(self.position[i])[1])

        time.sleep(2)

        #autoqueue
        self.sv.Start()
        img = []
        img.append(cv2.cvtColor(self.sv.GetScreen(), cv2.COLOR_BGR2GRAY))
        pyautogui.click(x=l+800,y=t+1030)
        entered_match = False
        while entered_match is False:
            img.append(cv2.cvtColor(self.sv.GetScreen(), cv2.COLOR_BGR2GRAY))
            err = np.sum((img[0].astype("float") - img[1].astype("float"))**2)
            err /= float(img[0].shape[0] * img[0].shape[1])
            if err >= 500:
                time.sleep(2)
                pyautogui.click(x=l+950,y=t+850)
                entered_match = True
            del img[1]

        time.sleep(12)
        #enter match & select champ
        pyautogui.click(x=l+570,y=t+250)
        pyautogui.typewrite(self.champion)
        pyautogui.click(x=l+950,y=t+920)

        #ban
        pyautogui.click(x=0,y=0)
        pyautogui.typewrite(self.ban)
        pyautogui.click(x=0,y=0)

        #finalize
        pyautogui.click(x=0,y=0)

        self.sv.Stop()
        self.sv = None

        pygame.init()
        pygame.font.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((self.res_x, self.res_y), pygame.NOFRAME)
        self.screen.fill(colors.get('fuschia')) #transparent background
        # Set window transparency color
        hwnd = pygame.display.get_wm_info()["window"]
        gamewindow = win32gui.FindWindow(None, "League of Legends (TM) Client")
        posX, posY, width, height = win32gui.GetWindowPlacement(gamewindow)[4]

        windowStyles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT


        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                            win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
        win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*colors.get("fuschia")), 0, win32con.LWA_COLORKEY)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, posX,posY, 0,0, win32con.SWP_NOSIZE)



        
        



    def step(self, action):
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
        obs = [self._get_video_frame(), self._get_mouse()]

        # append it to the list of prveious frames 

        reward = self._get_reward(self.current_action, action, 12) #also resets
        self._refresh_hud(0)


        return obs, reward, True, {}

    def reset(self): 
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self._refresh_hud(0)
        obs = [self._get_video_frame(), self._get_mouse()]
        return obs

    def shutdown(self):
        """ Request a server shutdown - currently used by the integration tests to repeatedly create and destroy fresh copies of the server running in a separate thread"""
  
        return 'Server shutting down'

    def _take_action(self, keys):
        # length 9 
        send_keys = []
        for i in range(7):
            if(keys[i] >= .8):
                #send_keys+=ALLOWED_ACTIONS[i]
                #remove temp for testing
                pass
        pyautogui.press(send_keys)
        print('Keys: {2}'.format(send_keys))
        return


    def _get_video_frame(self):
        #record screen
        self.img = self.sv.GetScreen() 
        if self.res_x != np.size(img,0):
            self.img = cv2.resize(img, (self.res_x,self.res_y), interpolation=cv2.INTER_CUBIC) #res is tuple
        return self.img

    def _get_mouse(self):
        return pyautogui.position()

    def _get_current_action(self):
        action = self.current_action 
        self.current_action = [0,0,0,0,0,0]
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
            total_error+= (human_keys[i] - predicted_keys[i])**2

        self.current_action = self.current_action*0.5 #decay reward for late key press
        return total_error/array_length


    def _move_mouse(self, x, y):
        pyautogui.moveTo(x+self.bounds[0], y+self.bounds[2])
        return

    def _key_thread(self):
        for i in range(len(ACTIONS)):
            keyboard.add_hotkey(ACTIONS[i], self._add_key, [i], suppress=False)
        return

    def _meta_thread(self):
        #TODO: collect metadata related to health and mana from image
        thread = Thread(target=self._update_meta, args=("I'ma", "thread"))
        thread.start()

    def _update_meta(self):
        #TODO: collect metadata related to health and mana from image
        
        self.img 

    def _add_key(self, keynum):
        self.current_action[keynum] = 1
        return

    def _refresh_hud(self, img):
        self.screen.blit(self.emotes[img],(0,0))
        pygame.display.update()


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

