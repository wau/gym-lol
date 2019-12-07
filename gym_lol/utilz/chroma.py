from ChromaPython import ChromaApp, ChromaAppInfo, ChromaColor, Colors, ChromaGrid
from time import sleep

Key_Locations = {
    'Q': (1,1),
    'W': (1,2),
    'E': (1,3),
    'R': (1,4),
    'D': (2,3),
    'F': (2,4)
}


class KeyboardChroma():
    def __init__(self):
        Info = ChromaAppInfo()
        Info.DeveloperName = 'moo'
        Info.DeveloperContact = 'moo@moo.com'
        Info.Category = 'application'
        Info.SupportedDevices = ['keyboard', 'mouse', 'mousepad']
        Info.Description = 'mooo.'
        Info.Title = 'App'
        self.app = ChromaApp(Info)
        self.keyboard_grid = ChromaGrid('Keyboard')


    def startup_animation(self):
        self.app.Keyboard.setStatic(Colors.RED)
        for i in range(0, len(self.keyboard_grid)):
            for j in range(0, len(self.keyboard_grid[i])):
                self.keyboard_grid[i][j].set(red=0, green=255, blue=0)
                self.app.Keyboard.setCustomGrid(self.keyboard_grid)
                self.app.Keyboard.applyGrid()
                sleep(0.1)

    def key_animation(self, keys, color):
        for key in keys:
            x, y = self._get_key_loc(key)
            self.keyboard_grid[x][y].set(red=5*color, green=255, blue=0)

    
    def _get_key_loc(self, key):
        x, y = Key_Locations.get(key)
        return x, y

