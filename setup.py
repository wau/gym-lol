from setuptools import setup, find_packages

setup(name='gym_lol',
      version='0.0.1',
      description='OpenAI Gym environment for League of Legends',
      url='https://github.com/wau/gym-lol',
      author='Taylor McNally',
      author_email='Redacted',
      license='MIT License',
      packages=find_packages(),
      package_data={'': ['assets/*.xml']},
      zip_safe=False,
      install_requires=['gym>=0.2.3', 'pyautogui', 'pytesseract', 'keyboard'],
      dependency_links=['git+https://github.com/tambetm/minecraft-py.git#egg=minecraft_py-0.0.2',
                        'http://www.pygame.org/ftp/pygame-1.9.1release.tar.gz#egg=pygame-1.9.1release']
)