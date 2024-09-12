#for i in range(6):
#	print(i,123)
#print('\033[A\x1b[2K')

#dict1={'state':'','name':'','score':'','r_cd':'','i_cd':'','HCstate':''}
#for k in dict1.keys():
#	print(k+":")
#print('\033[A\x1b[2k'*2)
#print('as')
#import curses
#screen=curses.initscr()
#for i in range(5):
#	screen.addstr(i,0,('Hello world!'+'abc'))
#screen.refresh()
from enum import Enum
a=0
class color_a(Enum):
	blue=0
	red=1
	green=2
	cyan=3
	pink=4
	yellow=5
	white=6
	black=7
color=[#BGR
	(255,   0,   0), #0  blue       
	(  0,   0, 255), #1  red     
	(  0, 255,   0), #2  green   
	(255, 255,   0), #3  cyan    
	(255,   0, 255), #4  pink    
	(  0, 255, 255), #5  yellow  
	(255, 255, 255), #6  white   
	(  0,   0,   0)  #7  black   
	]
a=color_a(a)
print(type(a.name))
