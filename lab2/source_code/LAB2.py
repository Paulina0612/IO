'''



'''



import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def Menu():
    print('Menu:')
    for i in range(1, 6):
        print(i, '. Exercise ', i)
    print('0. Exit')
    return


def ex1():
    inFilename = 'imgs\\original_kitty.jpg'

    original = cv2.imread(inFilename)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    array = np.asarray(original)

    height = len(array)
    width = len(array[0])
    
    ppm_ascii_header = ('P3\n%d %d\n255\n')%(width, height) 
    ppm_binary_header = ('P6\n%d %d\n255\n')%(width, height)

    with open('imgs\\ex1\\asciiPPM.ppm', 'w') as fh:
        fh.write(ppm_ascii_header)
        array.tofile(fh, sep=' ')
        fh.write('\n')

    with open('imgs\\ex1\\binaryPPM.ppm', 'wb') as fh:
        fh.write(bytearray(ppm_binary_header, 'ascii'))
        array.tofile(fh)

    return

def ex2():
    
    return

def ex3():

    return

def ex4():
    
    return 

def ex5():
    
    return





plt.rcParams["figure.figsize"] = (3, 4)

while True:
    Menu()
    opt = input('Choose option from menu (enter number): ')

    if opt == '1':
        ex1()
    elif opt == '2':
        ex2()
    elif opt == '3':
        ex3()
    elif opt == '4':
        ex4()
    elif opt == '5':
        ex5()
    elif opt == '0':
        break
    else:
        print('Wrong option!')

    print('\n\n\n')

 