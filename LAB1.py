'''

sprawozdanie
wszystko w jednym katalogu

piec opisow pieciu zadan
teoria - wszystkie filtry, opisac rozne kolorystyki, 


'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

def Menu():
    print('Menu:')
    for i in range(1, 6):
        print(i, '. Exercise ', i)
    print('0. Exit')
    return


def ex1():
    inFilename = 'original_kitty.jpg'
    outFilename = 'processed_kitty.jpg'

    original = cv2.imread(inFilename)

    img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    matrix = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

    img = cv2.filter2D(img, -1, matrix)
    cv2.imwrite(outFilename, img)
    print('Original image is %s and processed image is %s' % (inFilename, outFilename))
    return

def ex2():
    print('zad2')
    return

def ex3():
    print('zad3')
    return

def ex4():
    print('zad4')
    return

def ex5():
    print('zad5')
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

 

