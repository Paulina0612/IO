'''



'''



#import cv2
#import numpy as np
#from matplotlib import pyplot as plt

def Menu():
    print('Menu:')
    for i in range(1, 6):
        print(i, '. Exercise ', i)
    print('0. Exit')
    return


def ex1():
    
    return

def ex2():
    
    return

def ex3():

    return

def ex4():
    
    return 

def ex5():
    
    return





#plt.rcParams["figure.figsize"] = (3, 4)

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

 