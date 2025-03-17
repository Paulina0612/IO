'''

sprawozdanie
wszystko w jednym katalogu

piec opisow pieciu zadan
teoria - wszystkie filtry, opisac rozne kolorystyki, 


'''

import cv2

def Menu():
    print('Menu:')
    for i in range(1, 6):
        print(i, '. Exercise ', i)
    print('0. Exit')
    return


def ex1():
    inFilename = 'kitty.jpg'
    outFilename = 'kitty2.jpg'

    img = cv2.imread(inFilename)



    cv2.imwrite(outFilename, img)
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

 

