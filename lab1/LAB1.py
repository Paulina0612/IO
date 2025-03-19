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
    inFilename = 'lab1\\ex1\\original_kitty.jpg'
    outFilename = 'lab1\\ex1\\processed_kitty.jpg'

    original = cv2.imread(inFilename)

    # Konwersja BGR -> RGB
    img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Macierz przekształcenia
    matrix = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

    # Przekształcenie obrazu
    img = cv2.filter2D(img, -1, matrix)
    
    # Zapis obrazu
    cv2.imwrite(outFilename, img)

    print('Original image is %s and processed image is %s' % (inFilename, outFilename))
    return

def ex2():
    inFilename = 'lab1\\ex2\\original_kitty.jpg'
    outFilename = 'lab1\\ex2\\processed_kitty.jpg'

    original = cv2.imread(inFilename)

    # Konwersja BGR -> RGB
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Macierz przekształcenia
    transformation_matrix = np.array([[0.393, 0.769, 0.189],
                                      [0.349, 0.689, 0.168],
                                      [0.272, 0.534, 0.131]])

    # Przekształcenie obrazu
    processed_img = np.dot(original, transformation_matrix.T)

    # Ograniczenie wartości do przedziału [0, 1]
    processed_img = np.clip(processed_img, 0, 1)

    # Konwersja z powrotem do skali 0-255
    processed_img = (processed_img * 255).astype(np.uint8)

    # Konwersja RGB -> BGR przed zapisem do pliku
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

    # Zapis obrazu
    cv2.imwrite(outFilename, processed_img)

    print('Original image is %s and processed image is %s' % (inFilename, outFilename))
    return

def ex3():
    inFilename = 'lab1\\ex3\\original_kitty.jpg'

    original = cv2.imread(inFilename)

    # Konwersja BGR -> RGB
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Macierz przekształcenia
    add_matrix = [0, 128, 128]
    transformation_matrix = np.array([[0.229, 0.587, 0.114],
                                      [0.500, -0.418, -0.082],
                                      [-0.168, -0.331, 0.500]])

    # Przekształcenie obrazu
    processed_img = np.dot(original, transformation_matrix.T)
    processed_img += add_matrix

    fig, ax = plt.subplots(2,2)
    ax[0, 0].imshow(original)
    ax[0, 0].set_xlabel("Oryginal")
    ax[0, 1].imshow(processed_img[:,:,0], cmap="Greys_r")
    ax[0, 1].set_xlabel("Y")
    ax[1, 0].imshow(processed_img[:,:,2], cmap="Greys_r")
    ax[1, 0].set_xlabel("Cb")
    ax[1, 1].imshow(processed_img[:,:,1], cmap="Greys_r")
    ax[1, 1].set_xlabel("Cr")

    plt.show()

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

 

