'''



'''



import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import os

def Menu():
    print('Menu:')
    for i in range(1, 6):
        print(i, '. Exercise ', i)
    print('0. Exit')
    return


def writeASCII(header, filename, array):
    with open(filename, 'w') as fh:
        fh.write(header)
        array.tofile(fh, sep=' ')
        fh.write('\n')
    return

def writeBINARY(header, filename, array):
    with open(filename, 'wb') as fh:
        fh.write(bytearray(header, 'ascii'))
        array.astype(np.uint8).tofile(fh)
    return

def ex1():
    filenames = ['imgs\\ex1\\ascii_sketch_PPM.ppm', 
                 'imgs\\ex1\\binary_sketch_PPM.ppm',
                 'imgs\\ex1\\ascii_img_PPM.ppm', 
                 'imgs\\ex1\\binary_img_PPM.ppm']
    sketch = np.array([255, 0, 0, 0, 0, 255, 0, 255, 0, 255, 255, 0, 255, 0, 
                       255, 0, 255, 255, 0, 0, 0, 255, 255, 255, 255, 0, 0])
    
    sketch_ascii_header = ('P3\n%d %d\n255\n')%(3, 3) 
    sketch_binary_header = ('P6\n%d %d\n255\n')%(3, 3)

    writeASCII(sketch_ascii_header, filenames[0], sketch.copy())
    writeBINARY(sketch_binary_header, filenames[1], sketch.copy())
    
    sketch_from_ascii_file = cv2.imread(filenames[0])
    sketch_from_bin_file = cv2.imread(filenames[1])
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(sketch_from_ascii_file)
    ax[0].set_xlabel("Sketch from ascii file")
    ax[1].imshow(sketch_from_bin_file)
    ax[1].set_xlabel("Sketch from binary file")
    plt.show()

    inFilename = 'imgs\\original_kitty.jpg'

    original = cv2.imread(inFilename)

    array = np.asarray(original)

    height = len(array)
    width = len(array[0])

    img_ascii_header = ('P3\n%d %d\n255\n')%(width, height) 
    img_binary_header = ('P6\n%d %d\n255\n')%(width, height)

    writeASCII(img_ascii_header, filenames[2], array.copy())
    writeBINARY(img_binary_header, filenames[3], array.copy())
    
    img_from_ascii_file = cv2.imread(filenames[2])
    img_from_bin_file = cv2.imread(filenames[3])
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img_from_ascii_file)
    ax[0].set_xlabel("Image from ascii file")
    ax[1].imshow(img_from_bin_file)
    ax[1].set_xlabel("Image from binary file")
    plt.show()


    print("Size of ASCII file is ", os.path.getsize(filenames[2])
          , "and size of binary file is ", os.path.getsize(filenames[3])
    )

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

 