'''

na za tydzien 4 pierwsze zadania 

'''



import struct
import zlib
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
    header = ('P3\n%d %d\n255\n')%(120, 8)
    divider = 15

    step = np.array([0, 0, 0], dtype=np.uint8)
    image = np.array([0, 0, 0], dtype=np.uint8)

    # to blue
    for i in range(0,17):
        step[2] += divider
        image = np.append(image, step)

    # to cyjan
    for i in range(0,17):
        step[1] += divider
        image = np.append(image, step)

    # to green
    for i in range(0,17):
        step[2] -= divider
        image = np.append(image, step)

    # to yellow
    for i in range(0,17):
        step[0] += divider
        image = np.append(image, step)
    
    # to red
    for i in range(0,17):
        step[1] -= divider
        image = np.append(image, step)

    # to magenta
    for i in range(0,17):
        step[2] += divider
        image = np.append(image, step)

    # to white
    for i in range(0,17):
        step[1] += divider
        image = np.append(image, step)

    line = np.copy(image)
    img = np.array([0], dtype=np.uint8)
    image = np.append(image, image)
    image = np.append(image, image)
    image = np.append(image, image)

    img = np.array([0], dtype=np.uint8) 
    img = np.append(img, image)
    print(img)

    filename = 'lab2\\source_code\\imgs\\ex2\\output.ppm'
    with open(filename, 'w') as fh:
        fh.write(header)
        image.tofile(fh, sep=' ')
        fh.write('\n')

    return line

def ex3():
    # Image data
    line = ex2()
    image = np.array([0], dtype=np.uint8)
    image = np.append(image, line)
    image = np.append(image, image)
    image = np.append(image, image)
    image = np.append(image, image)

    # Signature
    png_file_signature = b'\x89PNG\r\n\x1a\n' 

    # Header
    header_id = b'IHDR' 
    header_content = b'\x00\x00\x00\x78\x00\x00\x00\x08\x08\x02\x00\x00\x00' 
    header_size = struct.pack('!I', len(header_content))  
    header_crc = struct.pack('!I', zlib.crc32(header_id + header_content)) 
    png_file_header = header_size + header_id + header_content + header_crc

    # Data
    data_id = b'IDAT'  
    data_content = zlib.compress(image,0)  
    data_size = struct.pack('!I', len(data_content))  
    data_crc = struct.pack('!I', zlib.crc32(data_id + data_content))  
    png_file_data = data_size + data_id + data_content + data_crc

    # End
    end_id = b'IEND'
    end_content = b''
    end_size = struct.pack('!I', len(end_content))
    end_crc = struct.pack('!I', zlib.crc32(end_id + end_content))
    png_file_end = end_size + end_id + end_content + end_crc

    # Save the PNG image as a binary file
    with open('lab4.png', 'wb') as fh:
        fh.write(png_file_signature)
        fh.write(png_file_header)
        fh.write(png_file_data)
        fh.write(png_file_end)
    
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
        filename = 'lab2\\source_code\\imgs\\ex2\\output.ppm'
        image_from_file = cv2.imread(filename)
        plt.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB))
        plt.show()
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

 