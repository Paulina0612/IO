import binascii
import math
import zlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.fftpack import dct
from scipy.fftpack import idct
from lorem.text import TextLorem

def Menu():
    print('Menu:')
    for i in range(1, 2):
        print(('%d. Exercise %d List %d')%(i, i, 1))
    for i in range(2, 7):
        print(('%d. Exercise %d List %d')%(i, i-1, 2))
    print('0. Exit')
    return


def gen_rainbow_img():
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

    return image

def dct2(array):
    return dct(dct(array, axis=0, norm='ortho'), axis=1, norm='ortho')

#
# Calculate quantisation matrices
#
# Based on: https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html
#           #step-3-and-4-discrete-cosinus-transform-and-quantisation
#
_QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 48, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

_QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]])


def _scale(QF):
    if QF < 50 and QF >= 1:
        scale = np.floor(5000 / QF)
    elif QF < 100:
        scale = 200 - 2 * QF
    else:
        raise ValueError('Quality Factor must be in the range [1..99]')

    scale = scale / 100.0
    return scale


def QY(QF=85):
    return _QY * _scale(QF)


def QC(QF=85):
    return _QC * _scale(QF)

def idct2(array):
    return idct(idct(array, axis=0, norm='ortho'), axis=1, norm='ortho')

def ex1():
    # 0. Generate the rainbow image with padding at the start and end
    rainbow = gen_rainbow_img()
    image = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)  # Start padding
    end = np.array([[255, 255, 255]] * 4, dtype=np.uint8)  # End padding
    image = np.append(image, rainbow)
    image = np.append(image, end)
    image = np.append(image, image)  # Duplicate 2x
    image = np.append(image, image)  # Duplicate 4x
    image = np.append(image, image)  # Duplicate 8x
    image = image.reshape(8, 128, 3)  # Final shape: 8 rows, 128 cols, RGB

    # 1. Convert image from RGB to YCrCb (luminance + chrominance)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    # 2. Perform chroma subsampling (downsample Cr and Cb channels)
    samp0 = np.copy(image)  # No subsampling
    samp2Cr, samp2Cb = [], []  # 4:2:2 subsampling
    samp4Cr, samp4Cb = [] ,[]  # 4:1:1 subsampling
    for i in range(len(image)):
        for j in range(len(image[0])):
            if (2*i + j) % 4 == 0:
                samp4Cr.append(image[i][j][1])
                samp4Cb.append(image[i][j][2])
            if (i + j) % 2 == 0:
                samp2Cr.append(image[i][j][1])
                samp2Cb.append(image[i][j][2])

    # 3. Divide image into 8x8 blocks for each channel
    horizontal = len(samp0[0]) // 8
    blocks = len(samp0) * horizontal // 8
    blocksY = np.zeros([blocks, 8, 8], dtype=np.uint8)
    blocks0Cr = np.zeros_like(blocksY)
    blocks0Cb = np.zeros_like(blocksY)
    blocks2Cr = np.zeros([blocks//2, 8, 8], dtype=np.uint8)
    blocks2Cb = np.zeros_like(blocks2Cr)
    blocks4Cr = np.zeros([blocks//4, 8, 8], dtype=np.uint8)
    blocks4Cb = np.zeros_like(blocks4Cr)

    # Fill 8x8 blocks from full YCrCb image
    x, y, z = 0, 0, 0
    for i in range(len(samp0)):
        for j in range(len(samp0[0])):
            blocksY[x][y][z] = samp0[i][j][0]
            blocks0Cr[x][y][z] = samp0[i][j][1]
            blocks0Cb[x][y][z] = samp0[i][j][2]
            z = (z+1) % 8
            if z == 0:
                x = (x+1) % horizontal
        y = (y+1) % 8

    # Fill 8x8 blocks from 4:2:2 (2x downsampled chroma)
    horizontal //= 2
    x, y, z = 0, 0, 0
    for i in range(len(samp2Cr)):
        blocks2Cr[x][y][z] = samp2Cr[i]
        blocks2Cb[x][y][z] = samp2Cb[i]
        z = (z+1) % 8
        if z == 0:
            x = (x+1) % horizontal
            if x == 0:
                y = (y+1) % 8

    # Fill 8x8 blocks from 4:1:1 (4x downsampled chroma)
    horizontal //= 2
    x, y, z = 0, 0, 0
    for i in range(len(samp4Cr)):
        blocks4Cr[x][y][z] = samp4Cr[i]
        blocks4Cb[x][y][z] = samp4Cb[i]
        z = (z+1) % 8
        if z == 0:
            x = (x+1) % horizontal
            if x == 0:
                y = (y+1) % 8

    # 4. Apply DCT (Discrete Cosine Transform) to each 8x8 block
    DCTY, DCT0Cr, DCT0Cb, DCT2Cr, DCT2Cb, DCT4Cr, DCT4Cb = [], [], [], [], [], [], []
    for i in range(len(blocksY)):
        DCTY = np.append(DCTY, dct2(blocksY[i]))
        DCT0Cr = np.append(DCT0Cr, dct2(blocks0Cr[i]))
        DCT0Cb = np.append(DCT0Cb, dct2(blocks0Cb[i]))
    DCTY = DCTY.reshape(-1, 8, 8)
    DCT0Cr = DCT0Cr.reshape(-1, 8, 8)
    DCT0Cb = DCT0Cb.reshape(-1, 8, 8)
    for i in range(len(blocks2Cr)):
        DCT2Cr = np.append(DCT2Cr, dct2(blocks2Cr[i]))
        DCT2Cb = np.append(DCT2Cb, dct2(blocks2Cb[i]))
    DCT2Cr = DCT2Cr.reshape(-1, 8, 8)
    DCT2Cb = DCT2Cb.reshape(-1, 8, 8)
    for i in range(len(blocks4Cr)):
        DCT4Cr = np.append(DCT4Cr, dct2(blocks4Cr[i]))
        DCT4Cb = np.append(DCT4Cb, dct2(blocks4Cb[i]))
    DCT4Cr = DCT4Cr.reshape(-1, 8, 8)
    DCT4Cb = DCT4Cb.reshape(-1, 8, 8)

    # 5. Quantize DCT coefficients using standard JPEG quantization matrices
    div0 = QY()
    div1 = QC()
    divY, div0Cr, div0Cb, div2Cr, div2Cb, div4Cr, div4Cb = [], [], [], [], [], [], []
    for i in range(len(DCTY)):
        divY = np.append(divY, np.divide(DCTY[i], div0))
        div0Cr = np.append(div0Cr, np.divide(DCT0Cr[i], div1))
        div0Cb = np.append(div0Cb, np.divide(DCT0Cb[i], div1))
    divY = divY.reshape(-1, 8, 8)
    div0Cr = div0Cr.reshape(-1, 8, 8)
    div0Cb = div0Cb.reshape(-1, 8, 8)
    for i in range(len(DCT2Cr)):
        div2Cr = np.append(div2Cr, np.divide(DCT2Cr[i], div1))
        div2Cb = np.append(div2Cb, np.divide(DCT2Cb[i], div1))
    div2Cr = div2Cr.reshape(-1, 8, 8)
    div2Cb = div2Cb.reshape(-1, 8, 8)
    for i in range(len(DCT4Cb)):
        div4Cr = np.append(div4Cr, np.divide(DCT4Cr[i], div1))
        div4Cb = np.append(div4Cb, np.divide(DCT4Cb[i], div1))
    div4Cr = div4Cr.reshape(-1, 8, 8)
    div4Cb = div4Cb.reshape(-1, 8, 8)

    # 6. Round quantized values
    blockY = np.asarray(divY).round()
    block0Cr = np.asarray(div0Cr).round()
    block0Cb = np.asarray(div0Cb).round()
    block2Cr = np.asarray(div2Cr).round()
    block2Cb = np.asarray(div2Cb).round()
    block4Cr = np.asarray(div4Cr).round()
    block4Cb = np.asarray(div4Cb).round()

    # 7. Apply zig-zag scan to convert 8x8 blocks to 1D arrays for entropy coding
    def zigZag(source):
        temp = np.zeros(64, dtype=np.uint8)
        n = x = y = 0
        while n < 63:
            while y < 8 and x > -1:
                temp[n] = source[x][y]
                y += 1
                x -= 1
                n += 1
            x += 1
            if y == 8:
                y -= 1
                x += 1
            while x < 8 and y > -1:
                temp[n] = source[x][y]
                y -= 1
                x += 1
                n += 1
            y += 1
            if x == 8:
                x -= 1
                y += 1
        temp[63] = source[7][7]
        return temp

    zigZagY, zigZag0Cr, zigZag0Cb, zigZag2Cr, zigZag2Cb, zigZag4Cr, zigZag4Cb = [], [], [], [], [], [], []
    for i in range(len(blockY)):
        zigZagY = np.append(zigZagY, zigZag(blockY[i]))
        zigZag0Cr = np.append(zigZag0Cr, zigZag(block0Cr[i]))
        zigZag0Cb = np.append(zigZag0Cb, zigZag(block0Cb[i]))
    for i in range(len(blocks2Cr)):
        zigZag2Cr = np.append(zigZag2Cr, zigZag(block2Cr[i]))
        zigZag2Cb = np.append(zigZag2Cb, zigZag(block2Cb[i]))
    for i in range(len(blocks4Cb)):
        zigZag4Cr = np.append(zigZag4Cr, zigZag(block4Cr[i]))
        zigZag4Cb = np.append(zigZag4Cb, zigZag(block4Cb[i]))

    # 8. Flatten arrays, concatenate Y and chroma, and compress with zlib
    flatY = zigZagY.flatten()
    flat0Cr = zigZag0Cr.flatten()
    flat0Cb = zigZag0Cb.flatten()
    flat2Cr = zigZag2Cr.flatten()
    flat2Cb = zigZag2Cb.flatten()
    flat4Cr = zigZag4Cr.flatten()
    flat4Cb = zigZag4Cb.flatten()

    im0 = np.concatenate((flatY, flat0Cr, flat0Cb))
    im2 = np.concatenate((flatY, flat2Cr, flat2Cb))
    im4 = np.concatenate((flatY, flat4Cr, flat4Cb))

    comp0 = zlib.compress(im0, 0)
    comp2 = zlib.compress(im2, 0)
    comp4 = zlib.compress(im4, 0)

    print("Without sampling: " + str(len(comp0)))
    print("Sampling every 2nd element: " + str(len(comp2)))
    print("Sampling every 4th element: " + str(len(comp4)))

        # 5'. Dequantize: multiply blocks by quantization matrix to restore original DCT scale
    mulY, mul0Cr, mul0Cb, mul2Cr, mul2Cb, mul4Cr, mul4Cb = [], [], [], [], [], [], []
    for i in range(len(blockY)):
        mulY = np.append(mulY, np.multiply(blockY[i], div0))
        mul0Cr = np.append(mul0Cr, np.multiply(block0Cr[i], div1))
        mul0Cb = np.append(mul0Cb, np.multiply(block0Cb[i], div1))
    mulY = mulY.reshape(-1, 8, 8)
    mul0Cr = mul0Cr.reshape(-1, 8, 8)
    mul0Cb = mul0Cb.reshape(-1, 8, 8)

    for i in range(len(block2Cr)):
        mul2Cr = np.append(mul2Cr, np.multiply(block2Cr[i], div1))
        mul2Cb = np.append(mul2Cb, np.multiply(block2Cb[i], div1))
    mul2Cr = mul2Cr.reshape(-1, 8, 8)
    mul2Cb = mul2Cb.reshape(-1, 8, 8)

    for i in range(len(block4Cb)):
        mul4Cr = np.append(mul4Cr, np.multiply(block4Cr[i], div1))
        mul4Cb = np.append(mul4Cb, np.multiply(block4Cb[i], div1))
    mul4Cr = mul4Cr.reshape(-1, 8, 8)
    mul4Cb = mul4Cb.reshape(-1, 8, 8)

    # 4'. Apply inverse DCT to each block to reconstruct spatial domain
    reverseY, reverse0Cr, reverse0Cb, reverse2Cr, reverse2Cb, reverse4Cr, reverse4Cb = [], [], [], [], [], [], []
    for i in range(len(blockY)):
        reverseY = np.append(reverseY, idct2(mulY[i]))
        reverse0Cr = np.append(reverse0Cr, idct2(mul0Cr[i]))
        reverse0Cb = np.append(reverse0Cb, idct2(mul0Cb[i]))
    reverseY = reverseY.reshape(-1, 8, 8)
    reverse0Cr = reverse0Cr.reshape(-1, 8, 8)
    reverse0Cb = reverse0Cb.reshape(-1, 8, 8)

    for i in range(len(block2Cr)):
        reverse2Cr = np.append(reverse2Cr, idct2(mul2Cr[i]))
        reverse2Cb = np.append(reverse2Cb, idct2(mul2Cb[i]))
    reverse2Cr = reverse2Cr.reshape(-1, 8, 8)
    reverse2Cb = reverse2Cb.reshape(-1, 8, 8)

    for i in range(len(block4Cb)):
        reverse4Cr = np.append(reverse4Cr, idct2(mul4Cr[i]))
        reverse4Cb = np.append(reverse4Cb, idct2(mul4Cb[i]))
    reverse4Cr = reverse4Cr.reshape(-1, 8, 8)
    reverse4Cb = reverse4Cb.reshape(-1, 8, 8)

    # 3'. Merge all 8x8 blocks back into full chroma planes
    restoredY, restored0Cr, restored0Cb = [], [], []
    restored2Cr, restored2Cb = [], []
    restored4Cr, restored4Cb = [], []

    # Helper for stitching blocks back together
    i = j = y = 0
    while (i + j) < len(reverse4Cr):
        for x in range(8):
            restored4Cr = np.append(restored4Cr, reverse4Cr[i + j][y][x])
            restored4Cb = np.append(restored4Cb, reverse4Cb[i + j][y][x])
        i = (i + 1) % horizontal
        if i == 0:
            y += 1
            if y == 8:
                y = 0
                j += horizontal
    horizontal *= 2

    i = j = y = 0
    while (i + j) < len(reverse2Cb):
        for x in range(8):
            restored2Cr = np.append(restored2Cr, reverse2Cr[i + j][y][x])
            restored2Cb = np.append(restored2Cb, reverse2Cb[i + j][y][x])
        i = (i + 1) % horizontal
        if i == 0:
            y += 1
            if y == 8:
                y = 0
                j += horizontal
    horizontal *= 2

    i = j = y = 0
    while (i + j) < len(reverseY):
        for x in range(8):
            restoredY = np.append(restoredY, reverseY[i + j][y][x])
            restored0Cr = np.append(restored0Cr, reverse0Cr[i + j][y][x])
            restored0Cb = np.append(restored0Cb, reverse0Cb[i + j][y][x])
        i = (i + 1) % horizontal
        if i == 0:
            y += 1
            if y == 8:
                y = 0
                j += horizontal

    # 2'. Upsample the Cr and Cb channels to match full resolution
    def imagine(Y, Cr, Cb, samp):
        # Reconstruct full YCrCb image from Y, Cr, Cb with optional duplication
        channelled = []
        x = 0
        y = 0
        for n in range(len(Y)):
            temp = np.zeros(3, dtype=np.uint8)
            temp[0] = Y[n]
            temp[1] = Cr[x]
            temp[2] = Cb[x]
            channelled = np.append(channelled, temp)
            if samp > 0:
                y = (y + 1) % samp
            if y == 0:
                x += 1
        channelled = channelled.reshape(8, 128, 3)
        channelled = channelled.astype(dtype=np.uint8)
        return channelled

    # Get final upsampled YCrCb images
    image0N = imagine(restoredY, restored0Cr, restored0Cb, 0)
    image2N = imagine(restoredY, restored2Cr, restored2Cb, 2)
    image4N = imagine(restoredY, restored4Cr, restored4Cb, 4)

    # 1'. Convert back from YCrCb to RGB
    image0 = cv2.cvtColor(image0N, cv2.COLOR_YCrCb2RGB)
    image2 = cv2.cvtColor(image2N, cv2.COLOR_YCrCb2RGB)
    image4 = cv2.cvtColor(image4N, cv2.COLOR_YCrCb2RGB)

    # 0'. Save reconstructed images to disk as PPM (text-based format)
    ppm_head = 'P3\n128 8\n255\n'
    with open('imgs\\ex1\\lab4_0.ppm', 'w') as fh:
        fh.write(ppm_head)
        image0.tofile(fh, sep=' ')
        fh.write('\n')
    with open('imgs\\ex1\\lab4_2.ppm', 'w') as fh:
        fh.write(ppm_head)
        image2.tofile(fh, sep=' ')
        fh.write('\n')
    with open('imgs\\ex1\\lab4_4.ppm', 'w') as fh:
        fh.write(ppm_head)
        image4.tofile(fh, sep=' ')
        fh.write('\n')

    # === Display output ===
    fig, ax = plt.subplots(3, 1)
    image_from0 = cv2.imread('imgs\\ex1\\lab4_0.ppm')
    image_from2 = cv2.imread('imgs\\ex1\\lab4_2.ppm')
    image_from4 = cv2.imread('imgs\\ex1\\lab4_4.ppm')

    ax[0].imshow(cv2.cvtColor(image_from0, cv2.COLOR_BGR2RGB))
    ax[0].set_xlabel('No sampling')

    ax[1].imshow(cv2.cvtColor(image_from2, cv2.COLOR_BGR2RGB))
    ax[1].set_xlabel('Sampling: every 2nd')

    ax[2].imshow(cv2.cvtColor(image_from4, cv2.COLOR_BGR2RGB))
    ax[2].set_xlabel('Sampling: every 4th')

    plt.tight_layout()
    plt.show()

    return




def encode_as_binary_array(msg):
    """Encode a message as a binary string."""
    msg = msg.encode("utf-8")
    msg = msg.hex()
    msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
    msg = [ "{:08b}".format(int(el, base=16)) for el in msg]
    return "".join(msg)

def decode_from_binary_array(binary_str):
    """Decode a binary string to UTF-8."""
    # Split binary string into 8-bit chunks
    byte_chunks = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]

    # Pad the last chunk if it's not 8 bits
    if len(byte_chunks[-1]) != 8:
        byte_chunks[-1] = byte_chunks[-1].ljust(8, '0')

    # Convert each chunk to hexadecimal
    hex_str = "".join("{:02x}".format(int(b, 2)) for b in byte_chunks)

    # Convert hex string to bytes, then decode
    result = binascii.unhexlify(hex_str)
    return result.decode("utf-8", errors="replace")

def load_image(path, pad=False):
    """Load an image.
    If pad is set then pad an image to multiple of 8 pixels.
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if pad:
        y_pad = 8 - (image.shape[0] % 8)
        x_pad = 8 - (image.shape[1] % 8)
        image = np.pad(
            image, ((0, y_pad), (0, x_pad) ,(0, 0)), mode='constant')
    return image

def save_image(path, image):
    """Save an image."""
    plt.imsave(path, image)

def clamp(n, minn, maxn):
    """Clamp the n value to be in range (minn, maxn)."""
    return max(min(maxn, n), minn)

def hide_message(image, message, nbits=1, spos=0):
    """Hide a message in an image (LSB).
    nbits: number of least significant bits
    spos: start position (in pixels) to hide the message
    """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()

    if len(message) > (len(image) - spos) * nbits:
        print("Message is too long :(")
        return None

    chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]

    for i, chunk in enumerate(chunks):
        index = spos + i
        byte = "{:08b}".format(image[index])
        new_byte = byte[:-nbits] + chunk
        image[index] = int(new_byte, 2)

    return image.reshape(shape)


def reveal_message(image, nbits=1, length=0, spos=0):
    """Reveal the hidden message.
    nbits: number of least significant bits
    length: length of the message in bits.
    spos: start position (in pixels) to begin revealing the message
    """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()

    length_in_pixels = math.ceil(length / nbits)
    message = ""

    if spos + length_in_pixels > len(image) or length_in_pixels <= 0:
        length_in_pixels = len(image) - spos

    for i in range(length_in_pixels):
        index = spos + i
        byte = "{:08b}".format(image[index])
        message += byte[-nbits:]

    mod = length % -nbits
    if mod != 0:
        message = message[:mod]

    return message


def ex2():
    original_image = load_image('imgs\\ex2\\original_kitty.png')
    message = "Canvas"
    binary = encode_as_binary_array(message)
    n = 1
    image_with_message = hide_message(original_image, binary, n) 
    save_image('imgs\\ex2\\kitty_with_message.png', image_with_message)
    
    image_with_message_png = load_image("imgs\\ex2\\kitty_with_message.png")
    # Wczytanie obrazka PNG
    secret_message_png = decode_from_binary_array(
        reveal_message(image_with_message_png, nbits=n,
            length=len(binary))) # Odczytanie ukrytej wiadomości z PNG
    
    print("Secret message from PNG: ", secret_message_png)
    
    # Wyświetlenie obrazków
    f, ar = plt.subplots(2,2)
    ar[0,0].imshow(original_image)
    ar[0,0].set_title("Original image")
    ar[0,1].imshow(image_with_message)
    ar[0,1].set_title("Image with message")
    ar[1,0].imshow(image_with_message_png)
    ar[1,0].set_title("PNG image")
    plt.show()

    return


def ex3_a():
    lorem = TextLorem(srange=(194401, 194401))
    message = lorem.sentence()
    original_image = load_image('imgs\\ex2\\original_kitty.png')
    binary = encode_as_binary_array(message)
    n = 1
    image_with_message = hide_message(original_image, binary, n) 
    if(image_with_message is None):
        return
    save_image('imgs\\ex3\\kitty_with_message.png', image_with_message)
    
    image_with_message_png = load_image("imgs\\ex3\\kitty_with_message.png")
    # Wczytanie obrazka PNG
    secret_message_png = decode_from_binary_array(
        reveal_message(image_with_message_png, nbits=n,
            length=len(binary))) # Odczytanie ukrytej wiadomości z PNG
    
    print("Secret message from PNG: ", secret_message_png)
    
    # Wyświetlenie obrazków
    f, ar = plt.subplots(2,2)
    ar[0,0].imshow(original_image)
    ar[0,0].set_title("Original image")
    ar[0,1].imshow(image_with_message)
    ar[0,1].set_title("Image with message")
    ar[1,0].imshow(image_with_message_png)
    ar[1,0].set_title("PNG image")
    plt.show()

    return


def mse(imageA, imageB):
    """Calculate the mean squared error between two images."""
    # The 'mean' function calculates the mean of the squared differences
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

def ex3_b():
    lorem = TextLorem(srange=(2000, 2000))
    message = lorem.sentence()
    original_image = load_image('imgs\\ex2\\original_kitty.png')
    binary = encode_as_binary_array(message)
    n = 1
    images_with_message = [None] * 8
    mse_values = [None] * 8

    while n < 9:
        images_with_message[n-1] = hide_message(original_image, binary, n) 
        if(images_with_message[n-1] is None):
            return
        save_image('imgs\\ex3\\kitty_with_message_%d.png'%(n), images_with_message[n-1])
        
        image_with_message_png = load_image('imgs\\ex3\\kitty_with_message_%d.png'%(n))
        mse_values[n-1] = mse(original_image, image_with_message_png)
        
        n += 1
    
    # Wyświetlenie obrazków
    f, ar = plt.subplots(3,4)
    ar[0,0].imshow(original_image)
    ar[0,0].set_title("Original image")
    ar[1,0].imshow(images_with_message[0])
    ar[1,0].set_title("Image with nbits=1")
    ar[1,1].imshow(images_with_message[1])
    ar[1,1].set_title("Image with nbits=2")
    ar[1,2].imshow(images_with_message[2])
    ar[1,2].set_title("Image with nbits=3")
    ar[1,3].imshow(images_with_message[3])
    ar[1,3].set_title("Image with nbits=4")
    ar[2,0].imshow(images_with_message[4])
    ar[2,0].set_title("Image with nbits=5")
    ar[2,1].imshow(images_with_message[5])
    ar[2,1].set_title("Image with nbits=6")
    ar[2,2].imshow(images_with_message[6])
    ar[2,2].set_title("Image with nbits=7")
    ar[2,3].imshow(images_with_message[7])
    ar[2,3].set_title("Image with nbits=8")
    plt.show()

    return mse_values

def ex3_c(mse:list):
    plt.plot(range(1,9), mse)
    plt.title('MSE to nbits')
    plt.ylabel('MSE')
    plt.xlabel('Number of bits')
    plt.show()
    return


def ex3():
    print("a)")
    ex3_a()
    print("b)")
    mse = ex3_b()
    print("c)")
    print("MSE values:")
    for i in range(8):
        print("MSE for nbits=%d: %f" % (i+1, mse[i]))
    print("d)")
    ex3_c(mse)
    return


def ex4():
    # Load the image
    image = load_image('imgs\\ex2\\original_kitty.png')
    message = "Canvas"
    binary = encode_as_binary_array(message)
    n = 1
    spos = 10
    image_with_message = hide_message(image, binary, n, spos)
    save_image('imgs\\ex4\\kitty_with_message.png', image_with_message)

    image_with_message_png = load_image("imgs\\ex4\\kitty_with_message.png")
    # Wczytanie obrazka PNG
    secret_message_png = decode_from_binary_array(
        reveal_message(image_with_message_png, nbits=n,
            length=len(binary), spos=spos)) # Odczytanie ukrytej wiadomości z PNG
    print("Secret message from PNG: ", secret_message_png)
    return


def hide_image(image, secret_image_path, nbits=1):
    with open(secret_image_path, "rb") as file:
        secret_img = file.read()
    secret_img = secret_img.hex()
    secret_img = [secret_img[i:i + 2] for i in range(0,
    len(secret_img), 2)]
    secret_img = ["{:08b}".format(int(el, base=16)) for el in
    secret_img]
    secret_img = "".join(secret_img)
    return hide_message(image, secret_img, nbits), len(secret_img)


def reveal_image(image, length, nbits=1):
    """Odzyskaj obraz z zakodowanego obrazu."""
    # Krok 1: odzyskaj ciąg binarny z ukrytego obrazu
    binary_data = reveal_message(image, length=length, nbits=nbits)

    # Krok 2: Podziel dane binarne na bajty (8-bitowe fragmenty)
    bytes_list = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]

    # Krok 3: Konwersja binarnego ciągu na bajty
    byte_values = bytearray(int(b, 2) for b in bytes_list)

    # Krok 4: Konwersja bajtów do tablicy numpy i próba odczytania obrazu
    image_array = np.frombuffer(byte_values, dtype=np.uint8)

    try:
        # Przypuszczenie: obraz RGB
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
    except:
        print("Nie udało się odczytać ukrytego obrazu.")
        return None

    return decoded_image


def ex5():
    original_image = load_image('imgs\\ex5\\original_kitty.png')
    secret_image_path = 'imgs\\ex5\\secret_kitty.png'
    n = 1
    image_with_message, length = hide_image(original_image, secret_image_path, n)
    save_image('imgs\\ex5\\kitty_with_message.png', image_with_message)

    image = reveal_image(image_with_message, length, nbits=n)
    
    f, ar = plt.subplots(1,2)
    ar[0].imshow(original_image)
    ar[0].set_title("Original image")
    ar[1].imshow(image)
    ar[1].set_title("Hidden image")
    plt.show()
    return


def ex6():
    return






plt.rcParams["figure.figsize"] = (6, 4)

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
    elif opt == '6':
        ex6()
    elif opt == '0':
        break
    else:
        print('Wrong option!')

    print('\n\n\n')

 