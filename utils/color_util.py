import numpy as np
from colour import Color
import matplotlib


def interpolate_color(color1="red", color2="green", space=10):
    red = Color(color1)
    colors = list(red.range_to(Color(color2), space))
    return colors

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    print(RGB)
    RGB = [int(x*256) for x in RGB]
    ret = "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])
    if len(ret) > 7:
        ret = ret[0] + ret[2:]
    return ret

def hex_to_int(hex):
    return int(hex.replace('#', '0x', 1), 0)

def int_to_hex(int_val):
    return hex(int(int_val)).replace('0x', '#', 1)

def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[int(RGB[0]) for RGB in gradient],
      "g":[int(RGB[1]) for RGB in gradient],
      "b":[int(RGB[2]) for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
          int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
          for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)

def get_color_gradient(count):
    start = '#ff0000'
    end = '#0000ff'
    RGB_list = interpolate_color(start, end, count)

    rgb_array = np.zeros((count, 3))
    for i in range(count):
        rgb_array[i, :] = RGB_list[i].get_rgb()
    return rgb_array

def diffuse_color(A, C):
    """
    :param A: Adjacency matrix (N x N)
    :param C: Hex color list (N x 1)
    :return: C_new: (N x 1)
    """

    A = A + np.eye(len(A))
    A /= A.sum(axis=1)[:, np.newaxis]
    C_new = np.array(C)
    count = 20
    for i in range(count):
        C_new = np.dot(A, C_new) # Diffusion

    return C_new
