import argparse
import scipy.misc
import numpy as np

def draw_checkerboard(px):
    im = np.zeros( (px, px), dtype=np.uint8)
    for row in range(px):
        for col in range(px):
            if (row + col) % 2 == 1:
                im[row,col]=255
    return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--px', type=int, default=16,
        help='image width')

    args = parser.parse_args()

    img = draw_checkerboard(args.px)

    image_fname = 'checkerboard%d.png' % args.px
    scipy.misc.imsave(image_fname,img)

    print "wrote", image_fname

