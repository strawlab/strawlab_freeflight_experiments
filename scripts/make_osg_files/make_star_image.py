import random
import argparse
import numpy as np
import PIL.Image, PIL.ImageDraw

def draw_image(W,S,N,do_random):
    im = PIL.Image.new("L",(W,W), "black")
    draw = PIL.ImageDraw.Draw(im)

    if not do_random:
        pts = np.linspace(0,W*W,num=N,endpoint=False).astype(np.int)
        xs,ys = np.unravel_index(pts, (W,W))

    for i in range(N):
        if do_random:
            x,y = random.randint(0,W-1), random.randint(0,W-1)
        else:
            x,y = xs[i],ys[i]

        draw.ellipse(
                (np.clip(x-S,0,W-1),np.clip(y-S,0,W-1),
                 np.clip(x+S,0,W-1),np.clip(y+S,0,W-1)),
                fill="white")

    im.save("stars%dx%d%s.png" % (S,N,"r" if do_random else "u"), "PNG")
    del im

if __name__ == "__main__":
    W = 512
    S = 4
    N = 200

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--width', type=int, default=512,
        help='image width')
    parser.add_argument(
        '--star-size', type=int, default=4,
        help='size of stars')
    parser.add_argument(
        '--num-stars', type=int, default=200,
        help='number of stars')
    parser.add_argument(
        '--uniform', action='store_true', default=False,
        help='uniformly arrange stars in a grid instead of random')

    args = parser.parse_args()    

    draw_image(
        args.width,
        args.star_size,
        args.num_stars,
        not args.uniform)
