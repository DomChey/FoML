from skimage import io, color


# crops piece out of given image
def crop(infile, height, width):
    im = io.imread(infile)
    imgwidth, imgheight = im.shape[0], im.shape[1]
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            yield im[(i*height):((i+1)*height), (j*width):((j+1)*width), :]


# cuts given image into pieces and returns list containing the pieces
# pieces are in LAB color space
def cutIntoPieces(infile, height, width):
    start_num = 0
    pieces = []
    for k, piece in enumerate(crop(infile, height, width), start_num):
        img = color.rgb2lab(piece)
        pieces.append(img)
    return pieces
