import os, sys, imghdr
from os.path import isfile, exists
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from numpy import asarray, resize

#hide keras / tensorflow errors, warnings, and info in import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import load_model as lm
sys.stderr = stderr

model = lm("id_char.h5")
file_formats = ['png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff']

def check(path):
    return (exists(path) and isfile(path) and imghdr.what(path) in file_formats)

if(len(sys.argv) > 1 and check(sys.argv[1])):
    file = sys.argv[1]
elif(len(sys.argv) > 1 and isfile(sys.argv[1])):
    print('Sorry, the file format ' + imghdr.what(sys.argv[1]) +
          ' is not compatible with this prediction program.')
    print('Please try converting this image to an acceptable format.')
    sys.exit(0)
else:
    while(True):
        print("Unable to identify image filepath in the command line.")
        file = input("Please provide the filepath for the image file: ")
        if(check(file)):
            break
        elif(isfile(file)):
            print('Sorry, the file format ' + imghdr.what(file) +
                  ' is not compatible with this prediction program.')
            print('Please try converting this image to an acceptable format.')
            sys.exit(0)
        elif(exists(file)):
            print('Sorry, this feature does not exist yet.')
            # Feature = running for all images in a directory
        elif(file=='exit'):
            print('Goodbye!')
            sys.exit(0)

test_img = asarray(ImageOps.invert(Image.open(file).convert('L').resize((28, 28)))).astype('float32') / 255
pred = model.predict(test_img.reshape(1,28,28,1)).argmax()
print(chr(pred+64) + " (" + chr(pred+96) + ")")

if('y' in input("Would you like to visually confirm this answer? (y/n) ")):
    plt.imshow(test_img.reshape(28,28), cmap='Greys')
    plt.show()
