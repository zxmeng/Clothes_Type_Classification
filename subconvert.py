# import cv2
import numpy as np
import time
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import Image, ImageEnhance
import glob
import os

desired_size = 32
start_time = time.time()
# traintxt = open("fashion-data/train.txt", 'r')
# noisy = open("noisy.txt", 'r')

path = 'category_v5/trousers/'
label = "12"

count = 0
# for filename in glob.glob(os.path.join(path, '*.jpeg')):
for filename in os.listdir(path):
	# line = line.strip("\n")
	# print line
	# filename = "fashion-data/images/" + line + ".jpg"
	# img = cv2.imread(filename)
	img = Image.open(path + filename)

	contrast = ImageEnhance.Contrast(img)
	img = contrast.enhance(2)
	# sharp = ImageEnhance.Sharpness(img)
	# img = sharp.enhance(2)

	old_size = img.size
	# size = max(min_size, row, col)
	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])
	img = img.resize(new_size, Image.ANTIALIAS)
	# img = img.filter(ImageFilter.GaussianBlur(1))
	img = img.filter(ImageFilter.SHARPEN)

	fill_color = (255,255,255)
	new_im = Image.new("RGB", (desired_size, desired_size), fill_color)
	new_im.paste(img, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
	new_im = ImageOps.invert(new_im)
	# new_im = ImageOps.grayscale(new_im)
	# token = filename.split("/")
	# new_im.save("processed/" + line + ".jpg")
	new_im.save("color/" + label + "_" + filename)

	count += 1

	# if count > 10:
	# 	break
print count

print "--- %s seconds ---" % (time.time() - start_time)