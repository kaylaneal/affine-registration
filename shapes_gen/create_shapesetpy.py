from PIL import Image, ImageDraw
import random

# Generate Imageset:
for i in range(50):
	# create image and drawing ability
	img = Image.new("RGB", (256, 256), "black")
	draw = ImageDraw.Draw(img)

	# random number generation 
	pointA = (random.randint(10, 250), random.randint(10, 250))
	pointB = (random.randint(10, 250), random.randint(10, 250))
	pointC = (random.randint(10, 250), random.randint(10, 250))

	# draw random triangle
	draw.polygon((pointA, pointB, pointC), fill = 'white')
	img.save(f'pairs/target{i}.png')

	# random transformation
	theta = random.randint(0, 360)
	dx = random.randint(0, 5)
	dy = random.randint(0, 5)

	src = img.copy()
	src = src.rotate(angle = theta, translate = (dx, dy))
	src.save(f'pairs/move{i}_{theta}_{dx}_{dy}.png')
