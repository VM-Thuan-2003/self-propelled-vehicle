import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('./img/img_0.jpg')

print(image[350,340])

# Display the image
plt.imshow(image)
plt.title("Input Image")
plt.show()