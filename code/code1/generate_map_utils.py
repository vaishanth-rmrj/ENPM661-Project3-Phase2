import cv2
import numpy as np

def draw_boundary(image):
  """
  function to draw the outer box boundary for the map

  Args:
      image (np.Array): numpy array of image pixel values

  Returns:
      numpy.Array: image with boundary drawn
  """
  image_copy = image.copy()

  #  bottom boundary
  for col in range(image_copy.shape[1]):
    image_copy[0, col] = 1

  # right boundary
  for row in range(image_copy.shape[0]):
    image_copy[row, 399] = 1

  # top boundary
  for col in range(image_copy.shape[1]):
    image_copy[image_copy.shape[0]-2, col] = 1

  # left boundary
  for row in range(image_copy.shape[0]):
    image_copy[row, 0] = 1

  return image_copy

def draw_circle(center, radius, image):
  """
  function to draw circle on the image 

  Args:
      center (tuple): center coordinate of the circle
      radius (int): radius of the circle
      image (np.array): numpy array of image pixel values

  Returns:
      numpy.Array: image with circle drawn
  """
  image_copy = image.copy()

  for col  in range(image_copy.shape[1]):
      for row in range(image_copy.shape[0]):
        if(((col-center[0])**2+(row-center[1])**2) < radius**2):
              image_copy[row,col]= 1.0

  return image_copy


def draw_rectangle(center, length, breadth, image):

  image_copy = image.copy()

  start_point = (int(center[0] - (length/2)), int(center[1] - (breadth/2)))  
  end_point = (int(center[0] + (length/2)), int(center[1] + (breadth/2)))  
    
  color = 1
    
  # Thickness of -1 will fill the entire shape
  thickness = -1
  image_copy = cv2.rectangle(image_copy, start_point, end_point, color, thickness)

  return image_copy

def colorize_image(image, color):
  """
  function to colorize the image

  Args:
      image (numpy.Array): numpy array of image pixel values
      color (tuple): (B, G, R) - color value

  Returns:
      numpy.Array: colourized image
  """
  image_copy = image.copy()
  color_img = np.full((image_copy.shape[0], image_copy.shape[1], 3), [241, 239, 236], dtype=np.uint8)
  for col  in range(image_copy.shape[1]):
      for row in range(image_copy.shape[0]):
        if image_copy[row,col]== 1.0:
            color_img[row,col] = color
  
  return color_img

def overlay_boundary(map_image, boundary_image, boundary_color):
  """
  function to overlay obstacle boundary on map image

  Args:
      map_image (numpy.Array): numpy array of map image pixel values
      boundary_image (numpy.Array): numpy array of boundary image pixel values
      boundary_color (tuple): (B, G, R) - boundary color value

  Returns:
      numpy.Array: colourized image along with obstacle boundary
  """

  map_image_copy = map_image.copy()
  for col  in range(boundary_image.shape[1]):
      for row in range(boundary_image.shape[0]):
        if boundary_image[row,col]== 1.0:
            map_image_copy[row,col] = boundary_color

  return map_image_copy

def generate_map(clearance):

  # numpy image canvas
  obstacle_map = np.zeros((1000, 1000))
  clearance_map = np.zeros((1000, 1000))

  # drawing obstacles
  circle1 = draw_circle((200, 200), 100, obstacle_map)
  rect1 = draw_rectangle((100, 500), 150, 150, obstacle_map)
  rect2 = draw_rectangle((500, 500), 250, 150, obstacle_map)
  rect3 = draw_rectangle((800, 300), 150, 200, obstacle_map)
  circle2 = draw_circle((200, 800), 100, obstacle_map)

  obstacle_map = cv2.bitwise_or(circle1, rect1)
  obstacle_map = cv2.bitwise_or(rect2, obstacle_map)
  obstacle_map = cv2.bitwise_or(rect3, obstacle_map)
  obstacle_map = cv2.bitwise_or(circle2, obstacle_map)
  obstacle_map =cv2.flip(obstacle_map, 0)
  obstacle_map = colorize_image(obstacle_map, [47, 47, 211])


  circle1_clearance = cv2.bitwise_xor(draw_circle((200, 200), 100, clearance_map), draw_circle((200, 200), 100 + clearance, clearance_map))
  circle2_clearance = cv2.bitwise_xor(draw_circle((200, 800), 100, clearance_map), draw_circle((200, 800), 100 + clearance, clearance_map))
  rect1_clearance = cv2.bitwise_xor(draw_rectangle((100, 500), 150, 150, clearance_map), draw_rectangle((100, 500), 150+ clearance, 150+ clearance, clearance_map))
  rect2_clearance = cv2.bitwise_xor(draw_rectangle((500, 500), 250, 150, clearance_map), draw_rectangle((500, 500), 250+ clearance, 150+ clearance, clearance_map))
  rect3_clearance = cv2.bitwise_xor(draw_rectangle((800, 300), 150, 200, clearance_map), draw_rectangle((800, 300), 150+ clearance, 200+ clearance, clearance_map))


  clearance_map = cv2.bitwise_or(circle1_clearance, circle2_clearance)
  clearance_map = cv2.bitwise_or(rect1_clearance, clearance_map)
  clearance_map = cv2.bitwise_or(rect2_clearance, clearance_map)
  clearance_map = cv2.bitwise_or(rect3_clearance, clearance_map)
  clearance_map =cv2.flip(clearance_map, 0)

  final_map_img = overlay_boundary(obstacle_map, clearance_map, [154, 154, 239])  
  return final_map_img

              
