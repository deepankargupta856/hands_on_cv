import cv2

img = cv2.imread('image.png')

cv2.imshow('image: ',img)
cv2.waitKey(0)

# cv2.imwrite('saved_img.png',img)

height, width, channels = img.shape  # shape returns (height, width, channels)
print(f"Width: {width}, Height: {height}, Channels: {channels}")

new_height = int(height*1.5)
new_width = int(width*1.5)
# resizing
resized = cv2.resize(img,(new_width,new_height),interpolation=cv2.INTER_CUBIC)

height, width, channels = resized.shape  # shape returns (height, width, channels)
print(f"Width: {width}, Height: {height}, Channels: {channels}")


#blurring 

blurred = cv2.GaussianBlur(img,(5,5),0) # 5 - blur intensity

# edge detection

edge = cv2.Canny(img,100,200) # arguments are gradient of pixel 

cv2.imshow("Resized", resized)
cv2.imshow("blurred", blurred)
cv2.imshow("edge", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()