import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# enter the path of dataset
model = load_model('entered the path where the trained dataset have.h5')

def predict_digit():
   
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (28, 28))
    input_data = resized_image.astype('float32') / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)
    prediction = model.predict(input_data)
    predicted_digit = np.argmax(prediction)
    result_label.config(text=f"Predicted digit: {predicted_digit}")
root = tk.Tk()
root.title("Handwritten Digit Recognition")
result_label = tk.Label(root, text="Predicted digit: ", font=("Helvetica", 20))
result_label.pack()


predict_button = tk.Button(root, text="Choose Image", command=predict_digit)
predict_button.pack()

root.mainloop()
