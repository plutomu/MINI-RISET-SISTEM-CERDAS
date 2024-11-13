from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd


app = Flask(__name__)

"""
dic = {0 : 'Penyakit Herpes',  
       1 : 'Penyakit Kudis',
       2 : 'Penyakit Kurap', 
       3 : 'Penyakit Panu'}

dicDesc = {0 : "Herpes adalah penyakit menular seksual yang disebabkan oleh virus herpes simpleks (HSV). Ada dua jenis utama HSV, yaitu HSV-1 yang lebih sering menyebabkan luka dingin di sekitar mulut, dan HSV-2 yang umumnya menyebabkan herpes genital.",  
           1 : 'Kudis adalah penyakit kulit yang disebabkan oleh tungau kecil yang menggali dan bertelur di dalam kulit. Gatal yang hebat, terutama di malam hari, adalah gejala yang paling umum.',
           2 : 'Kurap adalah infeksi kulit yang disebabkan oleh jamur. Penyakit ini ditandai dengan ruam merah yang seringkali berbentuk lingkaran dan terasa gatal. Kurap bisa muncul di berbagai bagian tubuh, seperti kulit kepala, wajah, tangan, selangkangan, dan kaki.', 
           3 : 'Panu adalah infeksi kulit yang disebabkan oleh jamur alami yang ada di kulit kita. Jamur ini biasanya tidak menimbulkan masalah, namun pada kondisi tertentu seperti cuaca panas dan lembap, atau ketika sistem kekebalan tubuh kita lemah, jamur ini bisa tumbuh berlebih dan menyebabkan bercak-bercak pada kulit. Bercak-bercak ini bisa berwarna putih, merah muda, atau cokelat, tergantung pada warna kulit asli Anda.'}

dicLink = {0 : "https://www.halodoc.com/kesehatan/herpes?srsltid=AfmBOorZZ0Whytu9HfV84xnxQ4BMRnqbgvMzMApD7EFRVxKHlDkosqTd",  
           1 : 'https://www.halodoc.com/kesehatan/kudis?srsltid=AfmBOorZ29R6ndGv8PHVEsTONbrIkhifJSHd4VrqGyQ3UZZBFqOlk8aA',
           2 : 'https://www.halodoc.com/kesehatan/kurap?srsltid=AfmBOooMmBNxmH1Zax5Z0inwGbyx3z4UeYhPmDURyrdvSiz41oJoDixv', 
           3 : 'https://www.halodoc.com/kesehatan/panu?srsltid=AfmBOopQVUoWdJ29eArp7CA0rJzVifU84jQFEnamBOBYn-UW8pwhOgII'}
"""
# Import CSV
dataCSV = pd.read_csv("data.csv")

dic = dataCSV["jenis"]
dicDesc = dataCSV["pengertian"]
dicAtasi = dataCSV["penanganan"]
dicLink = dataCSV["link"]


model = load_model('Cendekia-Penyakit Kulit-75.0.h5')
model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)
	predict_x = model.predict(i)
	#predict_x = model.predict_proba(i)
	p = np.argmax(predict_x,axis=1)
	#p = model.predict_classes(i)
	return dic[p[0]], dicDesc[p[0]], dicLink[p[0]], dicAtasi[p[0]]

def read_text(path):
    file = open(path, "r")
    content = file.read()
    file.close()
    return content
    


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("classification.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p, d, l, s = predict_label(img_path)
	return render_template("classification.html", prediction = p, description = d, link = l, penanganan = read_text(s), img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
