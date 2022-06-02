from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import cv2
from werkzeug.utils import secure_filename
import numpy as np
import image_similarity_measures
import face_recognition

app = Flask(__name__)


@app.route("/")             #HomePage
def uploader():
    
    path = 'static/uploads/'
    uploads = sorted(os.listdir(path), key=lambda x: os.path.getctime(
        path+x))        # Sorting as per image upload date and time
    print(uploads)
    
    uploads = ['uploads/' + file for file in uploads]
    uploads.reverse()
    
    # Pass filenames to front end for display in 'uploads' variable
    return render_template("index.html", uploads=uploads)


app.config['UPLOAD_FOLDER'] = 'static/uploads'             # Storage path


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route("/upload", methods=['GET', 'POST'])
def upload_file():                                       # This method is used to upload files
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Redirect to route '/' for displaying images on front end
        return redirect("/")


#capture image

@app.route('/static/uploads/')
def my_link():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            
            if key == ord('s'): 
                cv2.waitKey(1000)
                print("Processing image...")

                path = '/home/falcon/Documents/Office Docs/AutoKYC-POC/static/uploads'
                cv2.imwrite(os.path.join(path , 'frame.jpg'), img=frame)
                print("Processing image...")
                img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
        
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
        
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
            
    return redirect("/")

@app.route('/verified/')

def verify():
    def face_compare(path1, path2):

        img = cv2.imread(path1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_encoding = face_recognition.face_encodings(rgb_img, model="Large")[0]

        img2 = cv2.imread(path2)
        rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img_encoding2 = face_recognition.face_encodings(rgb_img2, model="Large")[0]

        result = face_recognition.compare_faces([img_encoding], img_encoding2, tolerance = 0.5)
        return result


    all_files = os.listdir('/home/falcon/Documents/Office Docs/AutoKYC-POC/static/uploads')
    picture_files = []
    
    for file in all_files:
        if file[-3:] == "png" or file[-4:] == "jpeg" or file[-3:] == "jpg":
            picture_files.append(file)
    
    i=0
    j=1
    
    input_1 = './static/uploads/' + picture_files[i]
    input_2 = './static/uploads/' + picture_files[j]
    outputs = face_compare(input_1,input_2)

    print(picture_files[i] + '  ', picture_files[j] + ' ', outputs)

    if outputs ==[True]:
        return render_template('verify.html')
            
    else:
        return render_template('not_verified.html')
        
    return redirect("/")

if __name__ == "__main__":
    app.debug = True
    app.run()