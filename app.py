from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import sklearn

# from tensorflow.keras.models import load_model


# covid_model = load_model('models/covid.h5')
# braintumor_model = load_model('models/braintumor.h5')


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

############################################# BRAIN TUMOR FUNCTIONS ################################################
def preprocess_imgs(set_name):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img,dsize=(224,224),interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)



def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

##################################################################################################

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')

@app.route('/breastcancer')
def breast_cancer():
    return render_template('breastcancer.html')

@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')


@app.route('/resultc',methods=['POST'])
def resultc():
    if request.method=='POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img/255.0
            pred = covid_model.predict(img)
            if pred<0.5:
                pred = 0
            else:
                pred = 1
            return render_template('resultc.html', filename=filename,fn=firstname,ln=lastname,age=age,r=pred,gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultbt',methods=['POST'])
def resultbt():
    if request.method=='POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = crop_imgs([img])
            img = img.reshape(img.shape[1:])
            img = preprocess_imgs([img],(224,224))
            pred = braintumor_model.predict(img)
            if pred<0.5:
                pred = 0
            else:
                pred = 1
            return render_template('resultbt.html', filename=filename,fn=firstname,ln=lastname,age=age,r=pred,gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)



@app.route('/resultd',methods=['POST'])
def resultd():
    if request.method=='POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigree = request.form['diabetespedigree']
        age = request.form['age']
        skinthickness = request.form['skin']
        diabetes_model = pickle.load(open('models/diabetes.sav', 'rb'))
        pred = diabetes_model.predict([[pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,diabetespedigree,age]])
        return render_template('resultd.html',fn=firstname,ln=lastname,age=age,r=pred,gender=gender)



@app.route('/resultbc',methods=['POST'])
def resultbc():
    if request.method=='POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)

            return render_template('resultbc.html', filename=filename,fn=firstname,ln=lastname,age=age,r=0,gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)



@app.route('/resulta',methods=['GET','POST'])
def resulta():
    if request.method=='POST':
        print(request.url)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)

            return render_template('resulta.html', filename=filename,fn=firstname,ln=lastname,age=age,r=0,gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')



# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)