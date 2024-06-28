import io
import PIL
import json
import keras
import numpy as np
import google.generativeai as genai

from flask import Flask, render_template, jsonify, request

genai.configure(api_key="AIzaSyDk4hQtIPZkmOnDNZJyxunug2Aj2OPqiaM")

app = Flask(__name__)

model = keras.models.load_model('./model/TFC-3.keras')

fruit_names = [
    "Alpukat",
    "Jambu",
    "Jeruk",
    "Lemon",
    "Nanas",
    "Pisang",
    "Salak",
    "Semangka",
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report/<int:id>')
def report(id):
    if id == 1:
        return render_template('report1.html')
    elif id == 2:
        return render_template('report2.html')
    else:
        return "Report not found", 404

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    img = PIL.Image.open(io.BytesIO(file.read())) # type: ignore
    img = img.resize((100, 100))
    img_arr = keras.preprocessing.image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr /= 255.0

    prediction = model.predict(img_arr) # type: ignore
    predicted_class = fruit_names[np.argmax(prediction)]
    confidence = prediction[0][np.argmax(prediction)]

    return json.dumps(
        {
            "predicted_class": predicted_class,
            "confidence": "{:.2f}".format(confidence * 100),
        },
        indent=4,
    )

@app.route('/generate', methods=['GET'])
def generate():
    fruit_name = request.args.get('fruit')
    if not fruit_name:
        return jsonify({'error': 'No fruit name provided'}), 400
    
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 500,
        "response_mime_type": "text/plain",
        }
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config, # type: ignore
        )

    try:
        response = model.generate_content([
            "input: Apa manfaat buah jeruk bagi kesehatan?",
            "output: Buah jeruk mengandung vitamin C yang tinggi, yang baik untuk sistem kekebalan tubuh dan membantu penyerapan zat besi.",
            "input: Mengapa buah pisang sering disarankan untuk dikonsumsi?",
            "output: Buah pisang kaya akan kalium, nutrisi penting untuk kesehatan jantung dan mengatur tekanan darah.",
            "input: Bagaimana buah apel dapat membantu menjaga kesehatan?",
            "output: Buah apel mengandung serat yang baik untuk pencernaan dan antioksidan seperti vitamin C yang melawan radikal bebas.",
            "input: Apa manfaat buah berry bagi tubuh?",
            "output: Buah berry, seperti blueberry dan strawberry, mengandung antioksidan tinggi yang dapat melindungi sel tubuh dari kerusakan dan meningkatkan kesehatan jantung.",
            "input: Apa yang membuat buah kiwi istimewa dari segi nutrisi?",
            "output: Buah kiwi mengandung vitamin C lebih banyak daripada jeruk, serta serat dan vitamin K yang baik untuk kesehatan tulang.",
            "input: Mengapa buah alpukat disebut sebagai sumber lemak sehat?",
            "output: Buah alpukat mengandung lemak tak jenuh tunggal yang baik untuk jantung dan mengurangi kadar kolesterol jahat (LDL).",
            "input: Bagaimana efek positif buah nanas terhadap pencernaan?",
            "output: Buah nanas mengandung enzim bromelain yang membantu dalam pencernaan protein dan mengurangi peradangan.",
            "input: Apa kandungan nutrisi dalam buah anggur?",
            "output: Buah anggur mengandung resveratrol, sebuah antioksidan yang dapat melindungi jantung dan mengurangi risiko penyakit kronis.",
            "input: Mengapa buah lemon sering disarankan untuk diminum airnya?",
            "output: Buah lemon mengandung vitamin C yang tinggi serta komponen sitrat yang dapat membantu meningkatkan sistem kekebalan tubuh dan menjaga keseimbangan pH.",
            "input: Apa peran buah plum dalam menjaga kesehatan tulang?",
            "output: Buah plum kaya akan vitamin K yang penting untuk pembekuan darah dan kesehatan tulang.",
            f"input: Apa kandungan nutrisi dalam buah {fruit_name} ?",
            "output: "
        ])
        fun_fact = response.text

        return jsonify({'fact': fun_fact})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
