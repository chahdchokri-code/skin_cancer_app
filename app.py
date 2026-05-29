from flask import Flask, render_template, request, redirect, session, flash, jsonify
import os
import numpy as np

app = Flask(__name__)
app.secret_key = "secret"

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chargement du modèle (optionnel)
model = None
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image
    model = load_model("model/vgg16_malignant_benign.h5")
    print("✅ Modèle IA chargé avec succès")
except Exception as e:
    print(f"⚠️  Modèle IA non disponible : {e}")
    print("⚠️  L'application démarre en mode démo (sans prédiction réelle)")

# Connexion MySQL (optionnelle)
db = None
cursor = None
patients_demo = [
    {"id": 1, "name": "Ahmed Ben Ali", "age": 45, "result": "Benign", "probability": 0.12, "image_path": "", "created_at": "2026-05-20 10:30:00"},
    {"id": 2, "name": "Sonia Ouaghlani", "age": 32, "result": "Malignant", "probability": 0.87, "image_path": "", "created_at": "2026-05-21 14:15:00"},
    {"id": 3, "name": "Mohamed Trabelsi", "age": 58, "result": "Benign", "probability": 0.08, "image_path": "", "created_at": "2026-05-22 09:00:00"},
]
try:
    import mysql.connector
    db = mysql.connector.connect(
        host="localhost", user="root", password="", database="skin_cancer_db"
    )
    cursor = db.cursor(dictionary=True)
    print("✅ Base de données connectée")
except Exception as e:
    print(f"⚠️  MySQL non disponible : {e}")
    print("⚠️  Mode démo activé (données fictives)")


def get_patients():
    if cursor:
        cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
        return cursor.fetchall()
    return patients_demo


def get_stats():
    pts = get_patients()
    total = len(pts)
    benign = sum(1 for p in pts if p["result"] == "Benign")
    malignant = sum(1 for p in pts if p["result"] == "Malignant")
    return total, benign, malignant


# LOGIN
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pwd = request.form["password"]

        if cursor:
            cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (user, pwd))
            result = cursor.fetchone()
            if result:
                session["user"] = user
                flash("Login réussi ✔", "success")
                return redirect("/dashboard")
            else:
                flash("Identifiants incorrects ✗", "danger")
        else:
            # Mode démo : accepte n'importe quel login
            if user and pwd:
                session["user"] = user
                flash("Login réussi ✔ (mode démo)", "success")
                return redirect("/dashboard")
            else:
                flash("Veuillez remplir tous les champs", "danger")

    return render_template("login.html")


# DASHBOARD
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    total, benign, malignant = get_stats()
    return render_template("dashboard.html", total=total, benign=benign, malignant=malignant)


# PREDICT
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        try:
            name = request.form["name"]
            age = request.form["age"]
            file = request.files["image"]

            if file.filename == "":
                flash("Veuillez choisir une image", "warning")
                return redirect("/predict")

            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            if model:
                from tensorflow.keras.preprocessing import image as keras_image
                img = keras_image.load_img(path, target_size=(224, 224))
                img = keras_image.img_to_array(img) / 255.0
                img = np.expand_dims(img, axis=0)
                pred = float(model.predict(img)[0][0])
            else:
                # Mode démo : résultat aléatoire
                import random
                pred = random.uniform(0.1, 0.9)

            result_label = "Malignant" if pred > 0.5 else "Benign"

            if cursor and db:
                cursor.execute("""
                    INSERT INTO patients (name, age, result, probability, image_path)
                    VALUES (%s, %s, %s, %s, %s)
                """, (name, age, result_label, pred, path))
                db.commit()
            else:
                patients_demo.insert(0, {
                    "id": len(patients_demo) + 1,
                    "name": name, "age": age,
                    "result": result_label,
                    "probability": pred,
                    "image_path": path,
                    "created_at": "2026-05-23 12:00:00"
                })

            flash("Analyse réussie ✔", "success")
            return render_template("result.html",
                                   result=result_label,
                                   prob=round(pred * 100, 2),
                                   img=path)

        except Exception as e:
            flash(f"Erreur : {str(e)}", "danger")
            return redirect("/predict")

    return render_template("predict.html")


# PATIENTS
@app.route("/patients")
def patients():
    if "user" not in session:
        return redirect("/")
    data = get_patients()
    return render_template("patients.html", patients=data)


# STATS API
@app.route("/api/stats")
def api_stats():
    total, benign, malignant = get_stats()
    return jsonify({"total": total, "benign": benign, "malignant": malignant})


# LOGOUT
@app.route("/logout")
def logout():
    session.clear()
    flash("Déconnecté", "info")
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
