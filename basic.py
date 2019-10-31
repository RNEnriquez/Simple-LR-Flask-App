from flask import Flask, render_template, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("admissions_data.csv")

X = data.drop(["Serial No.", "Chance of Admit "], axis=1)
y = data["Chance of Admit "]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression().fit(X_train, y_train)

app = Flask(__name__)

app.config["SECRET_KEY"] = "mysecretkey"

class InfroForm(FlaskForm):

    gre_score = StringField("What is your GRE Score?")
    toefl_score = StringField("What is your TOEFL Score?")
    university_rating = StringField("What is the rating of the University?")
    sop = StringField("What is the SOP?")
    lor = StringField("What is your LOR score?")
    cgpa = StringField("What is your CGPA?")
    research = StringField("Have you done research?")
    submit = SubmitField("Submit")

@app.route("/", methods=["GET", "POST"])
def index():

    form = InfroForm()
    if form.validate_on_submit():
        session["gre_score"] = form.gre_score.data
        session["toefl_score"] = form.toefl_score.data
        session["university_rating"] = form.university_rating.data
        session["sop"] = form.sop.data
        session["lor"] = form.lor.data
        session["cgpa"] = form.cgpa.data
        session["research"] = form.research.data

        return redirect(url_for("thankyou"))

    return render_template("home.html", form=form)

@app.route("/thankyou")
def thankyou():
    prediction = model.predict(np.array([[float(session["gre_score"]), float(session["toefl_score"]),
                                         float(session["university_rating"]), float(session["sop"]),
                                         float(session["lor"]), float(session["cgpa"]),
                                         float(session["research"])]]))

    return render_template("thankyou.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
