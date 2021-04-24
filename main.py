from flask import Flask, render_template, request
import pickle
app = Flask(__name__)

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def method_name():
   if request.method == "POST":
      myDict = request.form
      fever = float(myDict['fever'])
      age = int(myDict['age'])
      bodyPain = int(myDict['bodyPain'])
      runnyNose = int(myDict['runnyNose'])
      diffBreath = int(myDict['diffBreath'])

      input = [fever, bodyPain, age, runnyNose, diffBreath]
      inf = clf.predict_proba([input])[0][1]

      print(inf*100)
      return render_template('show.html', inf=round(inf))
   return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)