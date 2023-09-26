from flask import Flask,render_template,request
from src.components.pipeline.forecast import forecast

data = {}
app=Flask(__name__)
@app.route('/',methods=['GET',"POST"])
def hello():
    global data
    if request.method == 'POST':
        days=request.form['days']
        obj=forecast()
        date,open_prices,low,high,close,volume=obj.forecasting_data(int(days))
        open_prices=[round(i,2) for i in open_prices]
        print(open_prices)
        low=[round(i,2) for i in low]
        high=[round(i,2) for i in high]
        close=[round(i,2) for i in close]
        volume=[round(i,2) for i in volume]
        data = {"date": [str(i) for i in date],
                "open": open_prices,
                "low": low,
                "high": high,
                "close": close
        }
        return render_template("index2.html",date=date,open=open_prices,low=low,high=high,close=close,volume=volume)
    return render_template('index.html')

@app.route("/forecast",methods=['POST',"GET"])
def forecasting():
    return render_template('index.html')

@app.route("/visualize",methods=["POST"])
def visualize():
    return render_template('index3.html', data=data)

if __name__=='__main__':
    app.run(debug=True)