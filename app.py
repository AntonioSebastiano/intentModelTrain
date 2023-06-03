from Chat_test import chatbot_response
from flask import Flask,render_template,request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    #image = request.args.get('img')
    userText = request.args.get('msg')
    chatbot_res = chatbot_response(userText)
    #chatbot_res_image = chatbot_response(userText,image)
    #if type(chatbot_res) is tuple:
    #    return chatbot_res_image[0]
    #else:
    return chatbot_res

if __name__ == "main":
    app.run()
