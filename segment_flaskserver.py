from flask import Flask
from segment import segment_routes

app = Flask(__name__)
app.register_blueprint(segment_routes, url_prefix='/api/v1')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)