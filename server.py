from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    print(data['message'])  # Outputs: Hello from client!
    return jsonify({ 'message': 'Hello from server!' })

if __name__ == '__main__':
    app.run(port=5000)
