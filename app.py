from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ab import CodeGenerationModel
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize the model
model = CodeGenerationModel()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/generate', methods=['POST'])
def generate_code():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        generated_code = model.generate_code(prompt)
        return jsonify({'code': generated_code})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/complete', methods=['POST'])
def complete_code():
    try:
        data = request.get_json()
        code_prefix = data.get('code_prefix', '')
        completion = model.autocomplete_code(code_prefix)
        return jsonify({'completion': completion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/explain', methods=['POST'])
def explain_error():
    try:
        data = request.get_json()
        error_msg = data.get('error_message', '')
        code_context = data.get('code_context', '')
        explanation = model.explain_error(error_msg, code_context)
        return jsonify({'explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
