from flask import Flask, request, render_template, jsonify
from rag_backend import initialize_rag, search_knowledge

app = Flask(__name__)

initialize_rag("faq.txt")   # این خط در سطح ماژول باشد

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'سوالی ارسال نشده'}), 400
        answer = search_knowledge(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)