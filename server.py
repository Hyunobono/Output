from flask import Flask, request, jsonify
from pipeline_plus import run_pipeline

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_video():
    try:
        data = request.get_json()
        youtube_url = data.get("youtube_url")
        if not youtube_url:
            return jsonify({"error": "YouTube URL이 필요합니다."}), 400

        result = run_pipeline(youtube_url)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "YouTube Summarizer API is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
