from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the .h5 model
model = load_model("model.h5")  # Replace with your .h5 file name

def process_data_with_model(data):
    """
    Process data using the .h5 model.
    The model is expected to return predictions in the desired format.
    """
    # Convert input data into a format the model can process
    input_array = np.array([[
        len(data.get("name", "")),
        len(data.get("description", "")),
        float(data.get("duration", 0)),
    ]])  # Example transformation

    # Predict using the model
    predictions = model.predict(input_array)

    # Transform predictions to the desired output format
    return [
        {
            "name": f"Predicted Event {i + 1}",
            "description": f"Description {i + 1}",
            "startDateTime": "2023-12-19T09:00:00",
            "endDateTime": "2023-12-19T10:00:00",
            "type": "Predicted Type",
        }
        for i in range(len(predictions))
    ]

@app.route('/api/events', methods=['POST'])
def receive_events():
    try:
        # Parse JSON payload
        data = request.get_json()
        event_data = data.get("eventData")
        events = data.get("events")

        # Process event data using the model
        processed_data = process_data_with_model(event_data)

        # Response to client
        response = {
            "message": "Event data processed successfully.",
            "processedData": processed_data,
        }
        return jsonify(response), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"message": "An error occurred.", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
