from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the pre-trained model
model = load_model("user_adapted_model.h5")  # Replace with your .h5 file path

def get_free_slots(events, day, time_slots=96):
    """
    Identifies free time slots based on the provided events for the specified day.
    """
    slot_duration = 24 * 60 // time_slots  # Slot duration in minutes
    slots = np.zeros(time_slots, dtype=bool)  # False means slot is free

    for event in events:
        event_start = datetime.fromisoformat(event["startDateTime"])
        event_end = datetime.fromisoformat(event["endDateTime"])

        if event_start.date() == day.date():
            start_slot = event_start.hour * 60 // slot_duration + event_start.minute // slot_duration
            end_slot = event_end.hour * 60 // slot_duration + event_end.minute // slot_duration
            slots[start_slot:end_slot] = True

    free_slots = [i for i, occupied in enumerate(slots) if not occupied]
    return free_slots

def process_data_with_model(data, events, day):
    """
    Process input data and return top 4 predicted time slots for the given event data.
    """
    # Prepare input data for the model
    duration = float(data.get("duration", 0)) / (24 * 60)
    input_array = np.array([[len(data.get("name", "")), len(data.get("description", "")), duration]])

    # Predict probabilities for each slot
    predictions = model.predict(input_array)[0]

    # Get available slots
    free_slots = get_free_slots(events, day)

    # Filter predictions based on available slots
    filtered_probs = predictions[free_slots]
    sorted_indices = np.argsort(filtered_probs)[-4:][::-1]  # Top 4 slots

    # Prepare results
    result = []
    slot_duration = 24 * 60 // len(predictions)
    for idx in sorted_indices:
        slot_index = free_slots[idx]
        start_minutes = slot_index * slot_duration
        end_minutes = start_minutes + float(data.get("duration", 0))
        start_time = day + timedelta(minutes=start_minutes)
        end_time = day + timedelta(minutes=end_minutes)
        result.append({
            "name": data["name"],
            "description": data["description"],
            "startDateTime": start_time.isoformat(),
            "endDateTime": end_time.isoformat(),
            "type": data["type"],
        })

    return result

@app.route('/api/events', methods=['POST'])
def receive_events():
    try:
        # Parse the JSON payload
        data = request.get_json()
        event_data = data.get("eventData")
        events = data.get("events")
        event_date = datetime.strptime(event_data["startDateTime"].split("T")[0], "%Y-%m-%d")

        # Process event data with the model
        processed_data = process_data_with_model(event_data, events, event_date)

        # Respond to the client
        response = {
            "message": "Predicted time slots successfully.",
            "predictedSlots": processed_data,
        }
        return jsonify(response), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"message": "An error occurred.", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
