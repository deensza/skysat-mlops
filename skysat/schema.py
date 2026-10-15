"""Shared data schema for SkySat.

One place that defines the columns, so the data preparation, the model,
the API and the simulator always agree on names and types.
"""

# Raw Kaggle column name -> clean snake_case name used everywhere in the project
RAW_TO_CLEAN = {
    "id": "passenger_id",
    "Gender": "gender",
    "Customer Type": "customer_type",
    "Age": "age",
    "Type of Travel": "type_of_travel",
    "Class": "travel_class",
    "Flight Distance": "flight_distance",
    "Inflight wifi service": "inflight_wifi_service",
    "Departure/Arrival time convenient": "departure_arrival_time_convenient",
    "Ease of Online booking": "ease_of_online_booking",
    "Gate location": "gate_location",
    "Food and drink": "food_and_drink",
    "Online boarding": "online_boarding",
    "Seat comfort": "seat_comfort",
    "Inflight entertainment": "inflight_entertainment",
    "On-board service": "on_board_service",
    "Leg room service": "leg_room_service",
    "Baggage handling": "baggage_handling",
    "Checkin service": "checkin_service",
    "Inflight service": "inflight_service",
    "Cleanliness": "cleanliness",
    "Departure Delay in Minutes": "departure_delay_minutes",
    "Arrival Delay in Minutes": "arrival_delay_minutes",
    "satisfaction": "satisfaction",
}

ID_COLUMN = "passenger_id"
TARGET_RAW = "satisfaction"
TARGET = "satisfied"  # 1 = satisfied, 0 = neutral or dissatisfied
POSITIVE_LABEL = "satisfied"

# Text columns and the values allowed in them (taken from the dataset)
CATEGORICAL_VALUES = {
    "gender": ["Male", "Female"],
    "customer_type": ["Loyal Customer", "disloyal Customer"],
    "type_of_travel": ["Personal Travel", "Business travel"],
    "travel_class": ["Business", "Eco", "Eco Plus"],
}
CATEGORICAL_FEATURES = list(CATEGORICAL_VALUES)

# 14 service ratings, 1-5 stars; 0 means "not rated / not applicable"
RATING_FEATURES = [
    "inflight_wifi_service",
    "departure_arrival_time_convenient",
    "ease_of_online_booking",
    "gate_location",
    "food_and_drink",
    "online_boarding",
    "seat_comfort",
    "inflight_entertainment",
    "on_board_service",
    "leg_room_service",
    "baggage_handling",
    "checkin_service",
    "inflight_service",
    "cleanliness",
]
RATING_MIN, RATING_MAX = 0, 5

NUMERIC_FEATURES = [
    "age",
    "flight_distance",
    "departure_delay_minutes",
    "arrival_delay_minutes",
]

# All model inputs, in a fixed order
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + RATING_FEATURES
