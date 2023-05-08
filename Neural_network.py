import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv("hotel_bookings.csv")

data = data.drop(['Booking_ID','arrival_date'], axis=1)

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

data['type_of_meal_plan'] = le1.fit_transform(data['type_of_meal_plan'])
data['room_type_reserved'] = le2.fit_transform(data['room_type_reserved'])
data['market_segment_type'] = le3.fit_transform(data['market_segment_type'])
data['booking_status'] = le4.fit_transform(data['booking_status'])

train_data, test_data = train_test_split(data, test_size=0.2)

model = Sequential()
model.add(Dense(64, input_dim=train_data.shape[1]-1, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data.drop('booking_status', axis=1), train_data['booking_status'], epochs=50, batch_size=64, verbose=1)

loss, accuracy = model.evaluate(test_data.drop('booking_status', axis=1), test_data['booking_status'], verbose=0)
print(f"Accuracy on test data: {accuracy}")

new_data = pd.DataFrame({
    'no_of_adults': [2],
    'no_of_children': [1],
    'no_of_weekend_nights': [1],
    'no_of_week_nights': [2],
    'type_of_meal_plan': ['Meal Plan 1'],
    'required_car_parking_space': [0],
    'room_type_reserved': ['Room_Type 4'],
    'lead_time': [20],
    'arrival_year': [2016],
    'arrival_month': [6],
    'market_segment_type': ['Online'],
    'repeated_guest': [0],
    'no_of_previous_cancellations': [0],
    'no_of_previous_bookings_not_canceled': [1],
    'avg_price_per_room': [100],
    'no_of_special_requests': [0]
})
         
new_data['type_of_meal_plan'] = le1.transform(new_data['type_of_meal_plan'])
new_data['room_type_reserved'] = le2.transform(new_data['room_type_reserved'])
new_data['market_segment_type'] = le3.transform(new_data['market_segment_type'])
prediction = model.predict(new_data).argmax(axis=-1)
print(f"Prediction: {le4.inverse_transform(prediction)}")