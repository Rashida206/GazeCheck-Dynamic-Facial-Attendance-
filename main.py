import face_recognition
import cv2
import os
import numpy as np
import csv
from datetime import datetime
from collections import defaultdict
import tkinter as tk

class FaceRecognitionAttendance:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.recorded_attendance = set()

        self.load_known_faces()
        self.top_student = self.get_top_student()

    def load_known_faces(self):
        known_faces_dir = "known_faces"
        images_path = os.listdir(known_faces_dir)

        for img_filename in images_path:
            img_path = os.path.join(known_faces_dir, img_filename)
            img = face_recognition.load_image_file(img_path)
            img_encoding = face_recognition.face_encodings(img)[0]

            name = os.path.splitext(img_filename)[0]
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(name)

    def get_top_student(self):
        attendance_data = self.load_attendance_data()
        leaderboard = self.generate_leaderboard(attendance_data)
        return leaderboard[0][0] if leaderboard else None

    def load_attendance_data(self):
        attendance_data = defaultdict(list)

        with open("sheet.csv", mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                student = row[0]
                timestamp = row[1]
                attendance_data[student].append(timestamp)

        return attendance_data

    def generate_leaderboard(self, attendance_data):
        leaderboard = sorted(attendance_data.items(), key=lambda x: len(x[1]), reverse=True)
        return leaderboard

    def update_leaderboard_csv(self):
        attendance_data = self.load_attendance_data()
        leaderboard = self.generate_leaderboard(attendance_data)

        with open("attendance_leaderboard.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Rank", "Student", "Total Attendance"])
            for rank, (student, attendance) in enumerate(leaderboard, start=1):
                writer.writerow([rank, student, len(attendance)])

    def recognize_faces(self):
        video_capture = cv2.VideoCapture(0)

        csv_filename = 'sheet.csv'
        if not os.path.exists(csv_filename):
            with open(csv_filename, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Name', 'Timestamp'])

        while True:
            ret, frame = video_capture.read()
            recognized_names, result_frame = self.detect_known_faces(frame)

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            with open(csv_filename, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)

                for name in recognized_names:
                    if name not in self.recorded_attendance:
                        csv_writer.writerow([name, current_time])
                        self.recorded_attendance.add(name)

                        self.update_leaderboard_csv()  # Update leaderboard after adding attendance

                        if name == self.top_student:
                            self.display_congratulatory_message(name)

            cv2.imshow('Video', result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_locations_original = [(top * (1 / self.frame_resizing), right * (1 / self.frame_resizing), bottom * (1 / self.frame_resizing), left * (1 / self.frame_resizing)) for (top, right, bottom, left) in face_locations]

        recognized_names = []
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations_original):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                recognized_names.append(name)

            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, name, (int(left) + 6, int(bottom) - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

        return recognized_names, frame

    def display_congratulatory_message(self, student_name):
        if student_name == self.top_student:
            window = tk.Tk()
            window.title("Congratulatory Message")

            message = f"Congratulations {student_name} for being the top attendee!"
            label = tk.Label(window, text=message, padx=20, pady=20)
            label.pack()

            window.mainloop()

if __name__ == "__main__":
    attendance_system = FaceRecognitionAttendance()
    attendance_system.recognize_faces()
