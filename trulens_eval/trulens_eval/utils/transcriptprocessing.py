import difflib
from enum import Enum
import os
import pandas as pd
import chardet


# Define the Role Enum
class Role(Enum):
    MARKETER = ("marketer", ["I’m with ABCO health", "promotion", "offer"])
    PATIENT = ("patient", [])
    TELE_MEDICINE_OPERATOR = ("telemedicine operator", [])
    PRE_RECORDED_MESSAGE = ("pre-recorded message", ["You are caller number zero in our queue", 
                 "Thank you for calling my doctor's live. This call is being recorded for quality control and compliance purposes.",
                 "Did you know that according to the American Medical Association…"])
    DOWNLINE = ("downline", [])
    UNKNOWN = ("unknown", [])

    def __init__(self, label, phrases):
        self.label = label
        self.phrases = phrases

    def get_label(self):
        return self.label

    def get_phrases(self):
        return self.phrases

# PhoneCall class
class PhoneCall:
    def __init__(self, transcript_text):
        self.transcript = []  # List to store the structured transcript.
        self.roles = {}  # Dictionary to store roles.
        self._load_transcript(transcript_text)
        self.assign_pre_recorded_message()
        self.split_transcript_by_prerecorded()
        self.assign_marketer()
        self.assign_patient()
        self.assign_telemedicine_operator()

    def _load_transcript(self, transcript_text):
        """Parse the transcript text and load it into the structured transcript list."""
        current_speaker = None
        for line in transcript_text.splitlines():
            line = line.strip()
            if line.startswith("SPEAKER"):
                current_speaker = line
            elif current_speaker:
                entry = (current_speaker, self.roles.get(current_speaker, Role.UNKNOWN.get_label()), line)
                self.transcript.append(entry)

    def split_transcript_by_prerecorded(self):
        """Split the transcript by the first occurrence of a pre-recorded message."""
        for i, (_, role, _) in enumerate(self.transcript):
            if role == Role.PRE_RECORDED_MESSAGE.get_label():
                break
        else:
            i = len(self.transcript)  # No pre-recorded message found, use the full transcript length.

        self.transcript_before_prerecorded = self.transcript[:i]
        self.transcript_after_prerecorded = self.transcript[i:]


    def set_role(self, speaker, role):
        """Assign a role to a speaker."""
        if role in [r.get_label() for r in Role]:
            self.roles[speaker] = role
            # Update roles in the transcript only for the specified speaker, and only if the existing role is "unknown"
            self.transcript = [(spk, role if spk == speaker and current_role == Role.UNKNOWN.get_label() else current_role, line) for spk, current_role, line in self.transcript]
        else:
            raise ValueError(f"Invalid role '{role}'.")

    def get_role(self, speaker):
        """Retrieve the role assigned to a speaker."""
        return self.roles.get(speaker, Role.UNKNOWN.get_label())

    def get_lines_by_speaker(self, speaker):
        """Retrieve all sentences/lines spoken by a particular speaker."""
        return [line for spk, _, line in self.transcript if spk == speaker]

    def get_lines_by_role(self, role):
        """Retrieve all sentences/lines spoken by speakers with a specified role."""
        return [line for _, assigned_role, line in self.transcript if assigned_role == role]

    def find_speaker_by_phrase(self, phrase):
        """Find a speaker based on a particular phrase or word they said."""
        for spk, _, line in self.transcript:
            if phrase in line:
                return spk
        return None

    def formatted_transcript(self):
        """Return the transcript with speaker roles and lines formatted as described."""
        formatted_output = []
        for spk, role, line in self.transcript:
            role_or_speaker = role if role else spk
            formatted_output.append(f"{role_or_speaker.capitalize()}: {line}\n")
        return "\n".join(formatted_output)
    
    def compare_transcript(self):
        """Return the transcript with speaker roles and lines formatted as described."""
        formatted_output = []
        for spk, role, line in self.transcript:
            role_or_speaker = role if role else spk
            formatted_output.append(f"{spk} {role}: {line}\n")
        return "\n".join(formatted_output)

    def assign_pre_recorded_message(self):
        """Assign the role 'pre-recorded message' to lines similar to known pre-recorded messages."""
        known_messages = Role.PRE_RECORDED_MESSAGE.get_phrases()
    
        # Iterate through each line in the transcript
        for idx, (spk, role, line) in enumerate(self.transcript):
            for message in known_messages:
                matcher = difflib.SequenceMatcher(None, line, message)
                _, _, length = max(matcher.get_matching_blocks(), key=lambda x: x[2])
                matching_subseq = line[:length]
                similarity = difflib.SequenceMatcher(None, matching_subseq, message).ratio()
                if similarity > 0.8:
                    # Update only the specific line's role to "pre-recorded message"
                    self.transcript[idx] = (spk, Role.PRE_RECORDED_MESSAGE.get_label(), line)
                    break

    def assign_marketer(self):
    # Count the number of questions asked by each speaker before the first pre-recorded message
        question_counts = {}
        for speaker, _, line in self.transcript_before_prerecorded:
            question_counts[speaker] = question_counts.get(speaker, 0) + line.count("?")

        # Identify the speaker who asked the most questions
        max_questions = max(question_counts.values())
        marketer_speaker = [speaker for speaker, count in question_counts.items() if count == max_questions][0]

        # Assign the "marketer" role to all lines spoken by that speaker
        self.set_role(marketer_speaker, "marketer")

    def assign_patient(self):
        """Assign the role 'patient' to the speaker with the most lines 
        before the first pre-recorded message, excluding the marketer."""
        
        # Count the number of lines spoken by each speaker before the first pre-recorded message
        line_counts = {}
        for speaker, _, _ in self.transcript_before_prerecorded:
            if self.roles.get(speaker) != Role.MARKETER.get_label():  # Exclude the marketer
                line_counts[speaker] = line_counts.get(speaker, 0) + 1

        # Identify the speaker who spoke the most lines
        max_lines = max(line_counts.values(), default=0)
        patient_speaker = [speaker for speaker, count in line_counts.items() if count == max_lines][0]

        # Assign the "patient" role to that speaker
        self.set_role(patient_speaker, Role.PATIENT.get_label())

    def assign_telemedicine_operator(self):
        # Count the number of questions asked by each speaker after the last pre-recorded message
        question_counts = {}
        for speaker, _, line in self.transcript_after_prerecorded:
            question_counts[speaker] = question_counts.get(speaker, 0) + line.count("?")

        # Identify the speaker who asked the most questions
        max_questions = max(question_counts.values(), default=0)
        telemedicine_speaker = next((speaker for speaker, count in question_counts.items() if count == max_questions), None)

        # Assign the "telemedicine operator" role to all lines spoken by that speaker in transcript_after_prerecorded
        # without overwriting lines labeled as "pre-recorded message"
        for idx, (spk, role, line) in enumerate(self.transcript_after_prerecorded):
            if spk == telemedicine_speaker and role != Role.PRE_RECORDED_MESSAGE.get_label():
                self.transcript_after_prerecorded[idx] = (spk, Role.TELE_MEDICINE_OPERATOR.get_label(), line)
                # Update roles in the main transcript based on changes in transcript_after_prerecorded
                main_idx = self.transcript.index((spk, role, line))
                self.transcript[main_idx] = (spk, Role.TELE_MEDICINE_OPERATOR.get_label(), line)
                self.roles[spk] = Role.TELE_MEDICINE_OPERATOR.get_label()


true_values_path = "/Users/alec/Documents/Documents - Alec’s MacBook Pro/trulens_testing/historic_data/cleaned/true_scores.csv"
df_truth = pd.read_csv(true_values_path, dtype={'Answer': bool, 'Eureka ID': str, 'Question':str})

# Initialize a dictionary to store errors
# The keys will be file names, and the values will be error messages
error_dict = {}

# iterate over files in directory
def label_transcript(file_path, question_number):
    # extract file name without extension
    eureka_id = os.path.basename(file_path).split('.')[0]
    print(f"eureka_id: {eureka_id}")
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())

        # Open and read file
        with open(file_path, 'r', encoding=result['encoding']) as file:
            text = file.read()
        
        # Filter rows based on 'Eureka ID' and 'Question'
        filtered_rows = df_truth[(df_truth['Eureka ID'] == eureka_id) & (df_truth['Question'] == question_number)]

        # Check if any matching row is found
        if filtered_rows.empty:
            raise ValueError(f'No answer found for Eureka ID: {eureka_id} and Question: {question_number}')
        
        # Get the 'Answer' value
        answer = bool(filtered_rows['Answer'].values[0])
        print(f"found answer for {eureka_id} {question_number}: {answer}")
        # Replace Unicode characters with spaces
        text = text.replace('\u00ff', ' ')
        text = text.replace('\u00a0', ' ')

        phone_call = PhoneCall(text)
        print("transcript processed")
        labeled_transcript = phone_call.formatted_transcript()
        print(f"Transcript {eureka_id} has been labeled, now calling the chain...")

        return labeled_transcript, answer, eureka_id
    
    except Exception as e:
        # Store the error message and the corresponding file name
        error_dict[file_path] = str(e)
        print(f'Error processing file {file_path}: {e}')