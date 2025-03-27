import json
import os

class Seq2SeqDataProcessor:
    """
    A class to process and manage seq2seq model data.
    """

    def __init__(self, input_dir, output_file):
        """
        Initializes the data processor with input directory and output file path.
        """
        self.input_dir = input_dir
        self.output_file = output_file

    def load_data(self):
        """
        Loads JSON files from the input directory and returns combined data.
        """
        combined_data = []
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.input_dir, filename), 'r') as file:
                    data = json.load(file)
                    combined_data.extend(data)
        return combined_data

    def save_data(self, data):
        """
        Saves processed data to the specified output file in JSON format.
        """
        with open(self.output_file, 'w') as file:
            json.dump(data, file, indent=4)

    def process_and_save(self):
        """
        Loads, processes, and saves the data.
        """
        data = self.load_data()
        processed_data = self._process_data(data)
        self.save_data(processed_data)

    def _process_data(self, data):
        """
        Processes the loaded data, filtering out empty entries.
        """
        return [entry for entry in data if entry]

if __name__ == '__main__':
    processor = Seq2SeqDataProcessor('input_data', 'output_data/processed_data.json')
    processor.process_and_save()
    print('Data processing complete. Output saved to:', processor.output_file)