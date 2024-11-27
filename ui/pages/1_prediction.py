from utils.config import get_config
from tools.data_execution_metadata import DataExecutionMetadata
from ui.predict_ui import PredictUI

# Instantiate the prediction UI class
gui = PredictUI()

# set the input and output fields based on the data_execution_metadata_file file
data_execution_metadata_file = DataExecutionMetadata(get_config("job", "data_execution_metadata_file"))
gui.set_input_fields(data_execution_metadata_file.job_metadata_inputs,
                     data_execution_metadata_file.job_metadata_inputs_details)
gui.set_output_fields(data_execution_metadata_file.job_metadata_outputs)

# load the session state (saved user fields) from a file
gui.load_session_state()

# Show the UI
gui.show()

