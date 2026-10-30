import numpy as np
import subprocess
import json
import tempfile


def execute_vggt(image_names, root_dir):
    # temporary file to store the result
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmpfile:
        output_path = tmpfile.name
    try: 
        # call vggt_runner.py in env2
        results = subprocess.run([
            "../venv/vggt/bin/python",
            f"{root_dir}/src/loader/vggt_runner.py", 
            json.dumps(image_names),  # pass input as JSON string
            output_path
        ], capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)

    # load result
    T_targetW_sourceFP = np.load(output_path)
    return T_targetW_sourceFP
