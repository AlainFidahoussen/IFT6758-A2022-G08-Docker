## Environment variable

To build and run the dockers, you must define your `COMET_API_KEY` environment variable first, by:
 - exporting it on the bash prompt: 
```bash
 export COMET_API_KEY=<your key>
```
 - or defining it in a .env file placed on the root directory.

## Run the NHL App without Docker
In the `serving` folder, create a virtual python environment and run the gunicorn command:
```bash
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:8890 app:app
```

In the `streamlit` folder, create a virtual python environment and run the streamlit command:
```bash
pip install -r requirements.txt
export COMET_API_KEY=<your key>
streamlit run app.py --server.port=8892 --server.address=0.0.0.0
```

The serving log file is available at the address `http://localhost:8890`.

The streamlit service is available at the address `http://localhost:8892`.

## Run the NHL App with Docker
The dockers only work on Linux for now (or WSL on Windows).

To build and run the docker serving, the build.sh and run.sh scripts can be used:
```bash
./build.sh
./run.sh
```

To build both the streamlit and the serving dockers, run the following command from the root directory:
```bash
docker-compose up
```

Make sure that your `COMET_API_KEY` has been defined before, as mentionned above. You can check that by running the following command:
```bash
docker-compose config
```

The serving log file is available at the address `http://localhost:8890`.

The streamlit service is available at the address `http://localhost:8892`.



