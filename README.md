## Environment variable

To build and run the dockers, you must define your `COMET_API_KEY` environment variable first, by:
 - exporting it on the bash prompt: 
```bash
 export COMET_API_KEY=<your key>
```
 - or defining it in a .env file placed on the root directory.

## Run the NHL App without Docker
In the root folder, run the following commands:
```bash
pip install -r requirements.txt
./run_serving.sh
```

In the root folder, run the following commands:
```bash
pip install -r requirements.txt
export COMET_API_KEY=<your key>
./run_app.sh
```

The serving log file is available at the address `http://localhost:5000`.

The streamlit service (NHL App) is available at the address `http://localhost:5001`.

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

The serving log file is available at the address `http://localhost:5000`.

The streamlit service (NHL App) is available at the address `http://localhost:5001`.



