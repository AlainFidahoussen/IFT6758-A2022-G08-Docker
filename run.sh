#!/bin/bash
docker run -p 127.0.0.1:5000:5000/tcp --env COMET_API_KEY=$COMET_API_KEY -it ift6758/serving:1.0.0                 
