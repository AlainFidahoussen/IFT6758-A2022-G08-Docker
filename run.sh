#!/bin/bash
docker run -p 127.0.0.1:8892:8892/tcp --env COMET_API_KEY=$COMET_API_KEY -it ift6758/serving:1.0.0                 
