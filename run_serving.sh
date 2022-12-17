#!/bin/bash
cd serving
gunicorn --bind 0.0.0.0:5000 app:app
cd ..
