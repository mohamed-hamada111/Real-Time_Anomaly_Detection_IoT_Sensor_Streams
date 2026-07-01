

# microsoft prebuild pyton image that contain azure functions
FROM mcr.microsoft.com/azure-functions/python:4-python3.11

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true



# make the folder that contain the dependencies
WORKDIR /home/site/wwwroot

# get the dependencies of the project 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the code of the project
COPY anomify_project/configs ./project/configs
COPY anomify_project/data  ./project/data
COPY anomify_project/models ./project/models
COPY anomify_project/pipelines ./project/pipelines
COPY anomify_project/src ./project/src

COPY function_app/host.json .
COPY function_app/function_app.py .


CMD ["python", "anomify_project/pipelines/inference.py"]