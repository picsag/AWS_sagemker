# Invoke API Gateway Endpoint
# This example shows how to invoke SageMaker Endpoint from outside of AWS environment using API Gateway
# Ref: https://stackoverflow.com/questions/17301938/making-a-request-to-a-restful-api-using-python

# Common Data Formats
# https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-inference.html

# Endpoint: XGBoost - Kaggle Bike Rental - Regressor Trained in XGBoost Lectures
# Makesure Endpoint is deployed before running this example


# In[ ]:


import requests
import json


# In[ ]:


# Update the URL to point to your API Gateway endpoint
url = 'https://ryfvzaxt7j.execute-api.us-east-1.amazonaws.com/beta'


# In[ ]:


# Raw Data
#datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed,casual,registered,count
# Actual=562
sample_one = ['2012-12-19 17:00:00',4,0,1,1,16.4,20.455,50,26.0027]
# Actual=569
sample_two = ['2012-12-19 18:00:00',4,0,1,1,15.58,19.695,50,23.9994]
# Actual=4
sample_three = ['2012-12-10 01:00:00',4,0,1,2,14.76,18.94,100,0]


# In[ ]:


# Single Observation
request = {
    "instances": [
        {
            "features": sample_one
        }
    ]
}


response = requests.post(url, data=json.dumps(request))
result = response.json()


if result['statusCode'] == 200:
    predictions = json.loads(result['body'])
    print('Predicted Count: ', predictions)
else:
    print('Error',result['statusCode'], result['body'])


# Multiple Observations
request = {
    "instances": [
        # First instance.
        {
            "features": sample_one
        },
        # Second instance.
        {
            "features": sample_two
        },
        # Third instance.
        {
            "features": sample_three
        }
    ]
}


# In[ ]:


response = requests.post(url, data=json.dumps(request))


result = response.json()


if result['statusCode'] == 200:
    predictions = json.loads(result['body'])
    print('Predicted Count: ', predictions)
else:
    print('Error',result['statusCode'], result['body'])


# In[ ]:





# In[ ]:




