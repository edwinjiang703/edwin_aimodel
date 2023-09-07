import openai
import json
import requests

openai.organization = "org-Q2q2eUsNbWG6pxm1dbt2N5uW"
openai.api_key = "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"

#print(openai.Model.list())
#openai.Model.retrieve))

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    # """Get the current weather in a given location"""
    # weather_info = {
    #     "location": location,
    #     "temperature": "72",
    #     "unit": unit,
    #     "forecast": ["sunny", "windy"],
    # }
    # return json.dumps(weather_info)
    url= "http://api.openweathermap.org/data/2.5/forecast?q=shanghai,CN&APPID=917c9f8e898e0ecbd6985e1744d93e9f"
    r = requests.get(url)  
    j = r.json()['list']    # 返回40条， j[0] 查看第一条数据       
         
    i=5
    print('\nDay', int((i-2)/8) )   
    print('预报时刻：' , j[i]['dt_txt'])
    print('温度    ：' , '%.2f'%(j[i]['main']['temp'] -273.15))
    print('湿度    ：' , j[i]['main']['humidity'])
    print('气压    ：' , j[i]['main']['pressure'])
    print('天气    ：' , j[i]['weather'][0]['description'])
    print('风速    ：' , j[i]['wind']['speed'])
    print('能见度  ：' , j[i]['visibility'])

    

def run_conversation():
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    print(response_message)

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        
        fuction_to_call = available_functions[function_name]
        print('fuction_to_call %s',fuction_to_call)
        function_args = json.loads(response_message["function_call"]["arguments"])
        print('function_args %s',function_args.get("location"))
        function_response = fuction_to_call(
            location=function_args.get("location")
            #unit=function_args.get("unit"),
        )
        

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": '',
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response


print(run_conversation())