import boto3
import botocore.config
import json
from datetime import datetime

# function for generate blog on specific topic
def blog_generate_using_bedrock(blogtopic:str)-> str:
    prompt= f"""<s>[INST]HUMAN: write a 200 words blog on the topic {blogtopic}
    Assistant:[/INST]
    """
    body ={
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p ":0.9
        
    }
    try:
        # config bedrock client
        bedrock=boto3.client(service_name="bedrock-runtime", region_name='us-east-1',
                             config=botocore.config.Config(read_timeout= 300, retries = {"max_attempt": 3}))

        # upload model and invoke model
        response =bedrock.invoke_model(body =json.dumps(body),modelId="meta.llama2-13b-chat-v1")
        
        # get response
        response_content = response.get('body').read()
        response_data = json.loads(response_content)
        print(response_data)
        blog_details = response_data["generation"]
        return blog_details
        
    except Exception as e:
        print('Error generating the blog  {}'.format(str(e)))
        return ""
    
# function for save generated blog in s3 bucket
def save_blog_details_s3(s3_key , s3_bucket , generate_blog):
    s3 = boto3.client("s3")
    
    try :
        pass
        s3.put_object(Bucket =s3_bucket , Key = s3_key , Body = generate_blog)
        print("Blog saved to s3 Bucket ")
    
    except Exception as e:
        print("Error when saving the blog to s3")
    
# function for handle event
def lambda_handler(event , context):
    # TODO Implement
    
    event = json.loads(event["body"])
    blogtopic = event["blog_topic"]
    
    generate_blog = blog_generate_using_bedrock(blogtopic = blogtopic)
    
    if generate_blog:
        current_time = datetime.now().strftime("%H:%M:%S")
        
        s3_key = f"blog-output/{current_time}.txt"
        s3_bucket = "awsbedrockcourse1"
        save_blog_details_s3(s3_key , s3_bucket , generate_blog)
    else:
        print('No blog was generated')
        
    return {
        'stausCode':200,
        'body':json.dumps("Blog Generation is completed")
    }