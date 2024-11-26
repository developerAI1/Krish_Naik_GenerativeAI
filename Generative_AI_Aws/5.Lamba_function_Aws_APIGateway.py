"""
Description of how to deploy AWS LAmbda function and set API gateway , save output in s3 bucket.

Step 1 - 
    - First of all create lambda function in vs code , copy code and paste in aws lambda function.
    
    - then click on "Deploy"  option showing in above heading.
    

Step 2 -    (Update the version of lambda function or install any new library in aws Lambda function.)

    - first of all create folder with any name for example I created folder with name ("boto3_layer/python")
    
    - create a folder with name 'python'.
        
        a).  inside this 'python' folder install library which you are using in aws lambda function.

        b). command for install new library in boto3.

              *  pip install <package-name> -t python/
        
        c). Then create  a zip file , name can be anything of zip file.
        
    
Step 3 - 

    -- Create a layer.

        a).  Go to Lambda -> layers -> create layer.
        b).  give a name to layer and upload zip file.
        c).  Select python version for example python(3.10,3.11,3.12)
        
    -- Add a layer
    
        a). Go to Lambda -> Functions -> Layers -> Add layer.
        
Step 4 -    (Set Api gateway and Integerate with AWS Lambda function so we can trigger this function.)
    
        a). Go to API Gateway -> APIs -> Create API.
        
        b). Choose an API type like (http , https , etc.).
        
        c). give a API name.
        
        d). Add Route or Create Routes
        
            - Select Method (GET , POST , PUT , DELETE)
            - give api 'end point name'
        
        e). Then click on Inetegerations.
        
            - select integeration type like "lambda function"
            
            - then choose lambda function 
            
        f). if we have multiple server like (test , production , QA) so we can add multiple stages, otherwise give a name of stage and click on deploy.
            
      
Step 5 -        (Create Bucket )
        a). Go to AWS -> Buckets -> Create Bucket 
        
        b). Give a bucket name  then click on create the bucket.
            
            
Step 6 - " Test Api on  POST MAN

    a). pass the parameter which you aregiving in event body in event handler function for example:


            def lambda_handler(event , context):
                # TODO Implement
                
                event = json.loads(event["body"])
                blogtopic = event["blog_topic"]
                
            here blogtopic in parameter to use in api to fetch data from user,
"""     