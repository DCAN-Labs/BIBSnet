import boto3
import subprocess

def get_aws_keys():
    """
    Runs `s3info --keys` and extracts the AWS Access Key, Secret Key, and optional Session Token.
    """
    try:
        # Run the command and capture the output
        result = subprocess.run(["s3info", "--keys"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        # Extract keys from output
        keys = result.stdout.strip().split()

        if len(keys) < 2:
            raise ValueError("Unexpected output format from `s3info --keys`. Expected at least Access Key and Secret Key.")

        # Extract values safely
        access_key = keys[0]
        secret_key = keys[1]
        session_token = keys[2] if len(keys) > 2 else None  # Optional third token

        return access_key, secret_key, session_token
    except Exception as e:
        print(f"Error retrieving AWS keys: {e}")
        return None, None, None  # Ensure all three values return safely

def rename_s3_file(bucket_name, old_key, new_key):
    """
    Copies an S3 object to a new key (renaming it) and makes it publicly readable.
    """
    access_key, secret_key, session_token = get_aws_keys()
    
    if not access_key or not secret_key:
        print("Missing AWS credentials. Exiting.")
        return  # Prevents further execution if credentials are missing

    # Initialize Boto3 session
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token  # Include session token if available
    )

    # Create S3 client with custom endpoint
    s3 = session.client('s3', endpoint_url='https://s3.msi.umn.edu')

    # Copy the object
    try:
        s3.copy_object(
            Bucket=bucket_name,
            CopySource={'Bucket': bucket_name, 'Key': old_key},
            Key=new_key,
            ACL='public-read'
        )
        print(f"File copied from {old_key} to {new_key} in bucket {bucket_name}")
    except Exception as e:
        print(f"Error copying S3 file: {e}")

if __name__ == "__main__":
    bucket_name = "bibsnet-data"
    old_key = "bibsnet-latest.tar.gz"
    new_key = "bibsnet-test.tar.gz"

    rename_s3_file(bucket_name, old_key, new_key)
