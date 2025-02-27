import yaml


if __name__ == 'utils_vllm._load_config':
    ACCESS_TOKEN = yaml.load(open('config/huggingface_access_token.yaml'), Loader=yaml.FullLoader)['access_token']