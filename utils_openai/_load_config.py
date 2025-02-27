import yaml


if __name__ == 'utils_openai._load_config':
    API_KEY =  yaml.load(open('config/openai_api_key.yaml'), Loader=yaml.FullLoader)['api_key']

    PRICING = {
        'o3-mini-2025-01-31': {'input_token_price': 1.10/1e6, 'output_token_price': 4.40/1e6},
        'gpt-4o-2024-08-06': {'input_token_price': 2.50/1e6, 'output_token_price': 10.00/1e6}
    }