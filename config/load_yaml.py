import yaml

# 读取yaml
def load_config():
	with open('config/config.yaml', 'r', encoding='utf-8') as f:
		config = yaml.load(f.read(),Loader=yaml.Loader)
	return config
