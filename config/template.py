import yaml
import json


class Config:

    def __init__(self, dict =None) -> None:
        if dict is not None:
            for k, v in dict.items():
                setattr(self, k, v)

    def to_dict(self):
        return self.__dict__

    def save(self, save_path):

        with open(save_path, 'w') as outfile:
            yaml.dump(self.to_dict(), outfile, default_flow_style=False)

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)