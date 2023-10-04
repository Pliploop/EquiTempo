import yaml

class Config:

    def __init__(self) -> None:
        pass

    def to_dict(self):
        return self.__dict__

    def save(self,save_path):
        
        with open(save_path, 'w') as outfile:
            yaml.dump(self.to_dict(), outfile, default_flow_style=False)
