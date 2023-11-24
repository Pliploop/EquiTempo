from src.model.model import Siamese
from config.full import GlobalConfig
from config.preprocessing import PreprocessingConfig
import torch
from src.data_loading.datasets import EvaluationDataset
import numpy as np
from src.evaluation.metrics import *
from src.data_loading.preprocessing import *
import librosa
import torchaudio.transforms as T
from tqdm import tqdm


class NoFineTuneInfer:
    
    def __init__(self, model = None, model_path = None, known_bpm = 120, globalconfig_path = None, device = "cpu") -> None:
        if model:
            self.model = model
        else:
            self.model = Siamese()
            
            
        self.device = device
        if model_path :
            self.load_model(model_path)
            
            
        self.sr = 44100
        self.known_bpm = known_bpm
        
        if globalconfig_path:
            self.globalconfig = GlobalConfig().from_yaml(globalconfig_path)
        else:
            self.globalconfig = GlobalConfig()
            
        
        self.melgram = T.MelSpectrogram(
            sample_rate=44100,
            f_min=30,
            f_max=17000,
            n_mels=81,
            n_fft=2048,
            win_length=2048,
            hop_length=441,
            power=1,
        )
        self.preprocessing_config= PreprocessingConfig(dict = self.globalconfig.preprocessing_config)
        self.click_mel, self.click_track = self.generate_click()
        
        
    
    def generate_click(self, bpm = None):
        if bpm:
            bpm = bpm
        else:
            bpm = self.known_bpm
            
        bps = bpm/60
        
        intraclick_s = (1/bps)
        click_track = np.zeros((self.preprocessing_config.len_audio_n))
        
        audio_len = self.preprocessing_config.len_audio_s
        
        print('intraclick_s:', intraclick_s)
        print("audio_len_s:", audio_len)
        
        times = [k * intraclick_s for k in range(0,int(np.floor(audio_len/intraclick_s))+1)]
        
        print('click times:', times)
        
        click_track_truncated = librosa.clicks(times=times, sr = self.sr)
        click_track[:len(click_track_truncated)] = click_track_truncated
        
        click_track = torch.from_numpy(click_track).unsqueeze(0).float()
        
        return logcomp(self.melgram(click_track)), click_track
        
        
        
    
    def load_model(self, model_path):
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["gen_state_dict"], strict=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def tempo_from_reg_out(self,reg_out, reg_out_known):
        
        print(reg_out.squeeze())
        print(reg_out_known.squeeze())
        
        ratio_1 = reg_out/reg_out_known
        estimated_tempo = ratio_1 * self.known_bpm
        return estimated_tempo.long()
        
        
        
        
        
class NoFineTuneEval:
    
    def __init__(self, dataset_name = "gtzan", global_config = GlobalConfig(), known_bpm = 120, model_path = None) -> None:
        self.dataset = EvaluationDataset(dataset_name=dataset_name, global_config=global_config, stretch=False)
        self.dataloader = self.dataset.create_dataloader()
        self.infer = NoFineTuneInfer(known_bpm=known_bpm, model_path=model_path)
        
    def evaluate(self):
        preds = []
        truths = []
        with torch.no_grad():
            for item_dict in tqdm(self.dataloader):
                audio = item_dict["audio"]
                tempo = item_dict["tempo"]
                rf = item_dict["rf"]
                _, regression_out = self.infer.model(audio)
                _, regression_out_known = self.infer.model(torch.cat([self.infer.click_mel] * audio.shape[0], dim=0))
                
                # preds.append(np.argmax(classification_pred.cpu().numpy(), axis=1))
                estimated_pred = self.infer.tempo_from_reg_out(regression_out, regression_out_known)
                preds.append(estimated_pred.squeeze().cpu().numpy())
                truths.append((tempo * rf).squeeze().long().cpu().numpy())
                print("preds: ",preds[-1])
                print("truths:", truths[-1])
                print("acc_1: ", compute_accuracy_1(truths[-1],preds[-1]))
                print("acc_2: ", compute_accuracy_2(truths[-1],preds[-1]))

        # flatten arrays
        preds = [item for sublist in preds for item in sublist]
        truths = [item for sublist in truths for item in sublist]

        # save predictions, truths, results, and accuracies as .json files
        accuracy_1, results_1 = compute_accuracy_1(truths, preds)
        accuracy_2, results_2 = compute_accuracy_2(truths, preds)

        results = {
            # "preds": preds,
            # "truths": truths,
            "accuracy_1": f"{accuracy_1:.4f}",
            "accuracy_2": f"{accuracy_2:.4f}",
        }
        
        print(results)