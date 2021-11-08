from os import path, listdir
import os
import numpy as np
from scipy import stats as st
import math
import json
import librosa
import pickle
import torch
from progress.bar import Bar

class AtariDataset():

    TRAJS_SUBDIR = 'trajectories'
    SCREENS_SUBDIR = 'screens'
    ANNS_SUBDIR = 'annotations'
    ATARI_SUBDIR = 'atari_audio'
    AUDIO_SUBDIR = 'human_audio'
    PRECOMPUTED_HEATMAP = 'gaze_maps'
    PASE_VEC = 'pase_vec'

    def __init__(self, data_path):
        
        '''
            Loads the dataset trajectories into memory. 
            data_path is the root of the dataset (the folder, which contains
            the 'screens' and 'trajectories' folders. 
        '''

        self.trajs_path = path.join(data_path, AtariDataset.TRAJS_SUBDIR)       
        self.screens_path = path.join(data_path, AtariDataset.SCREENS_SUBDIR)
        self.anns_path = path.join(data_path, AtariDataset.ANNS_SUBDIR)
        self.atari_path = path.join(data_path, AtariDataset.ATARI_SUBDIR)
        self.audio_path = path.join(data_path, AtariDataset.AUDIO_SUBDIR)
        self.heatmap_path = path.join(data_path, AtariDataset.PRECOMPUTED_HEATMAP)
        self.pasevec_path = path.join(data_path, AtariDataset.PASE_VEC)

        with open('complete.json') as f:
            self.complete = json.load(f)
        with open('gamekeys.json') as f:
            self.mapping  = json.load(f)
    
        #check that the we have the trajs where expected
        assert path.exists(self.trajs_path)
        
        self.pasevec = self.load_pase()
        self.raw_audio = self.load_raw_audio()
        self.annotations  = self.load_annotations()
        self.heatmap = self.load_heatmap()
        self.trajectories = self.load_trajectories()

        # compute the stats after loading
        self.stats = {}
        for g in self.trajectories.keys():
            self.stats[g] = {}
            nb_games = self.trajectories[g].keys()

            total_frames = sum([len(self.trajectories[g][traj]) for traj in self.trajectories[g]])
            final_scores = [self.trajectories[g][traj][-1]['score'] for traj in self.trajectories[g]]

            self.stats[g]['total_replays'] = len(nb_games)
            self.stats[g]['total_frames'] = total_frames
            self.stats[g]['max_score'] = np.max(final_scores)
            self.stats[g]['min_score'] = np.min(final_scores)
            self.stats[g]['avg_score'] = np.mean(final_scores)
            self.stats[g]['stddev'] = np.std(final_scores)
            self.stats[g]['sem'] = st.sem(final_scores)


    def load_trajectories(self):
        print("Loading trajectories...")        

        trajectories = {}
        for game in listdir(self.trajs_path):
            trajectories[game] = {}
            game_dir = path.join(self.trajs_path, game)
            for traj in listdir(game_dir):
                curr_traj = []
                traj_num = int(traj.split(".")[0])
                f = open(path.join(game_dir, traj))

                # heatmaps for spaceinvaders > 10 are not present
                if traj_num > 10:
                    continue

                # TODO: tidy up lengths 
                f_len = len(f.readlines())
                a_len = len(self.annotations[game][traj_num])
                h_len = len(self.heatmap[game][traj_num])
                diff = max(f_len - h_len, f_len - a_len)

                with open(path.join(game_dir, traj)) as f:
                    traj_num = int(traj.split(".")[0])
                    for i,line in enumerate(f):
                        #first line is the metadata, second is the header
                        if i > 1 and i > diff:
                            #TODO will fix the spacing and True/False/integer in the next replay session
                            #frame,reward,score,terminal, action
                    
                            curr_data = line.rstrip('\n').replace(" ","").split(',')
                            curr_trans = {}
                            curr_trans['frame']    = int(curr_data[0])
                            curr_trans['reward']   = int(curr_data[1])
                            curr_trans['score']    = int(curr_data[2])
                            curr_trans['terminal'] = int(curr_data[3])
                            curr_trans['action']   = int(curr_data[4])
                            curr_trans['ann']      = self.annotations[game][traj_num][i - diff]
                            curr_trans['heatmap']  = self.heatmap[game][traj_num][i - diff]
                            curr_trans['pase']     = self.pasevec[game][traj_num][i - diff]
                            curr_traj.append(curr_trans)
                trajectories[game][int(traj.split('.txt')[0])] = curr_traj
        return trajectories

    def load_annotations(self):
        print("Loading annotations...")
        annotations = {}
        for game in listdir(self.anns_path):
            annotations[game] = {}
            ann_game_dir   = path.join(self.anns_path, game)
            audio_game_dir = path.join(self.audio_path, game)

            for ann in listdir(ann_game_dir):

                # compute total frames and audio length
                key = int(ann.split(".")[0])
                game_folder = path.join(self.screens_path, game)
                #NUM_FRAMES = len(listdir(path.join(game_folder, str(key))))
                wav_file = f"{ann.split('.')[0]}.wav"
                NUM_FRAMES = self.frames_finder(wav_file, game)

                audio_file_path = path.join(audio_game_dir, f"{key}.wav")
                y, sr = librosa.load(audio_file_path, sr=48000)
                SECONDS = librosa.get_duration(y=y, sr=sr)

                anns_file_path = path.join(ann_game_dir, ann)

                curr_ann = [None] * NUM_FRAMES
                json_info = json.load(open(anns_file_path, "r"))

                for frame in range(NUM_FRAMES):
                    data = {}

                    data["word"] = None
                    data["conf"] = 0.0

                    for k in json_info.keys():
                        start = int((json_info[k][0] / SECONDS) * NUM_FRAMES)
                        end   = int((json_info[k][1] / SECONDS) * NUM_FRAMES)
                        word  = json_info[k][2]
                        conf  = json_info[k][3]

                        if start <= frame and frame < end:
                            data["word"] = word
                            data["conf"] = conf

                    curr_ann[frame] = data
                annotations[game][key] = curr_ann
        return annotations

    def load_heatmap(self):
        heatmaps = {}
        for game in listdir(self.heatmap_path):
            if "npy" in game:
                heatmap_game = path.join(self.heatmap_path, game)
                data = np.load(heatmap_game, allow_pickle=True)

                game = game.split(".")[0].split("gaze_")[1]
                if game == "montezumarevenge":
                    game = "revenge"

                heatmaps[game] = data.item()
                #for i in range(0, len(data.item().keys())):
                #    print(game)
                #    print(data.item()[i+1].shape)

        return heatmaps

    def load_pase(self):
        print('Loading pase...')
        pase = {}
        for game in listdir(self.pasevec_path):
            if "npy" in game:
                pase_game = path.join(self.pasevec_path, game)
                data = np.load(pase_game, allow_pickle=True).item()
                game = game.split('_vec.npy')[0]
                pase[game] = data

        return pase

    def frames_finder(self, wav_file: str, game: str):
        name = wav_file.split(".")[0]
        traj_name = self.mapping[game][name]
        if 'R0YWVY6RKQ' in traj_name:
            return -1
        return len(self.complete[traj_name].keys())
    
    def pase_on_file(self, data_path: str, num_frames: int, name: str):
        raw = [None] * num_frames
        y, sr = librosa.load(data_path, sr=48000)
        SECONDS = librosa.get_duration(y=y, sr=sr)
        y = torch.tensor(y).view((1, 1, -1))
        CONST = 16
        AUDIO_FRAME = y.shape[2]

        #sec_per_frame = SECONDS / num_frames
        frame_divide_16 = num_frames / CONST
        divided = int(AUDIO_FRAME / frame_divide_16)

        bar = Bar(f'Traj #{name}', max=num_frames)
        for i in range(num_frames):
            if i % CONST == 0:
                num = i / CONST
                start = int(divided * (num))
                end   = int(divided * (num + 1))

                s = y[:,:,start:end]

                temp = s.numpy().reshape((1, -1))
                assert temp.shape[1] != 0 and temp.shape[1] != 1

            raw[i] = temp
            bar.next()
        bar.finish()
        return raw

    def load_raw_audio(self):
        print('Loading raw audio...')
        raw_pase = {}
        count = 0
        for game in listdir(self.audio_path):
            game_path = path.join(self.audio_path, game)
            final = {}
            for wav_file in os.listdir(game_path):
                wav_path = path.join(game_path, wav_file)

                num_frames = self.frames_finder(wav_file, game)
                if num_frames == -1:
                    continue

                # actually get raw data
                raw = self.pase_on_file(wav_path, num_frames, str(count))
                assert len(raw) != 0

                num = int(wav_file.split('.')[0])
                final[num] = raw
                count += 1

            raw_pase[game] = final

        return raw_pase

    def compile_data(self, dataset_path, game, score_lb=0, score_ub=math.inf, max_nb_transitions=None):

        data = []
        shuffled_trajs = np.array(list(self.trajectories.keys()))
        np.random.shuffle(shuffled_trajs)

        for t in shuffled_trajs:
            st_dir   = path.join(self.screens_path, str(t))
            cur_traj = self.trajectories[t]
            cur_traj_len = len(listdir(st_dir))

            # cut off trajectories with final score beyound the limit
            if not score_lb <= cur_traj[-1]['score'] <= score_ub:
                continue

            #we're here if the trajectory is within lb/ub
            for pid in range(0, cur_traj_len):

                #screens are numbered from 1, transitions from 0
                #TODO change screen numbering from zero during next data replay
                state = preprocess(cv2.imread(path.join(st_dir, str(pid) + '.png'), cv2.IMREAD_GRAYSCALE))

                data.append({'action': get_action_name(cur_traj[pid]['action']),
                             'state':  state,
                             'reward': cur_traj[pid]['reward'],
                             'terminal': cur_traj[pid]['terminal'] == 1,
                             'word': cur_traj[pid]['ann']['word'],
                             'conf': cur_traj[pid]['ann']['conf'],
                             'heatmap': cur_traj[pid]['heatmap'],
                            })

                # if nb_transitions is None, we want the whole dataset limited only by lb and ub
                if max_nb_transitions and len(data) == max_nb_transitions:
                    print("Total frames: %d" % len(dataset))
                    return data

        #we're here if we need all the data
        return data
     
