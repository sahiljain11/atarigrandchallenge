from os import path, listdir
import numpy as np
from scipy import stats as st
import math
import json
import librosa

class AtariDataset():

    TRAJS_SUBDIR = 'trajectories'
    SCREENS_SUBDIR = 'screens'
    ANNS_SUBDIR = 'annotations'
    ATARI_SUBDIR = 'atari_audio'
    AUDIO_SUBDIR = 'human_audio'

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
    
        #check that the we have the trajs where expected
        assert path.exists(self.trajs_path)
        
        self.trajectories = self.load_trajectories()
        self.annotations  = self.load_annotations()

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

        trajectories = {}
        for game in listdir(self.trajs_path):
            trajectories[game] = {}
            game_dir = path.join(self.trajs_path, game)
            for traj in listdir(game_dir):
                curr_traj = []
                with open(path.join(game_dir, traj)) as f:
                    for i,line in enumerate(f):
                        #first line is the metadata, second is the header
                        if i > 1:
                            #TODO will fix the spacing and True/False/integer in the next replay session
                            #frame,reward,score,terminal, action
                    
                            curr_data = line.rstrip('\n').replace(" ","").split(',')
                            curr_trans = {}
                            curr_trans['frame']    = int(curr_data[0])
                            curr_trans['reward']   = int(curr_data[1])
                            curr_trans['score']    = int(curr_data[2])
                            curr_trans['terminal'] = int(curr_data[3])
                            curr_trans['action']   = int(curr_data[4])
                            curr_traj.append(curr_trans)
                trajectories[game][int(traj.split('.txt')[0])] = curr_traj
        return trajectories

    def load_annotations(self):
        annotations = {}
        for game in listdir(self.anns_path):
            annotations[game] = {}
            ann_game_dir   = path.join(self.anns_path, game)
            audio_game_dir = path.join(self.audio_path, game)

            for ann in listdir(ann_game_dir):
                
                # compute total frames and audio length
                key = int(ann.split(".")[0])
                NUM_FRAMES = len(self.trajectories[game][key])

                audio_file_path = path.join(audio_game_dir, f"{key}.wav")
                y, sr = librosa.load(audio_file_path, sr=48000)
                SECONDS = librosa.get_duration(y=y, sr=sr)

                anns_file_path = path.join(ann_game_dir, ann)

                curr_ann = []
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

                    curr_ann.append(data)

                annotations[game][key] = curr_ann
        return annotations
                   

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
                             'word': self.annotations[t][pid]['word'],
                             'conf': self.annotations[t][pid]['conf'],
                            })

                # if nb_transitions is None, we want the whole dataset limited only by lb and ub
                if max_nb_transitions and len(data) == max_nb_transitions:
                    print("Total frames: %d" % len(dataset))
                    return data

        #we're here if we need all the data
        return data
     
