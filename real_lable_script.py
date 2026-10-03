import numpy as np
import torch

# get human label.
# sa_t: the segment; physics_seg: the physics information of the segment;
# env: the environment; video_recorder: the recorder for physics rendering.
def get_label(self, sa_t_1, sa_t_2, physics_seg1, physics_seg2, env, video_recorder):
    # get human label
    human_labels = np.zeros(sa_t_1.shape[0])
    for seg_index in range(physics_seg1.shape[0]):
        # render the pairs of segments and save the video
        video_recorder[0].init(env, enabled=True)
        for i in range(physics_seg1[seg_index].shape[0]):
            with env.physics.reset_context():
                env.physics.set_state(physics_seg1[seg_index][i])
            video_recorder[0].record(env)
        video_recorder[0].save('seg1.mp4')

        video_recorder[1].init(env, enabled=True)
        for i in range(physics_seg2[seg_index].shape[0]):
            with env.physics.reset_context():
                env.physics.set_state(physics_seg2[seg_index][i])
            video_recorder[1].record(env)
        video_recorder[1].save('seg2.mp4')

        labeling = True
        # provide labeling instruction and query human for preferences
        while (labeling):
            print("\n")
            print("---------------------------------------------------")
            print("Feedback number:", seg_index)
            # preference:
            # 0: segment 0 is better
            # 1: segment 1 is better
            # other number: hard to judge, skip this pair of segment
            while True:
                # check if it is 0/1/number type preference
                try:
                    rational_label = input("Preference: 0 or 1 or other number")
                    rational_label = int(rational_label)
                    break
                except:
                    print("Wrong label type. Please enter 0/1/other number.\n")

            print("---------------------------------------------------")
            human_labels[seg_index] = rational_label
            labeling = False
    # remove the hard-to-judge pairs of segments
    cancel = np.where((human_labels != 0) & (human_labels != 1))[0]
    human_labels = np.delete(human_labels, cancel, axis=0)
    sa_t_1 = np.delete(sa_t_1, cancel, axis=0)
    sa_t_2 = np.delete(sa_t_2, cancel, axis=0)
    print("valid query number:", len(human_labels))
    return sa_t_1, sa_t_2, human_labels
