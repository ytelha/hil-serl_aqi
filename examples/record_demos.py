import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.utils.obs_view import compose_obs_view, resolve_state_slices

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_bool("render", False, "Show a live window of the scene while collecting demos.")
flags.DEFINE_bool("filtered", True, "Keep only successful episodes; false = keep all N episodes.")
flags.DEFINE_enum("peg_shape", None, ["cylinder", "hex", "cuboid"],
                  "Sim only: insertion geometry to record on -- 'cylinder' (easy, "
                  "round, no yaw) or 'hex' (exact tight task). Defaults to the "
                  "experiment config's peg_shape.")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    if FLAGS.peg_shape is not None:
        setattr(config, "peg_shape", FLAGS.peg_shape)
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    # Resolve how the flattened "state" vector maps back to the proprio keys so
    # the live view can label each slice (tcp_pose, tcp_vel, ...).
    state_slices = resolve_state_slices(env, config)

    cv2 = None
    if FLAGS.render:
        import cv2  # local import: only needed for the optional live view
        cv2.namedWindow("record_demos", cv2.WINDOW_NORMAL)

    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    episode_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0

    while (success_count if FLAGS.filtered else episode_count) < success_needed:
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]

        if FLAGS.render:
            frame = env.render()  # RGB scene (front 3/4 monitoring camera)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"successes {success_count}/{success_needed}  return {returns}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frame = compose_obs_view(cv2, frame, next_obs, config.image_keys, state_slices)
            cv2.imshow("record_demos", frame)
            cv2.waitKey(1)
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done:
            episode_count += 1
            if info["succeed"]:
                success_count += 1
            if FLAGS.filtered and not info["succeed"]:
                pass  # discard failed episode
            else:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                pbar.update(1)
            trajectory = []
            returns = 0
            obs, info = env.reset()
            
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    shape = getattr(config, "peg_shape", None)
    subdir = os.path.join("./demo_data", shape or FLAGS.exp_name, "human")
    os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{success_needed}_demos_{uuid}.pkl")
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

    if FLAGS.render and cv2 is not None:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(main)