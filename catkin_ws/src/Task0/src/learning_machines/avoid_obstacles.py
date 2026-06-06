from collections import deque
from pathlib import Path
from datetime import datetime
import cv2
from .constants import *
from robobo_interface import IRobobo, SimulationRobobo, SoundEmotion, Emotion
from .visualize_metrics import SimMetrics

def front_blocked(IR_list):
    return(
        IR_list[IR_indices["FRONT_R"]] > IR_THRESHOLD or
        IR_list[IR_indices["FRONT_L"]] > IR_THRESHOLD or
        IR_list[IR_indices["FRONT_C"]] > IR_THRESHOLD
    )

def side_clear_score(history: deque, side: str):
    
    if side == "left":
        indices = LEFT_INDICES
    elif side == "right":
        indices = RIGHT_INDICES
    
    score = 0
    for weight, IR_list in enumerate(history, start = 1):
        step_blocked = any([IR_list[i] > IR_THRESHOLD for i in indices])
        score += weight * (1 if not step_blocked else -1)
    return score


def avoid_obstacles(rob: IRobobo, max_iter: int = 100, n_runs: int = 5): 

    metrics = SimMetrics(label = "Simulation" if isinstance(rob, SimulationRobobo) else "Hardware")
    
    # Setup video writer for first run
    video_writer = None
    # Navigate up from: catkin_ws/src/learning_machines/src/learning_machines/avoid_obstacles.py to catkin_ws/../results
    results_dir = Path(__file__).parent.parent.parent.parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"Results directory: {results_dir}")
    print(f"Results directory exists: {results_dir.exists()}")

    for run in range(n_runs):

        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
            
        ir_history = deque(maxlen = HISTORY_LEN)

        for step in range(max_iter):
            irs = rob.read_irs()
            metrics.record(irs)
            ir_history.append(irs)
            
            # Capture frame for video (first run only)
            if run == 0 and video_writer is not None:
                try:
                    # Capture bird's eye view from CoppeliaSim scene at high resolution
                    frame, resolution = rob._sim.getViewImage(rob._sim.handle_scene, 2048, 2048)
                    if frame:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        # Resize down to 1024x1024
                        frame = cv2.resize(frame, (1024, 1024))
                    else:
                        frame = rob.read_image_front()
                except:
                    frame = rob.read_image_front()
                video_writer.write(frame)

            print(f"[step: {step}]"
                f"FrontC: {irs[IR_indices['FRONT_C']]}"
                f"FrontL: {irs[IR_indices['FRONT_L']]}"
                f"FrontLL: {irs[IR_indices['FRONT_LL']]}"
                f"FrontR: {irs[IR_indices['FRONT_R']]}"
                f"FrontRR: {irs[IR_indices['FRONT_RR']]}"
                f"BACKC: {irs[IR_indices['BACK_C']]}"
                f"BACKL: {irs[IR_indices['BACK_L']]}"
                f"BACKR: {irs[IR_indices['BACK_R']]}"
            )

            if front_blocked(irs):
                left_score = side_clear_score(ir_history, "left")
                right_score = side_clear_score(ir_history, "right")

                print(
                    f"OBSTACLE DETECTED"
                    f"left_score = {left_score:.1f}   right_score = {right_score:.1f}"
                )
                
                # Show emotion and play sound
                rob.set_emotion(Emotion.SURPRISED)
                rob.play_emotion_sound(SoundEmotion.APPROVE)
                print("ROBOT: I found an obstacle! I will turn around!")

                # Turn around (180 degrees)
                print("OBSTACLE - TURNING AROUND")
                rob.move_blocking(-TURN_SPEED, TURN_SPEED, TURN_MS * 2)  # Double time for 180°
            else:
                # Clear path, moving forwards
                rob.move_blocking(DRIVE_SPEED, DRIVE_SPEED, DRIVE_MS)

        metrics.end_run()
        
        # Release video writer after first run
        if run == 0 and video_writer is not None:
            video_writer.release()
            print("Video saved successfully!")

        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()                  

    metrics.plot_runs()                          
    print(metrics.run_summaries)                 

    return metrics                                 