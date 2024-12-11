from mpi4py import MPI
import argparse
import os
import cv2
from CountsPerSec import CountsPerSec
from VideoShow import VideoShow

def putIterationsPerSec(frame, iterations_per_sec):
    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    return frame

def threadVideoShow(source=0):
    """
    A separate thread for showing video frames
    """
    cap = cv2.VideoCapture(source)
    (grabbed, frame) = cap.read()
    video_shower = VideoShow(frame).start()
    cps = CountsPerSec().start()

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or video_shower.stopped:
            video_shower.stop()
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        cps.increment()

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Rank of the process
    size = comm.Get_size()  # Total number of processes

    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", default="C:/video/video1.mp4",
        help="Path to the video file, or camera index for live feed.")
    ap.add_argument("--thread", "-t", default="none",
        help="Threading mode: get (video read in its own thread),"
             + " show (video show in its own thread), both"
             + " (video read and video show in their own threads),"
             + " none (default--no multithreading)")
    args = vars(ap.parse_args())

    # If source is a string of digits, assume it's a camera index
    if (
        isinstance(args["source"], str)
        and args["source"].isdigit()
        and not os.path.isfile(args["source"])
    ):
        args["source"] = int(args["source"])

    # Assign tasks to processes
    if rank == 0:
        # Process 0 will show video frames
        print(f"Process {rank} handling video display.")
        threadVideoShow(args["source"])
    elif rank == 1:
        # Process 1 could handle additional processing, like frame logging
        print(f"Process {rank} handling additional frame processing.")
        # Implement any additional processing here

    # Ensure synchronization
    comm.Barrier()

if __name__ == "__main__":
    main()
