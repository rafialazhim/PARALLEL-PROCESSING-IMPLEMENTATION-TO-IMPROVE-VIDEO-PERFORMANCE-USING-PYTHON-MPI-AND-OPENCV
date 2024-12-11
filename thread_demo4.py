from mpi4py import MPI
import argparse
import os
import cv2
from CountsPerSec import CountsPerSec
from VideoGet import VideoGet
from VideoShow import VideoShow

def putIterationsPerSec(frame, iterations_per_sec):
    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    return frame

def threadBoth(source=0):
    """
    Separate threads for both getting and showing video frames
    """

    video_getter = VideoGet(source).start()
    video_shower = VideoShow(video_getter.frame).start()
    cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        cps.increment()

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the process
    size = comm.Get_size()  # Total number of processes

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", default="C:/video/video1.mp4",
        help="C:/video/video2.mp4 (default 0).")
    ap.add_argument("--thread", "-t", default="none",
        help="Threading mode: get (video read in its own thread),"
            + " show (video show in its own thread), both"
            + " (video read and video show in their own threads),"
            + " none (default--no multithreading)")
    args = vars(ap.parse_args())

    # If source is a string consisting only of integers, check that it doesn't
    # refer to a file. If it doesn't, assume it's an integer camera ID and
    # convert to int.
    if (
        isinstance(args["source"], str)
        and args["source"].isdigit()
        and not os.path.isfile(args["source"])
    ):
        args["source"] = int(args["source"])
    
    # Distribute tasks based on MPI rank
    if rank == 0:
        # Process 0 could handle video capture
        print(f"Process {rank} handling video capture")
        threadBoth(args["source"])
    elif rank == 1:
        # Process 1 could handle additional tasks, e.g., processing frames or saving to a file
        print(f"Process {rank} handling frame processing or other tasks")
        # Implement additional processing here if needed

    # Optionally, synchronize processes or gather results if needed
    comm.Barrier()

if __name__ == "__main__":
    main()
