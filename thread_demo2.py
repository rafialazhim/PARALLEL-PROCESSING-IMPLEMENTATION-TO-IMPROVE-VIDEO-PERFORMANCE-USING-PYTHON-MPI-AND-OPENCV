from mpi4py import MPI
import argparse
import os
import cv2
from CountsPerSec import CountsPerSec
from VideoGet import VideoGet

def putIterationsPerSec(frame, iterations_per_sec):
    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    return frame

def threadVideoGet(source=0):
    """
    A separate thread for getting video and webcam frames
    """
    video_getter = VideoGet(source).start()
    cps = CountsPerSec().start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the process
    size = comm.Get_size()  # Total number of processes

    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", default="C:/video/video1.mp4",
        help="C:/video/video2.mp4 (default 0).")
    ap.add_argument("--thread", "-t", default="none",
        help="Threading mode: get (video read in its own thread),"
             + " show (video show in its own thread), both"
             + " (video read and video show in their own threads),"
             + " none (default--no multithreading)")
    args = vars(ap.parse_args())

    # If source is a string with numbers, check if it's a file or camera ID
    if (
        isinstance(args["source"], str)
        and args["source"].isdigit()
        and not os.path.isfile(args["source"])
    ):
        args["source"] = int(args["source"])

    # Distribute tasks based on MPI rank
    if rank == 0:
        # Process 0 will handle video capturing
        print(f"Process {rank} handling video capture")
        threadVideoGet(args["source"])
    elif rank == 1:
        # Process 1 could handle additional processing, e.g., frame analysis or logging
        print(f"Process {rank} handling additional frame processing or logging")
        # Insert additional processing functions as needed

    # Optionally, synchronize or finalize tasks across processes
    comm.Barrier()

if __name__ == "__main__":
    main()
